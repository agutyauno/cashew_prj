import cv2
import time
import os
import json
import threading
import queue
import datetime
import numpy as np
from ultralytics import YOLO

# Thiết lập môi trường hiển thị cho Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

# --- TỰ ĐỘNG PHÁT HIỆN MÔI TRƯỜNG & MOCK SERVO ---
try:
    from gpiozero import AngularServo
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    class AngularServo:
        def __init__(self, pin, min_pulse_width=0.0005, max_pulse_width=0.0026):
            self.pin = pin
            self._angle = 0
            
        @property
        def angle(self):
            return self._angle
            
        @angle.setter
        def angle(self, val):
            self._angle = val
            print(f"[MÔ PHỎNG SERVO Pin {self.pin}] -> Góc: {val}°")

# --- LỚP THEO DÕI HẠT ĐIỀU (CENTROID TRACKER) ---
class CentroidTracker:
    def __init__(self, max_disappeared=8):
        self.next_id = 1
        self.objects = {}       # id -> (cx, cy, label)
        self.disappeared = {}   # id -> số frame biến mất
        self.max_disappeared = max_disappeared

    def register(self, centroid, label):
        self.objects[self.next_id] = (centroid[0], centroid[1], label)
        self.disappeared[self.next_id] = 0
        assigned_id = self.next_id
        self.next_id += 1
        return assigned_id

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_labels = []
        for i, (x1, y1, x2, y2, label) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_labels.append(label)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_labels[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([ (v[0], v[1]) for v in self.objects.values() ])

            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col][0], input_centroids[col][1], input_labels[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], input_labels[col])

        return self.objects

# --- LUỒNG ĐIỀU KHIỂN SERVO PHI NGHẼN (SERVO WORKER) ---
class ServoWorker(threading.Thread):
    def __init__(self, pin, min_pulse, max_pulse, default_angle, active_angle, hold_time, name=""):
        super().__init__(daemon=True)
        self.pin = pin
        self.default_angle = default_angle
        self.active_angle = active_angle
        self.hold_time = hold_time
        self.name_tag = name
        
        self.servo = AngularServo(pin, min_pulse_width=min_pulse, max_pulse_width=max_pulse)
        self.servo.angle = default_angle
        # Ngắt xung ngay sau khi khởi tạo để tránh giật lúc chờ hạt đầu tiên
        time.sleep(0.5)
        self.servo.value = None
        
        self.queue = queue.Queue()
        self.is_running = True
        
    def queue_actuation(self, trigger_time):
        self.queue.put(trigger_time)
        
    def run(self):
        print(f"Luồng điều khiển Servo {self.name_tag} (Pin {self.pin}) đã khởi chạy.")
        while self.is_running:
            try:
                trigger_time = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            now = time.time()
            sleep_duration = trigger_time - now
            if sleep_duration > 0:
                time.sleep(sleep_duration)
                
            try:
                print(f"[KÍCH HOẠT] Servo {self.name_tag} (Pin {self.pin}) -> Góc chạy {self.active_angle}°")
                self.servo.angle = self.active_angle
                time.sleep(self.hold_time)
                self.servo.value = None

                print(f"[HOÀN TÁC] Servo {self.name_tag} (Pin {self.pin}) -> Về góc chờ {self.default_angle}°")
                self.servo.angle = self.default_angle
                
                # Chờ một khoảng ngắn để servo hoàn tất việc di chuyển về vị trí mặc định
                time.sleep(0.4)
                # Ngắt xung PWM để tránh hiện tượng giật (jitter) khi ở trạng thái nghỉ
                self.servo.value = None
                
            except Exception as e:
                print(f"Lỗi điều khiển Servo {self.name_tag}: {e}")
                
            self.queue.task_done()

# --- HỆ THỐNG PHÂN LOẠI CHÍNH ---
class CashewSortingSystem:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.load_config()
        
        # Thống kê hệ thống
        self.stats = {
            "total_a": 0,
            "total_b": 0,
            "good": 0,
            "bad": 0
        }
        
        # Hàng đợi FIFO để lưu trạng thái hạt từ Vùng A sang Vùng B
        self.cashew_fifo = []
        self.fifo_lock = threading.Lock()
        
        # Bộ đếm ID để ghi nhận lịch sử xử lý hạt
        self.processed_count = 0
        
        # Khởi tạo các bộ theo dõi centroid
        self.tracker_a = CentroidTracker(max_disappeared=6)
        self.tracker_b = CentroidTracker(max_disappeared=6)
        
        # Các ID đã kích hoạt gạt/lật để tránh double trigger
        self.triggered_a = set()
        self.triggered_b = set()
        
        self.init_servos()
        self.init_model()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        print("Đã nạp file cấu hình config.json thành công (Băng tải PHẢI -> TRÁI).")

    def init_servos(self):
        s1_cfg = self.cfg["servo_1"]
        s2_cfg = self.cfg["servo_2"]
        
        self.servo1_worker = ServoWorker(
            pin=s1_cfg["pin"],
            min_pulse=s1_cfg["min_pulse_width"],
            max_pulse=s1_cfg["max_pulse_width"],
            default_angle=s1_cfg["default_angle"],
            active_angle=s1_cfg["active_angle"],
            hold_time=s1_cfg["hold_time"],
            name="Lật mặt (Servo 1)"
        )
        
        self.servo2_worker = ServoWorker(
            pin=s2_cfg["pin"],
            min_pulse=s2_cfg["min_pulse_width"],
            max_pulse=s2_cfg["max_pulse_width"],
            default_angle=s2_cfg["default_angle"],
            active_angle=s2_cfg["active_angle"],
            hold_time=s2_cfg["hold_time"],
            name="Gạt hạt xấu (Servo 2)"
        )
        
        self.servo1_worker.start()
        self.servo2_worker.start()

    def init_model(self):
        m_cfg = self.cfg["model"]
        print(f"Đang nạp mô hình AI từ: {m_cfg['path']}...")
        try:
            self.model = YOLO(m_cfg["path"], task="detect")
            print("Đã nạp mô hình AI thành công!")
        except Exception as e:
            print(f"Lỗi nạp mô hình AI: {e}")
            raise e

    def log_result(self, cashew_num, side_a, side_b, result):
        log_cfg = self.cfg["logging"]
        log_msg = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [HẠT #{cashew_num}] Mặt A: {side_a.upper()} | Mặt B: {side_b.upper()} -> KẾT QUẢ: {result.upper()}\n"
        
        with open(log_cfg["log_file"], "a", encoding="utf-8") as f:
            f.write(log_msg)
        print(f"{log_msg.strip()}")

    def cleanup_old_fifo(self):
        # Dọn dẹp các hạt bị kẹt trong FIFO quá 15 giây (phòng trường hợp hạt bị trượt khỏi băng tải)
        now = time.time()
        with self.fifo_lock:
            original_len = len(self.cashew_fifo)
            self.cashew_fifo = [item for item in self.cashew_fifo if now - item["time_a"] < 15.0]
            diff = original_len - len(self.cashew_fifo)
            if diff > 0:
                print(f"Đã dọn {diff} hạt quá hạn khỏi hàng đợi FIFO.")

    def run(self):
        cam_cfg = self.cfg["camera"]
        cap = cv2.VideoCapture(cam_cfg["device_id"], cv2.CAP_V4L2)
        
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Không thể kết nối với Camera.")
            return

        print("Hệ thống đã sẵn sàng. Đang chạy vòng lặp giám sát...")
        prev_time = time.time()
        
        reg_cfg = self.cfg["regions"]
        x_trigger_a = reg_cfg["region_a"]["line_trigger_x"]
        x_trigger_b = reg_cfg["region_b"]["line_trigger_x"]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            input_frame = frame.copy()
            current_time = time.time()
            height, width = frame.shape[:2]

            # 1. Nhận diện YOLO NCNN
            results = self.model(input_frame, imgsz=self.cfg["model"]["imgsz"], conf=self.cfg["model"]["conf_threshold"], verbose=False)
            
            boxes_a = []
            boxes_b = []
            annotated_frame = frame.copy()
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().item())
                    label = self.model.names[cls_id]
                    
                    cx = int((x1 + x2) / 2.0)
                    
                    # Phân loại hộp giới hạn vào Vùng A hoặc Vùng B dựa trên trục X (Phải -> Trái)
                    # Vùng A (Bên Phải): x_min <= cx <= x_max
                    # Vùng B (Bên Trái): x_min <= cx <= x_max
                    if reg_cfg["region_a"]["x_min"] <= cx <= reg_cfg["region_a"]["x_max"]:
                        boxes_a.append([x1, y1, x2, y2, label])
                    elif reg_cfg["region_b"]["x_min"] <= cx <= reg_cfg["region_b"]["x_max"]:
                        boxes_b.append([x1, y1, x2, y2, label])
            
            # 2. Cập nhật các bộ theo dõi centroid
            objects_a = self.tracker_a.update(boxes_a)
            objects_b = self.tracker_b.update(boxes_b)
            
            # 3. Xử lý logic tại VÙNG A (Mặt A bên phải & Servo 1 - Lật mặt)
            # Hạt điều đi từ Phải sang Trái, nên X giảm dần. 
            # Kích hoạt khi hạt đi VƯỢT QUA vạch kích hoạt về phía bên TRÁI (cx <= x_trigger_a)
            for obj_id, (cx, cy, label) in objects_a.items():
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(annotated_frame, f"ID_A:{obj_id} ({label})", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                if cx <= x_trigger_a and obj_id not in self.triggered_a:
                    self.triggered_a.add(obj_id)
                    self.stats["total_a"] += 1
                    
                    with self.fifo_lock:
                        self.cashew_fifo.append({
                            "side_a": label,
                            "time_a": current_time,
                            "side_b": None
                        })
                    
                    trigger_t1 = current_time + self.cfg["servo_1"]["delay"]
                    self.servo1_worker.queue_actuation(trigger_t1)
                    print(f"VÙNG A (Bên Phải): Hạt A_{obj_id} ({label}) vượt vạch trigger x={x_trigger_a}. Lên lịch lật mặt sau {self.cfg['servo_1']['delay']}s.")
            
            # 4. Xử lý logic tại VÙNG B (Mặt B bên trái & Servo 2 - Gạt bỏ)
            # Kích hoạt khi hạt đi VƯỢT QUA vạch kích hoạt về phía bên TRÁI (cx <= x_trigger_b)
            for obj_id, (cx, cy, label) in objects_b.items():
                cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)
                cv2.putText(annotated_frame, f"ID_B:{obj_id} ({label})", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                if cx <= x_trigger_b and obj_id not in self.triggered_b:
                    self.triggered_b.add(obj_id)
                    self.stats["total_b"] += 1
                    
                    matched_cashew = None
                    with self.fifo_lock:
                        for item in self.cashew_fifo:
                            if item["side_b"] is None:
                                item["side_b"] = label
                                matched_cashew = item
                                break
                    
                    self.processed_count += 1
                    side_a = matched_cashew["side_a"] if matched_cashew else "unknown (mất dấu mặt A)"
                    side_b = label
                    
                    if side_a == "xau" or side_b == "xau":
                        self.stats["bad"] += 1
                        result_str = "xau"
                        
                        trigger_t2 = current_time + self.cfg["servo_2"]["delay"]
                        self.servo2_worker.queue_actuation(trigger_t2)
                        print(f"VÙNG B (Bên Trái): Hạt B_{obj_id} bị loại! (Mặt A: {side_a}, Mặt B: {side_b}). Lên lịch gạt sau {self.cfg['servo_2']['delay']}s.")
                    else:
                        self.stats["good"] += 1
                        result_str = "dep"
                        print(f"VÙNG B (Bên Trái): Hạt B_{obj_id} ĐẠT YÊU CẦU (Mặt A: {side_a}, Mặt B: {side_b}).")
                        
                    self.log_result(self.processed_count, side_a, side_b, result_str)

            if self.processed_count % 10 == 0:
                self.cleanup_old_fifo()

            # --- VẼ GIAO DIỆN HIỂN THỊ (GUI) ---
            # Vẽ các đường ranh giới dọc và vạch kích hoạt đứng dọc màn hình
            # Vùng A (Màu xanh dương đậm ở bên PHẢI)
            cv2.line(annotated_frame, (reg_cfg["region_a"]["x_min"], 0), (reg_cfg["region_a"]["x_min"], cam_cfg["height"]), (255, 120, 0), 1)
            cv2.line(annotated_frame, (reg_cfg["region_a"]["x_max"], 0), (reg_cfg["region_a"]["x_max"], cam_cfg["height"]), (255, 120, 0), 1)
            cv2.line(annotated_frame, (x_trigger_a, 0), (x_trigger_a, cam_cfg["height"]), (255, 0, 0), 2) # Vạch đứng kích hoạt Servo 1 (Xanh dương nét dày)
            cv2.putText(annotated_frame, "VUNG A (Ben Phai - Mat A)", (reg_cfg["region_a"]["x_min"] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
            cv2.putText(annotated_frame, "TRIGGER 1 (SERVO 1)", (x_trigger_a - 150, cam_cfg["height"] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

            # Vùng B (Màu hồng sen đậm ở bên TRÁI)
            cv2.line(annotated_frame, (reg_cfg["region_b"]["x_min"], 0), (reg_cfg["region_b"]["x_min"], cam_cfg["height"]), (180, 0, 180), 1)
            cv2.line(annotated_frame, (reg_cfg["region_b"]["x_max"], 0), (reg_cfg["region_b"]["x_max"], cam_cfg["height"]), (180, 0, 180), 1)
            cv2.line(annotated_frame, (x_trigger_b, 0), (x_trigger_b, cam_cfg["height"]), (0, 0, 255), 2) # Vạch đứng kích hoạt Gạt (Màu đỏ nét dày)
            cv2.putText(annotated_frame, "VUNG B (Ben Trai - Mat B)", (reg_cfg["region_b"]["x_min"] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 180), 1)
            cv2.putText(annotated_frame, "TRIGGER 2 (SERVO 2)", (x_trigger_b - 150, cam_cfg["height"] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # Hướng mũi tên di chuyển băng tải (phải qua trái)
            cv2.arrowedLine(annotated_frame, (cam_cfg["width"] - 50, cam_cfg["height"] - 30), (50, cam_cfg["height"] - 30), (0, 255, 255), 2, tipLength=0.05)
            cv2.putText(annotated_frame, "HUONG BANG CHUYEN", (cam_cfg["width"] // 2 - 80, cam_cfg["height"] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Vẽ bounding boxes hiện thời
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().item())
                    label = self.model.names[cls_id]
                    conf = box.conf[0].cpu().item()
                    
                    color = (0, 255, 0) if label == "dep" else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Bảng thống kê
            overlay = annotated_frame.copy()
            # Đặt bảng thống kê ở giữa phía trên để không cản trở góc nhìn trái/phải
            cv2.rectangle(overlay, (width // 2 - 125, 10), (width // 2 + 125, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            fps = 1.0 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0.0
            prev_time = time.time()
            
            cv2.putText(annotated_frame, "MAY PHAN LOAI HAT DIEU", (width // 2 - 110, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (width // 2 - 110, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Da kiem (A): {self.stats['total_a']} | FIFO: {len(self.cashew_fifo)}", (width // 2 - 110, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Dat (Dep): {self.stats['good']}", (width // 2 - 110, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Loai (Xau): {self.stats['bad']}", (width // 2 - 110, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            cv2.imshow("He Thong Phan Loai Hat Dieu AI", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Đang dừng hệ thống và dọn dẹp tài nguyên...")
                break
            if cv2.waitKey(1) & 0xFF == ord('s'):
                self.servo2_worker.run()
                break
        cap.release()
        cv2.destroyAllWindows()
        
        self.servo1_worker.is_running = False
        self.servo2_worker.is_running = False

if __name__ == "__main__":
    system = CashewSortingSystem()
    system.run()
