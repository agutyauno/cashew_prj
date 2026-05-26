import cv2
import time
import os
import json
import threading
import queue
import datetime
import numpy as np
from ultralytics import YOLO

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
            # In ra console để người dùng quan sát khi chạy thử trên máy tính
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
        # rects: danh sách dạng [x1, y1, x2, y2, label]
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

            # Tính khoảng cách Euclidean
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

# --- LUỒNG ĐIỀU KHIỂN SERVO (SERVO WORKER) ---
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
        
        self.queue = queue.Queue()
        self.is_running = True
        
    def queue_actuation(self, trigger_time):
        self.queue.put(trigger_time)
        
    def run(self):
        print(f"Luồng điều khiển Servo {self.name_tag} (Pin {self.pin}) đã khởi chạy.")
        while self.is_running:
            try:
                # Lấy lịch trình kích hoạt tiếp theo từ hàng đợi
                trigger_time = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            # Chờ tới đúng thời điểm kích hoạt
            now = time.time()
            sleep_duration = trigger_time - now
            if sleep_duration > 0:
                time.sleep(sleep_duration)
                
            # Thực hiện chu kỳ gạt/lật
            try:
                print(f"[KÍCH HOẠT] Servo {self.name_tag} (Pin {self.pin}) -> Góc chạy {self.active_angle}°")
                self.servo.angle = self.active_angle
                time.sleep(self.hold_time)
                
                print(f"[HOÀN TÁC] Servo {self.name_tag} (Pin {self.pin}) -> Về góc chờ {self.default_angle}°")
                self.servo.angle = self.default_angle
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
        # Định dạng: {'side_a': 'dep'/'xau', 'time_a': timestamp, 'side_b': None}
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
        
        # Khởi động luồng điều khiển servo
        self.init_servos()
        
        # Khởi động mô hình YOLO NCNN
        self.init_model()

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        print("Đã nạp file cấu hình config.json thành công.")

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
        
        # Ghi log ra file
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
        
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Không thể kết nối với Camera.")
            return

        print("Hệ thống đã sẵn sàng. Đang chạy vòng lặp giám sát...")
        prev_time = time.time()
        
        # Đọc các thông số vùng và kích hoạt từ cấu hình
        reg_cfg = self.cfg["regions"]
        y_trigger_a = reg_cfg["region_a"]["line_trigger_y"]
        y_trigger_b = reg_cfg["region_b"]["line_trigger_y"]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            input_frame = frame.copy()
            current_time = time.time()

            # 1. Nhận diện YOLO NCNN
            results = self.model(input_frame, imgsz=self.cfg["model"]["imgsz"], conf=self.cfg["model"]["conf_threshold"], verbose=False)
            
            # Khởi tạo danh sách hộp giới hạn theo phân vùng
            boxes_a = []
            boxes_b = []
            
            # Vẽ các đối tượng nhận diện ban đầu lên khung hình
            annotated_frame = frame.copy()
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls_id = int(box.cls[0].cpu().item())
                    label = self.model.names[cls_id] # "dep" hoặc "xau"
                    
                    cy = int((y1 + y2) / 2.0)
                    
                    # Phân loại hộp giới hạn vào Vùng A hoặc Vùng B dựa trên tọa độ y
                    if reg_cfg["region_a"]["y_min"] <= cy <= reg_cfg["region_a"]["y_max"]:
                        boxes_a.append([x1, y1, x2, y2, label])
                    elif reg_cfg["region_b"]["y_min"] <= cy <= reg_cfg["region_b"]["y_max"]:
                        boxes_b.append([x1, y1, x2, y2, label])
            
            # 2. Cập nhật các bộ theo dõi centroid
            objects_a = self.tracker_a.update(boxes_a)
            objects_b = self.tracker_b.update(boxes_b)
            
            # 3. Xử lý logic tại VÙNG A (Mặt A & Servo 1 - Lật mặt)
            for obj_id, (cx, cy, label) in objects_a.items():
                # Vẽ tâm hạt và ID
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)
                cv2.putText(annotated_frame, f"ID_A:{obj_id} ({label})", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Điều kiện kích hoạt Servo 1: Vượt qua vạch kích hoạt và chưa từng kích hoạt trước đó
                if cy >= y_trigger_a and obj_id not in self.triggered_a:
                    self.triggered_a.add(obj_id)
                    self.stats["total_a"] += 1
                    
                    # Đưa hạt vào hàng đợi FIFO
                    with self.fifo_lock:
                        self.cashew_fifo.append({
                            "side_a": label,
                            "time_a": current_time,
                            "side_b": None
                        })
                    
                    # Lên lịch kích hoạt Servo 1 lật mặt
                    trigger_t1 = current_time + self.cfg["servo_1"]["delay"]
                    self.servo1_worker.queue_actuation(trigger_t1)
                    print(f"👉 VÙNG A: Hạt A_{obj_id} ({label}) vượt vạch. Lên lịch lật mặt sau {self.cfg['servo_1']['delay']}s.")
            
            # 4. Xử lý logic tại VÙNG B (Mặt B & Servo 2 - Gạt bỏ hạt lỗi)
            for obj_id, (cx, cy, label) in objects_b.items():
                # Vẽ tâm hạt và ID
                cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)
                cv2.putText(annotated_frame, f"ID_B:{obj_id} ({label})", (cx - 20, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                
                # Điều kiện kích hoạt kiểm tra cuối và gạt: Vượt qua vạch kích hoạt và chưa từng kiểm tra trước đó
                if cy >= y_trigger_b and obj_id not in self.triggered_b:
                    self.triggered_b.add(obj_id)
                    self.stats["total_b"] += 1
                    
                    # Lấy thông tin từ FIFO để so khớp 2 mặt
                    matched_cashew = None
                    with self.fifo_lock:
                        # Tìm hạt lâu nhất chưa được quét mặt B
                        for item in self.cashew_fifo:
                            if item["side_b"] is None:
                                item["side_b"] = label
                                matched_cashew = item
                                break
                    
                    # Ghi nhận kết quả cuối cùng
                    self.processed_count += 1
                    side_a = matched_cashew["side_a"] if matched_cashew else "unknown (mất dấu mặt A)"
                    side_b = label
                    
                    # Quy tắc quyết định: Hạt xấu nếu mặt A HOẶC mặt B xấu
                    if side_a == "xau" or side_b == "xau":
                        self.stats["bad"] += 1
                        result_str = "xau"
                        
                        # Lên lịch kích hoạt Servo 2 gạt bỏ
                        trigger_t2 = current_time + self.cfg["servo_2"]["delay"]
                        self.servo2_worker.queue_actuation(trigger_t2)
                        print(f"VÙNG B: Hạt B_{obj_id} bị loại! (Mặt A: {side_a}, Mặt B: {side_b}). Lên lịch gạt sau {self.cfg['servo_2']['delay']}s.")
                    else:
                        self.stats["good"] += 1
                        result_str = "dep"
                        print(f"VÙNG B: Hạt B_{obj_id} ĐẠT YÊU CẦU (Mặt A: {side_a}, Mặt B: {side_b}).")
                        
                    # Ghi kết quả vào log
                    self.log_result(self.processed_count, side_a, side_b, result_str)

            # Dọn dẹp các hạt bị kẹt trong FIFO (tránh tràn bộ nhớ)
            if self.processed_count % 10 == 0:
                self.cleanup_old_fifo()

            # --- VẼ GIAO DIỆN HIỂN THỊ (GUI) ---
            # 1. Vẽ các đường phân chia vùng và vạch kích hoạt
            # Vùng A (Màu xanh dương đậm)
            cv2.line(annotated_frame, (0, reg_cfg["region_a"]["y_min"]), (cam_cfg["width"], reg_cfg["region_a"]["y_min"]), (255, 120, 0), 1)
            cv2.line(annotated_frame, (0, reg_cfg["region_a"]["y_max"]), (cam_cfg["width"], reg_cfg["region_a"]["y_max"]), (255, 120, 0), 1)
            cv2.line(annotated_frame, (0, y_trigger_a), (cam_cfg["width"], y_trigger_a), (255, 0, 0), 2) # Vạch kích hoạt Servo 1 (Xanh dương nét dày)
            cv2.putText(annotated_frame, "VUNG A (Kiem tra Mat A)", (15, reg_cfg["region_a"]["y_min"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 1)
            cv2.putText(annotated_frame, "VACH KICH HOAT (SERVO 1)", (15, y_trigger_a - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Vùng B (Màu hồng sen đậm)
            cv2.line(annotated_frame, (0, reg_cfg["region_b"]["y_min"]), (cam_cfg["width"], reg_cfg["region_b"]["y_min"]), (180, 0, 180), 1)
            cv2.line(annotated_frame, (0, reg_cfg["region_b"]["y_max"]), (cam_cfg["width"], reg_cfg["region_b"]["y_max"]), (180, 0, 180), 1)
            cv2.line(annotated_frame, (0, y_trigger_b), (cam_cfg["width"], y_trigger_b), (0, 0, 255), 2) # Vạch kích hoạt Gạt (Màu đỏ nét dày)
            cv2.putText(annotated_frame, "VUNG B (Kiem tra Mat B)", (15, reg_cfg["region_b"]["y_min"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 180), 1)
            cv2.putText(annotated_frame, "VACH KICH HOAT GAT (SERVO 2)", (15, y_trigger_b - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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

            # 2. Vẽ bảng điều khiển thông tin, thống kê trực quan
            # Tạo một hình chữ nhật nền tối góc trên bên phải để ghi thông tin thống kê
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (cam_cfg["width"] - 250, 10), (cam_cfg["width"] - 10, 160), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            
            # Tính toán FPS
            fps = 1.0 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0.0
            prev_time = time.time()
            
            # In các dòng text thống kê
            cv2.putText(annotated_frame, "MAY PHAN LOAI HAT DIEU", (cam_cfg["width"] - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (cam_cfg["width"] - 240, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Hat Da Kiem (A): {self.stats['total_a']}", (cam_cfg["width"] - 240, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Dat Yeu Cau (Dep): {self.stats['good']}", (cam_cfg["width"] - 240, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(annotated_frame, f"Bi Loai (Xau): {self.stats['bad']}", (cam_cfg["width"] - 240, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Hiển thị hàng đợi FIFO hiện tại dưới dạng text nhỏ
            fifo_len = len(self.cashew_fifo)
            cv2.putText(annotated_frame, f"Queue FIFO: {fifo_len} hat", (cam_cfg["width"] - 240, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Hiển thị cửa sổ
            cv2.imshow("He Thong Phan Loai Hat Dieu AI", annotated_frame)

            # Phím tắt đóng chương trình (nhấn 'q')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Đang dừng hệ thống và dọn dẹp tài nguyên...")
                break

        # Giải phóng thiết bị
        cap.release()
        cv2.destroyAllWindows()
        
        # Dừng luồng servo
        self.servo1_worker.is_running = False
        self.servo2_worker.is_running = False

if __name__ == "__main__":
    system = CashewSortingSystem()
    system.run()
