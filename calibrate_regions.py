import cv2
import json
import os

# Thiết lập môi trường hiển thị cho Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Không tìm thấy tệp cấu hình {CONFIG_PATH}. Vui lòng tạo tệp trước.")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("💾 Đã lưu cấu hình mới vào config.json!")

def nothing(x):
    pass

def main():
    cfg = load_config()
    if cfg is None:
        return

    # Lấy các thông số ban đầu từ config
    cam_cfg = cfg["camera"]
    reg_cfg = cfg["regions"]
    
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    
    val_trigger_a = reg_cfg["region_a"]["line_trigger_x"]
    val_boundary_x = reg_cfg["region_a"]["x_min"] # Ranh giới trục X giữa vùng A và B
    val_trigger_b = reg_cfg["region_b"]["line_trigger_x"]

    # Khởi tạo camera
    print("🚀 Đang khởi động camera căn chỉnh (Trục X: PHẢI -> TRÁI)...")
    cap = cv2.VideoCapture(cam_cfg["device_id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Không thể kết nối với Camera.")
        return

    # Tạo cửa sổ OpenCV và các thanh trượt (Trackbars) dọc theo chiều rộng (Width)
    window_name = "Can Chinh Phan Vung Camera"
    cv2.namedWindow(window_name)
    
    # Thanh trượt cho Vách kích hoạt A (Servo 1)
    cv2.createTrackbar("Vach Trigger A", window_name, val_trigger_a, width, nothing)
    # Thanh trượt cho Ranh giới dọc giữa Vùng A và Vùng B
    cv2.createTrackbar("Ranh Gioi A-B", window_name, val_boundary_x, width, nothing)
    # Thanh trượt cho Vách kích hoạt B (Servo 2)
    cv2.createTrackbar("Vach Trigger B", window_name, val_trigger_b, width, nothing)

    print("\n--- HƯỚNG DẪN CĂN CHỈNH (BĂNG TẢI CHẠY PHẢI -> TRÁI) ---")
    print("1. Kéo các thanh trượt ở trên cửa sổ để di chuyển các đường vách màu dọc.")
    print("   - ĐƯỜNG XANH DƯƠNG: Vạch kích hoạt Servo 1 (Lật mặt) bên PHẢI (Vùng A).")
    print("   - ĐƯỜNG MÀU XÁM: Đường ranh giới dọc phân chia Vùng A (Phải) và Vùng B (Trái).")
    print("   - ĐƯỜNG MÀU ĐỎ: Vạch kích hoạt Servo 2 (Gạt hạt xấu) bên TRÁI (Vùng B).")
    print("2. Nhấn phím 's' trên bàn phím để LƯU cấu hình mới vào config.json.")
    print("3. Nhấn phím 'q' để THOÁT chương trình.")
    print("-------------------------------------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        draw_frame = frame.copy()

        # Đọc giá trị thời gian thực từ các thanh trượt
        t_a = cv2.getTrackbarPos("Vach Trigger A", window_name)
        boundary = cv2.getTrackbarPos("Ranh Gioi A-B", window_name)
        t_b = cv2.getTrackbarPos("Vach Trigger B", window_name)

        # --- VẼ CÁC ĐƯỜNG PHÂN VÙNG DỌC LÊN ẢNH ---
        # 1. Đường ranh giới dọc chia Vùng A - Vùng B (Màu xám)
        cv2.line(draw_frame, (boundary, 0), (boundary, height), (150, 150, 150), 2)
        cv2.putText(draw_frame, f"RANH GIOI A-B: {boundary}px", (boundary + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # 2. Vùng A - Bên Phải (Màu xanh dương)
        cv2.line(draw_frame, (t_a, 0), (t_a, height), (255, 0, 0), 2)
        cv2.putText(draw_frame, f"TRIGGER A: {t_a}px (SERVO 1)", (t_a + 10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        # 3. Vùng B - Bên Trái (Màu đỏ)
        cv2.line(draw_frame, (t_b, 0), (t_b, height), (0, 0, 255), 2)
        cv2.putText(draw_frame, f"TRIGGER B: {t_b}px (SERVO 2)", (t_b - 180, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Mũi tên hướng di chuyển băng tải (phải sang trái)
        cv2.arrowedLine(draw_frame, (width - 50, height - 30), (50, height - 30), (0, 255, 255), 2, tipLength=0.05)
        cv2.putText(draw_frame, "HUONG BANG CHUYEN (PHAI -> TRAI)", (width // 2 - 120, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Vẽ bảng hướng dẫn phím tắt trực quan
        overlay = draw_frame.copy()
        cv2.rectangle(overlay, (10, height - 100), (320, height - 15), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, draw_frame, 0.5, 0, draw_frame)
        
        cv2.putText(draw_frame, "PHIM TAT CAN CHINH:", (20, height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(draw_frame, "Nhan 's' : Luu cau hinh vao file", (20, height - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(draw_frame, "Nhan 'q' : Thoat chuong trinh", (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, draw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Cập nhật các giá trị dọc trục X vào biến cấu hình chính
            reg_cfg["region_a"]["line_trigger_x"] = t_a
            reg_cfg["region_a"]["x_min"] = boundary
            reg_cfg["region_a"]["x_max"] = width
            reg_cfg["region_b"]["x_min"] = 0
            reg_cfg["region_b"]["x_max"] = boundary
            reg_cfg["region_b"]["line_trigger_x"] = t_b
            
            save_config(cfg)
            
            # Phản hồi đồ họa "ĐÃ LƯU" màu xanh lá cây chớp nháy nhanh
            feedback_img = draw_frame.copy()
            cv2.rectangle(feedback_img, (width//2 - 120, height//2 - 40), (width//2 + 120, height//2 + 20), (0, 255, 0), -1)
            cv2.putText(feedback_img, "DA LUU THANH CONG!", (width//2 - 100, height//2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow(window_name, feedback_img)
            cv2.waitKey(1000)

        elif key == ord('q'):
            print("👋 Đang đóng chương trình căn chỉnh.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
