import cv2
import json
import os

# Thiết lập môi trường hiển thị cho Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Không tìm thấy tệp cấu hình {CONFIG_PATH}. Vui lòng tạo tệp trước.")
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("Đã lưu cấu hình mới vào config.json!")

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
    
    val_trigger_a = reg_cfg["region_a"]["line_trigger_y"]
    val_boundary_y = reg_cfg["region_a"]["y_max"] # Ranh giới giữa vùng A và B
    val_trigger_b = reg_cfg["region_b"]["line_trigger_y"]

    # Khởi tạo camera
    print("Đang khởi động camera căn chỉnh...")
    cap = cv2.VideoCapture(cam_cfg["device_id"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Không thể kết nối với Camera.")
        return

    # Tạo cửa sổ OpenCV và các thanh trượt (Trackbars)
    window_name = "Can Chinh Phan Vung Camera"
    cv2.namedWindow(window_name)
    
    # Thanh trượt cho Vách kích hoạt A (Servo 1)
    cv2.createTrackbar("Vach Trigger A", window_name, val_trigger_a, height, nothing)
    # Thanh trượt cho Ranh giới giữa Vùng A và Vùng B
    cv2.createTrackbar("Ranh Gioi A-B", window_name, val_boundary_y, height, nothing)
    # Thanh trượt cho Vách kích hoạt B (Servo 2)
    cv2.createTrackbar("Vach Trigger B", window_name, val_trigger_b, height, nothing)

    print("\n--- HƯỚNG DẪN CĂN CHỈNH ---")
    print("1. Kéo các thanh trượt ở trên cửa sổ để di chuyển các đường vách màu.")
    print("   - ĐƯỜNG XANH DƯƠNG: Vạch kích hoạt Servo 1 (Lật mặt) trong Vùng A.")
    print("   - ĐƯỜNG MÀU XÁM: Đường ranh giới chia đôi Vùng A và Vùng B.")
    print("   - ĐƯỜNG MÀU ĐỎ: Vạch kích hoạt Servo 2 (Gạt hạt xấu) trong Vùng B.")
    print("2. Nhấn phím 's' trên bàn phím để LƯU cấu hình mới vào config.json.")
    print("3. Nhấn phím 'q' để THOÁT chương trình.")
    print("---------------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        draw_frame = frame.copy()

        # Đọc giá trị thời gian thực từ các thanh trượt
        t_a = cv2.getTrackbarPos("Vach Trigger A", window_name)
        boundary = cv2.getTrackbarPos("Ranh Gioi A-B", window_name)
        t_b = cv2.getTrackbarPos("Vach Trigger B", window_name)

        # Đảm bảo logic hình học cơ bản (Vách A < Ranh giới < Vách B)
        # Giới hạn trên của Vùng A là 0, giới hạn dưới là boundary.
        # Giới hạn trên của Vùng B là boundary, giới hạn dưới là height.

        # --- VẼ CÁC ĐƯỜNG PHÂN VÙNG LÊN ẢNH ---
        # 1. Đường ranh giới chia Vùng A - Vùng B (Màu xám)
        cv2.line(draw_frame, (0, boundary), (width, boundary), (150, 150, 150), 2)
        cv2.putText(draw_frame, f"RANH GIOI A-B: {boundary}px", (15, boundary - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 2. Vùng A (Màu xanh dương)
        cv2.line(draw_frame, (0, t_a), (width, t_a), (255, 0, 0), 2)
        cv2.putText(draw_frame, f"VACH KICH HOAT A: {t_a}px (SERVO 1)", (15, t_a - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 3. Vùng B (Màu đỏ)
        cv2.line(draw_frame, (0, t_b), (width, t_b), (0, 0, 255), 2)
        cv2.putText(draw_frame, f"VACH KICH HOAT B: {t_b}px (SERVO 2)", (15, t_b - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Vẽ bảng hướng dẫn phím tắt trực quan lên góc trên bên trái
        overlay = draw_frame.copy()
        cv2.rectangle(overlay, (10, 10), (320, 95), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, draw_frame, 0.5, 0, draw_frame)
        
        cv2.putText(draw_frame, "PHIM TAT CAN CHINH:", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(draw_frame, "Nhan 's' : Luu cau hinh vao file", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(draw_frame, "Nhan 'q' : Thoat chuong trinh", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, draw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Cập nhật các giá trị vào biến cấu hình chính
            reg_cfg["region_a"]["line_trigger_y"] = t_a
            reg_cfg["region_a"]["y_max"] = boundary
            reg_cfg["region_b"]["y_min"] = boundary
            reg_cfg["region_b"]["line_trigger_y"] = t_b
            
            # Lưu lại vào file config.json
            save_config(cfg)
            
            # Vẽ chữ "ĐÃ LƯU" màu xanh lá cây chớp nháy nhanh lên màn hình để phản hồi
            feedback_img = draw_frame.copy()
            cv2.rectangle(feedback_img, (width//2 - 120, height//2 - 40), (width//2 + 120, height//2 + 20), (0, 255, 0), -1)
            cv2.putText(feedback_img, "DA LUU THANH CONG!", (width//2 - 100, height//2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow(window_name, feedback_img)
            cv2.waitKey(1000) # Hiển thị thông báo trong 1 giây

        elif key == ord('q'):
            print("Đang đóng chương trình căn chỉnh.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
