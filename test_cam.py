import cv2
import time
import os
from ultralytics import YOLO

# Ép hệ thống dùng giao diện X11 để tránh lỗi hiển thị trên Pi
os.environ["QT_QPA_PLATFORM"] = "xcb"

def test_camera():
    # Thay đổi đường dẫn này nếu tên file ONNX của bạn khác
    model_path = '/home/admin/cashew/cashew_26n_ncnn_model'
    
    print("--- Đang nạp bộ não AI ---")
    try:
        model = YOLO(model_path, task='detect')
        print("✅ Nạp model thành công!")
    except Exception as e:
        print(f"❌ Lỗi nạp model: {e}")
        return

    print("\n--- Đang cấu hình Camera... ---")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Nâng độ phân giải lên HD (720p) để nhìn hạt điều nét hơn
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 

    if not cap.isOpened():
        print("❌ Không thể kết nối Camera.")
        return

    print("🚀 Camera đã sẵn sàng. Đang đợi khung hình đầu tiên...")
    time.sleep(2) 

    # Khởi tạo biến thời gian để tính FPS
    prev_time = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            continue

        # Tạo bản sao để tránh lỗi xung đột bộ nhớ
        input_frame = frame.copy()

        # 1. Ghi nhận thời gian bắt đầu xử lý khung hình này
        current_time = time.time()

        # 2. Nhận diện YOLO 
        results = model(input_frame, imgsz=736, conf=0.3, verbose=False)

        # 3. Vẽ khung nhận diện
        annotated_frame = results[0].plot()

        # --- TÍNH TOÁN VÀ HIỂN THỊ FPS ---
        # Tính khoảng thời gian giữa 2 khung hình
        time_diff = current_time - prev_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_time = current_time # Cập nhật lại thời gian cho vòng lặp sau
        
        fps_text = f"FPS: {fps:.1f}"
        
        # Vẽ chữ lên ảnh (Góc trên bên trái, màu xanh lá cây, độ dày=2)
        cv2.putText(annotated_frame, fps_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # ---------------------------------

        # Hiển thị cửa sổ
        cv2.imshow("May Phan Loai Hat Dieu", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Đang đóng chương trình...")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
