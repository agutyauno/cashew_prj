import os
import cv2
import time
import json
import sys
from ultralytics import YOLO

# --- HỖ TRỢ HIỂN THỊ MÀU SẮC TRÊN TERMINAL WINDOWS ---
if sys.platform == 'win32':
    os.system('color')

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# --- THỬ LẤY BỘ CHỌN FILE ĐỒ HỌA (GUI) ---
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import filedialog
    GUI_AVAILABLE = True
except Exception:
    pass

def load_config(config_path="config.json"):
    """Nạp cấu hình mặc định từ config.json."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"{Colors.YELLOW}[CẢNH BÁO] Không thể đọc file {config_path}: {e}. Sẽ dùng cấu hình mặc định.{Colors.END}")
    return {
        "model": {
            "path": "./cashew_26n_ncnn_model",
            "conf_threshold": 0.3,
            "imgsz": 736
        }
    }

def select_file_gui(title="Chọn tệp ảnh"):
    """Mở hộp thoại chọn tệp đồ họa trên Windows."""
    if not GUI_AVAILABLE:
        return None
    try:
        root = tk.Tk()
        root.withdraw()  # Ẩn cửa sổ chính của tkinter
        root.attributes("-topmost", True)  # Đưa hộp thoại lên trên cùng
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        root.update()  # Xử lý hết các sự kiện đồ họa còn đọng để đóng hoàn toàn Explorer
        root.destroy()
        return file_path if file_path else None
    except Exception:
        return None

def select_directory_gui(title="Chọn thư mục chứa ảnh"):
    """Mở hộp thoại chọn thư mục đồ họa trên Windows."""
    if not GUI_AVAILABLE:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        dir_path = filedialog.askdirectory(title=title)
        root.update()
        root.destroy()
        return dir_path if dir_path else None
    except Exception:
        return None

def show_image_fixed_size(window_name, img, width=720, height=720):
    """Hiển thị ảnh với kích thước cửa sổ cố định (mặc định 720x720)."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    cv2.imshow(window_name, img)

def wait_for_key_or_close(window_name):
    """Đợi người dùng nhấn một phím hoặc click nút X để đóng cửa sổ ảnh mà không bị treo terminal."""
    print(f"\n{Colors.YELLOW}[HƯỚNG DẪN] Click vào cửa sổ ảnh và nhấn PHÍM BẤT KỲ (hoặc click nút [X] đỏ của cửa sổ) để đóng và tiếp tục.{Colors.END}")
    while True:
        # Nhận sự kiện phím (đợi 50ms)
        key = cv2.waitKey(50) & 0xFF
        if key != 255:  # 255 nghĩa là không có phím nào được nhấn
            break
        
        # Kiểm tra xem người dùng đã click nút X để tắt cửa sổ chưa
        try:
            # WINDOW_AUTOSIZE hoặc WND_PROP_VISIBLE đều trả về giá trị âm hoặc 0 khi cửa sổ bị đóng
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break
            
    cv2.destroyWindow(window_name)

def choose_model(cfg):
    """Hiển thị menu để chọn model cần test."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== CHỌN MÔ HÌNH AI ĐỂ KIỂM TRA ==={Colors.END}")
    
    default_path = cfg["model"]["path"]
    models = []
    
    # 1. Thêm model mặc định từ config.json
    models.append(("Model mặc định (từ config.json)", default_path))
    
    # 2. Quét các model khả dụng trong thư mục hiện tại
    potential_models = [
        ("best.pt", "best.pt"),
        ("tinhchinh.pt", "tinhchinh.pt"),
        ("tinhchinh_ncnn_model", "tinhchinh_ncnn_model"),
        ("cashew_26n_ncnn_model", "cashew_26n_ncnn_model")
    ]
    
    for label, path in potential_models:
        # Tránh trùng lặp với model mặc định
        if os.path.exists(path) and os.path.abspath(path) != os.path.abspath(default_path):
            models.append((label, path))
            
    models.append(("Nhập đường dẫn model khác thủ công...", "custom"))
    
    # Hiển thị danh sách cho người dùng lựa chọn
    for idx, (label, path) in enumerate(models):
        status = f" ({Colors.GREEN}khả dụng{Colors.END})" if path != "custom" and os.path.exists(path) else ""
        if path == "custom":
            status = ""
        print(f"[{idx + 1}] {Colors.BOLD}{label}{Colors.END} -> {Colors.CYAN}{path}{Colors.END}{status}")
        
    while True:
        try:
            choice = input(f"\nNhập số lựa chọn của bạn (mặc định [1]): ").strip()
            if not choice:
                idx_choice = 0
                break
            idx_choice = int(choice) - 1
            if 0 <= idx_choice < len(models):
                break
            else:
                print(f"{Colors.RED}Lựa chọn không hợp lệ. Vui lòng nhập từ 1 đến {len(models)}.{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Vui lòng nhập một số hợp lệ.{Colors.END}")
            
    selected_label, selected_path = models[idx_choice]
    
    if selected_path == "custom":
        while True:
            custom_path = input("Nhập đường dẫn chi tiết tới model (.pt hoặc thư mục NCNN): ").strip()
            if os.path.exists(custom_path):
                selected_path = custom_path
                break
            else:
                print(f"{Colors.RED}Đường dẫn không tồn tại. Vui lòng kiểm tra lại.{Colors.END}")
                
    return selected_path

def run_inference(model, image_path, conf_threshold, imgsz, output_dir):
    """Thực hiện nhận diện trên một ảnh và lưu kết quả."""
    print(f"\n{Colors.BLUE}--------------------------------------------------{Colors.END}")
    print(f"Đang xử lý ảnh: {Colors.BOLD}{os.path.basename(image_path)}{Colors.END}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"{Colors.RED}[LỖI] Không thể đọc ảnh từ {image_path}{Colors.END}")
        return None
        
    height, width = img.shape[:2]
    
    # Đo thời gian inference
    start_time = time.time()
    results = model(img, imgsz=imgsz, conf=conf_threshold, verbose=False)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    annotated_img = img.copy()
    count_good = 0
    count_bad = 0
    
    detected_objects = []
    
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].cpu().item())
            label = model.names[cls_id]
            conf = box.conf[0].cpu().item()
            
            # Thống kê
            if label == "dep":
                count_good += 1
                color = (0, 255, 0)      # Xanh lá cho hạt đẹp
                text_color = Colors.GREEN
            else:
                count_bad += 1
                color = (0, 0, 255)      # Đỏ cho hạt xấu
                text_color = Colors.RED
                
            detected_objects.append((label, conf, (x1, y1, x2, y2)))
            
            # Vẽ hộp giới hạn và text
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # In ra console từng hạt nhận diện được
            print(f"  + Phát hiện: {text_color}{Colors.BOLD}{label}{Colors.END} | Độ tin cậy: {Colors.CYAN}{conf:.2f}{Colors.END} | Vị trí: ({x1}, {y1}) -> ({x2}, {y2})")

    # Vẽ bảng thông tin tóm tắt lên ảnh
    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0, annotated_img)
    
    cv2.putText(annotated_img, "KET QUA NHAN DIEN AI", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(annotated_img, f"Thoi gian xu ly: {inference_time:.1f} ms", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated_img, f"Tong hat phat hien: {count_good + count_bad}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated_img, f"Hat DEP: {count_good}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(annotated_img, f"Hat XAU: {count_bad}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # In tóm tắt ra console
    print(f"\n{Colors.GREEN}Kết quả nhận diện cho {os.path.basename(image_path)}:{Colors.END}")
    print(f"  - Thời gian xử lý: {Colors.BOLD}{inference_time:.1f} ms{Colors.END}")
    print(f"  - Số hạt ĐẸP: {Colors.GREEN}{Colors.BOLD}{count_good}{Colors.END}")
    print(f"  - Số hạt XẤU: {Colors.RED}{Colors.BOLD}{count_bad}{Colors.END}")
    
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu ảnh kết quả
    output_filename = f"result_{os.path.basename(image_path)}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, annotated_img)
    print(f"  - {Colors.BOLD}Đã lưu ảnh vẽ kết quả tại:{Colors.END} {Colors.UNDERLINE}{output_path}{Colors.END}")
    
    return {
        "image_path": image_path,
        "output_path": output_path,
        "inference_time_ms": inference_time,
        "good_count": count_good,
        "bad_count": count_bad,
        "annotated_image": annotated_img
    }

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("=================================================================")
    print("      CHUONG TRINH KIEM TRA KHAN NANG NHAN DIEN MODEL AI         ")
    print("             PHAN LOAI HAT DIEU (DEP / XAU)                      ")
    print("=================================================================")
    print(f"{Colors.END}")
    
    cfg = load_config()
    
    # 1. Cho phép chọn Model
    model_path = choose_model(cfg)
    
    print(f"\n{Colors.YELLOW}Đang tải mô hình AI từ: {model_path}...{Colors.END}")
    try:
        start_load = time.time()
        model = YOLO(model_path, task="detect")
        load_time = time.time() - start_load
        print(f"{Colors.GREEN}Đã tải mô hình AI thành công trong {load_time:.2f}s!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}[LỖI NẠP MODEL] Không thể nạp mô hình: {e}{Colors.END}")
        input("Nhấn Enter để thoát...")
        return
        
    conf_threshold = cfg["model"]["conf_threshold"]
    imgsz = cfg["model"]["imgsz"]
    output_dir = "test_results"
    
    print(f"\n{Colors.BOLD}Cấu hình nhận diện:{Colors.END}")
    print(f"  - Confidence Threshold: {Colors.CYAN}{conf_threshold}{Colors.END} (Độ tin cậy tối thiểu)")
    print(f"  - Image Size: {Colors.CYAN}{imgsz}{Colors.END} (Kích thước ảnh đầu vào YOLO)")
    print(f"  - Thư mục lưu kết quả: {Colors.CYAN}{output_dir}/{Colors.END}")
    
    # Thay đổi thông số cấu hình nếu muốn
    change_conf = input(f"\nBạn có muốn thay đổi Threshold ({conf_threshold}) không? (Nhấn Enter để giữ nguyên, hoặc nhập số mới từ 0.0 đến 1.0): ").strip()
    if change_conf:
        try:
            val = float(change_conf)
            if 0.0 <= val <= 1.0:
                conf_threshold = val
                print(f"-> Đã đổi Confidence Threshold thành: {Colors.BOLD}{conf_threshold}{Colors.END}")
            else:
                print(f"{Colors.RED}Giá trị nằm ngoài khoảng [0, 1]. Giữ nguyên mặc định.{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Định dạng số không hợp lệ. Giữ nguyên mặc định.{Colors.END}")
            
    while True:
        print(f"\n{Colors.HEADER}{Colors.BOLD}=== MENU CHỌN NGUỒN ẢNH ĐỂ TEST ==={Colors.END}")
        print("[1] Chọn 1 ảnh bất kỳ từ máy tính")
        print("[2] Nhập đường dẫn ảnh thủ công")
        print("[3] Kiểm tra toàn bộ ảnh trong một thư mục (Batch Mode)")
        print("[4] Thoát chương trình")
        
        choice = input(f"\nNhập số lựa chọn của bạn: ").strip()
        
        if choice == "1":
            if GUI_AVAILABLE:
                print(f"\n{Colors.YELLOW}Đang mở hộp thoại chọn ảnh...{Colors.END}")
                image_path = select_file_gui("Chọn tệp ảnh hạt điều")
                if image_path:
                    res = run_inference(model, image_path, conf_threshold, imgsz, output_dir)
                    if res:
                        win_name = "Ket qua nhan dien AI"
                        show_image_fixed_size(win_name, res["annotated_image"])
                        wait_for_key_or_close(win_name)
                else:
                    print(f"{Colors.YELLOW}Hủy chọn tệp ảnh.{Colors.END}")
            else:
                print(f"{Colors.RED}Hệ thống không hỗ trợ giao diện chọn file đồ họa. Hãy dùng lựa chọn [2] để nhập đường dẫn.{Colors.END}")
                
        elif choice == "2":
            image_path = input("Nhập đường dẫn đầy đủ tới tệp ảnh: ").strip()
            # Bỏ dấu nháy kép bọc quanh đường dẫn nếu người dùng kéo thả file vào
            image_path = image_path.replace('"', '').replace("'", "")
            if os.path.exists(image_path) and os.path.isfile(image_path):
                res = run_inference(model, image_path, conf_threshold, imgsz, output_dir)
                if res:
                    win_name = "Ket qua nhan dien AI"
                    show_image_fixed_size(win_name, res["annotated_image"])
                    wait_for_key_or_close(win_name)
            else:
                print(f"{Colors.RED}[LỖI] Tệp ảnh không tồn tại: {image_path}{Colors.END}")
                
        elif choice == "3":
            dir_path = None
            if GUI_AVAILABLE:
                use_gui = input("Bạn có muốn dùng hộp thoại đồ họa chọn thư mục không? (Y/n): ").strip().lower()
                if use_gui != 'n':
                    print(f"\n{Colors.YELLOW}Đang mở hộp thoại chọn thư mục...{Colors.END}")
                    dir_path = select_directory_gui("Chọn thư mục chứa ảnh hạt điều")
            
            if not dir_path:
                dir_path = input("Nhập đường dẫn tới thư mục chứa ảnh: ").strip()
                dir_path = dir_path.replace('"', '').replace("'", "")
                
            if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Quét các định dạng ảnh phổ biến
                valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(valid_extensions)]
                
                if len(files) == 0:
                    print(f"{Colors.YELLOW}Không tìm thấy tệp ảnh hợp lệ nào trong thư mục {dir_path}{Colors.END}")
                    continue
                    
                print(f"\n{Colors.GREEN}Tìm thấy {len(files)} ảnh trong thư mục.{Colors.END}")
                print(f"Bắt đầu nhận diện hàng loạt...")
                
                total_good = 0
                total_bad = 0
                total_time = 0.0
                processed_count = 0
                
                for f in files:
                    res = run_inference(model, f, conf_threshold, imgsz, output_dir)
                    if res:
                        total_good += res["good_count"]
                        total_bad += res["bad_count"]
                        total_time += res["inference_time_ms"]
                        processed_count += 1
                        
                        # Hiện ảnh lướt nhanh với kích thước cố định
                        show_image_fixed_size("Batch testing - Dang chay...", res["annotated_image"])
                        # Chờ 500ms giữa các ảnh để người dùng có thể xem kết quả lướt qua
                        if cv2.waitKey(500) & 0xFF == ord('q'):
                            print(f"{Colors.YELLOW}Đã dừng quét hàng loạt trước thời hạn.{Colors.END}")
                            break
                            
                cv2.destroyAllWindows()
                
                if processed_count > 0:
                    avg_time = total_time / processed_count
                    print(f"\n{Colors.GREEN}{Colors.BOLD}=== TỔNG KẾT QUÉT HÀNG LOẠT (BATCH REPORT) ==={Colors.END}")
                    print(f"  - Tổng số ảnh đã quét: {Colors.BOLD}{processed_count}/{len(files)}{Colors.END}")
                    print(f"  - Thời gian xử lý trung bình mỗi ảnh: {Colors.BOLD}{avg_time:.1f} ms{Colors.END}")
                    print(f"  - Tổng hạt ĐẸP tìm thấy: {Colors.GREEN}{Colors.BOLD}{total_good}{Colors.END}")
                    print(f"  - Tổng hạt XẤU tìm thấy: {Colors.RED}{Colors.BOLD}{total_bad}{Colors.END}")
            else:
                print(f"{Colors.RED}[LỖI] Thư mục không hợp lệ hoặc không tồn tại.{Colors.END}")
                
        elif choice == "4":
            print(f"\n{Colors.GREEN}Cảm ơn bạn đã sử dụng chương trình! Tạm biệt.{Colors.END}")
            break
        else:
            print(f"{Colors.RED}Lựa chọn không hợp lệ. Vui lòng nhập số từ 1 đến 4.{Colors.END}")

if __name__ == "__main__":
    main()
