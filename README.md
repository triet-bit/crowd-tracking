# Pipeline Giám Sát Đám Đông và Phát Hiện Bất Thường (Crowd Tracking & Loitering Detection)

## Tổng Quan Dự Án
Dự án này triển khai hệ thống giám sát thời gian thực sử dụng trí tuệ nhân tạo để phát hiện, theo dõi, đếm người đi qua khu vực địa lý (Geofencing) và phát hiện hành vi lảng vảng (Loitering). Hệ thống kết hợp sức mạnh của YOLOv8 (phát hiện đối tượng) và thuật toán ByteTrack (theo dõi đa đối tượng) với logic kiểm tra dựa trên vận tốc và khu vực để phát hiện bất thường.

### Các Tính Năng Được Cải Tiến & Mô-đun Hóa
- **Theo dõi đối tượng (Tracking):** Sử dụng YOLOv8 và ByteTrack để phát hiện và gán ID ổn định cho mỗi người.
- **Geofencing (Hàng rào ảo):** Đếm số lượng người thuộc khu vực địa lý (Polygon) linh hoạt.
- **Phát hiện lảng vảng (Loitering):** Phát hiện người đứng quá lâu hoặc bám trụ tại một khu vực với tốc độ di chuyển cực thấp dựa vào thời gian định mức.
- **Thiết kế mô-đun:** Dễ dàng bảo trì, phát triển và chạy riêng biệt từng chức năng qua cấu trúc lệnh trên Terminal.

## Yêu Cầu Hệ Thống
- Python 3.8+
- Webcam hoặc camera IP (cho video thời gian thực)
- RAM: 4GB+ (khuyến nghị 8GB)
- GPU: Tùy chọn (cho tốc độ cao hơn)

## Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/your-username/crowd-tracking.git
cd crowd-tracking
```

### 2. Tạo Virtual Environment
```bash
python3 -m venv venv
# Trên Windows:
venv\Scripts\activate
# Trên Linux/Mac:
source venv/bin/activate
```

### 3. Cài Đặt Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib supervision
```

## Hướng Dẫn Sử Dụng

Mã nguồn chính của hệ thống nằm trong thư mục `BytetrackCountingLoitering/`. Hệ thống được chia theo module và hỗ trợ chạy qua Terminal cho từng mục đích cụ thể bằng file `run_video.py`.

### Cú pháp chung
```bash
python BytetrackCountingLoitering/run_video.py --video <đường_dẫn_video> --mode <kịch_bản>
```

### Các Kịch Bản (Modes) Hoạt Động
1. **Chế độ Tracking:** Chỉ thực hiện nhận diện và theo dõi (tracking).
   ```bash
   python BytetrackCountingLoitering/run_video.py --video output.mp4 --mode tracking
   ```

2. **Chế độ Geofencing:** Nhận diện và đếm số lượng người khi đi qua khu vực giới hạn.
   ```bash
   python BytetrackCountingLoitering/run_video.py --video output.mp4 --mode geofence
   ```

3. **Chế độ Loitering:** Hoạt động đầy đủ (Tracking, Geofencing và Bắt hành vi lảng vảng).
   ```bash
   python BytetrackCountingLoitering/run_video.py --video output.mp4 --mode loitering
   ```

## Cấu Trúc Dự Án
```
crowd-tracking/
├── BytetrackCountingLoitering/  # Thư mục module chính của pipeline giám sát
│   ├── config.py                # Cấu hình đa giác giám sát, tham số tốc độ/thời gian
│   ├── detection.py             # Cấu hình xử lý nhận diện YOLOv8
│   ├── loitering.py             # Thư viện tính toán lảng vảng và vận tốc
│   ├── main.py                  # Module chính kết nối các luồng xử lý
│   ├── run_video.py             # Script entrypoint cho Command Line
│   ├── tracking.py              # Logic tracking quản lý ByteTrack
│   └── utils.py                 # Hàm vẽ hiển thị (bounding box, text, polygon)
├── CrowdCounting/               # Các module đếm đám đông thử nghiệm khác
├── crowd_monitoring_pipeline.ipynb # File phác thảo ý tưởng trên Jupyter Notebook
├── README.md                    # Tài liệu hướng dẫn (Tài liệu này)
├── .gitignore                   # Cấu hình bỏ qua tệp tin rác, cache, thư mục ảo
└── ...
```

## Ghi Chú Quan Trọng
- **Cấu trúc `.gitignore`:** Đã chuẩn hóa việc ẩn đi các Virtual Environments (`venv/`, `crowd_env/`), thư mục `__pycache__`, các tập tin video đầu ra lớn (`*.mp4`, `*.avi`) và mô hình tracking dung lượng cao (`*.pt`) để dọn dẹp repo.
- **Tuỳ chỉnh Vùng Giám Sát:** Bạn có thể tự do tinh chỉnh toạ độ đa giác, ngưỡng vận tốc lảng vảng, và thời gian cảnh báo trực tiếp trong `BytetrackCountingLoitering/config.py`.

## Kế Hoạch Phát Triển (To-do)
- Gắn thêm API trả kết quả luồng stream về Frontend / Mobile App.
- Phát triển thêm các phân đoạn logic tính toán cảnh báo hành vi phức tạp hơn (VD: chạy tán loạn, ngã).
- Đóng gói mã nguồn thành môi trường Docker.

## Giấy Phép
Dự án sử dụng đa phần là các thư viện mã nguồn mở phân phối miễn phí theo quy định của YOLOv8 và chuẩn thư viện Python mở khác. Vui lòng tuân thủ bản quyền của thư viện bên thứ ba đang được áp dụng.
