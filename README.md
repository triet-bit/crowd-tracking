# Pipeline Giám Sát Đám Đông và Phát Hiện Bất Thường

## Tổng Quan Dự Án
Dự án này triển khai hệ thống giám sát đám đông thời gian thực sử dụng trí tuệ nhân tạo để phát hiện, theo dõi và đếm người trong khu vực giám sát. Hệ thống sử dụng YOLOv8 cho phát hiện đối tượng, ByteTrack cho theo dõi đa đối tượng, và logic tùy chỉnh để phát hiện lảng vảng. Được tối ưu cho thiết bị edge như Raspberry Pi.

### Tính Năng Chính
- Phát hiện người, xe đạp và ô tô.
- Theo dõi đa đối tượng với ID ổn định.
- Đếm người trong khu vực đa giác.
- Phát hiện lảng vảng với kiểm tra vận tốc.
- Hiển thị kết quả thời gian thực.

## Yêu Cầu Hệ Thống
- Python 3.8+
- Webcam hoặc camera IP (cho video real-time)
- RAM: 4GB+ (khuyến nghị 8GB)
- GPU: Tùy chọn (cho tốc độ cao hơn)

## Cài Đặt

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Anomaly-Detection-in-Surveillance-Videos.git
cd Anomaly-Detection-in-Surveillance-Videos
```

### 2. Tạo Virtual Environment
```bash
python -m venv crowd_env
# Trên Windows:
crowd_env\Scripts\activate
# Trên Linux/Mac:
source crowd_env/bin/activate
```

### 3. Cài Đặt Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib
```

### 4. Chuẩn Bị Dữ Liệu
- Tải dataset ShanghaiTech từ [link chính thức](https://github.com/desenzhou/ShanghaiTechDataset) hoặc nguồn khác.
- Giải nén vào thư mục `Shanghaidataset/` trong root project.
- Cấu trúc thư mục:
```
Shanghaidataset/
├── shanghaitech/
    ├── shanghaitech/
        ├── testing/
            ├── frames/
                ├── 01_0014/
                    ├── 186.jpg
                    └── ...
```

## Chạy Hệ Thống

### Chạy Demo Trên Ảnh
1. Mở Jupyter Notebook:
```bash
jupyter notebook
```

2. Mở file `crowd_monitoring_pipeline.ipynb`.

3. Chạy từng cell theo thứ tự:
   - Cell 1: Cài đặt thư viện.
   - Cell 2: Import modules.
   - Cell 3: Tải mô hình YOLOv8s.
   - Cell 4: Tải ảnh mẫu từ Shanghai dataset.
   - Cell 5: Định nghĩa đa giác và hàm tiện ích.
   - Cell 6: Phát hiện lảng vảng.
   - Cell 7: Xử lý ảnh và hiển thị kết quả.

4. Kết quả: Ảnh được xử lý với bounding boxes, track IDs, class labels, và count.

### Chạy Hệ Thống Bằng Công Cụ Dòng Lệnh Trên Video (Tool Mới)
Hệ thống hiện đã hỗ trợ chạy trực tiếp trên file video hoặc camera bằng lệnh `run_video.py` với nhiều kịch bản linh hoạt:

```bash
# 1. Chế độ Tracking (Chỉ nhận diện và theo dõi yolo)
python3 BytetrackCountingLoitering/run_video.py --video output.mp4 --mode tracking

# 2. Chế độ Geofence (Vẽ đa giác và đếm số người đi ngang vùng)
python3 BytetrackCountingLoitering/run_video.py --video output.mp4 --mode geofence

# 3. Chế độ Loitering (Đầy đủ chức năng + Bắt hành vi lảng vảng - Default)
python3 BytetrackCountingLoitering/run_video.py --video output.mp4 --mode loitering
```

**Các tham số bổ trợ hữu ích:**
- `--video 0`: Nếu bạn muốn chạy bằng Webcam thay vì truyền đường dẫn file `output.mp4`.
- `--output ket_qua.mp4`: Nếu bạn muốn hệ thống lưu kết quả xử lý thành file video trong khi chạy.
- `--draw-roi`: Cờ chế độ **Interactive Polygon**. Hệ thống sẽ dừng ở frame đầu tiên và cho phép bạn tự dùng chuột khoanh vùng giám sát (Geofence) trên video mà không cần sửa file `config.py`. Cực kỳ hữu dụng khi thay đổi camera!

## Cấu Trúc Dự Án
```
Anomaly-Detection-in-Surveillance-Videos/
├── crowd_monitoring_pipeline.ipynb  # Notebook chính
├── yolo-model.ipynb                 # Benchmark YOLO models
├── ByteTrack/                       # Thư viện ByteTrack (nếu cần)
├── Shanghaidataset/                 # Dataset ShanghaiTech
├── yolov8n.pt                       # Mô hình YOLOv8n
├── README.md                        # Hướng dẫn này
└── crowd_env/                       # Virtual environment
```

## Ghi Chú Quan Trọng
- Mô hình YOLOv8s được chọn từ benchmarking để cân bằng tốc độ và độ chính xác trên edge.
- ByteTrack cải tiến SORT để giảm ID switch và xử lý low-confidence detections.
- Đếm chỉ áp dụng cho class "person" (0).
- Phát hiện lảng vảng kết hợp thời gian (>5s) và vận tốc (<0.5 pixel/frame).

## Phát Triển Tiếp Theo
- Mở rộng sang video real-time.
- Thêm phát hiện bất thường (chạy/đẩy).
- Tích hợp giao diện web, cơ sở dữ liệu, và điều khiển phần cứng.
- Tối ưu hóa với TensorRT/ONNX cho edge.

## Giấy Phép
Dự án này sử dụng mã nguồn mở. Vui lòng tuân thủ giấy phép của các thư viện bên thứ ba.
