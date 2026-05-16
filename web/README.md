# Crowd Tracking Web Dashboard

Đây là hệ thống quản trị và giám sát đám đông trên nền tảng Web, đóng vai trò giao diện trực quan và bộ điều khiển cho mô hình AI lõi (YOLOv8 + ByteTrack). Hệ thống cung cấp khả năng truyền phát Video theo thời gian thực (MJPEG), giám sát số liệu qua WebSocket, quản lý Camera, vẽ phân vùng giám sát (Geofencing) linh hoạt, và tự động gửi cảnh báo (Telegram/Hardware).

## 🏗️ Kiến Trúc Hệ Thống
*   **Backend:** FastAPI (RESTful API & WebSockets)
*   **Frontend:** Vanilla HTML/CSS/JS (Giao diện kính mờ Glassmorphism hiện đại)
*   **Database:** SQLite + SQLAlchemy (Lưu trữ cấu hình Camera, Zone, và Lịch sử cảnh báo)
*   **Background Processing:** Celery + Redis (Chạy ngầm Model AI tránh gây nghẽn Web)
*   **Core AI Integration:** Adapter Pattern kết nối mượt mà với thư mục `BytetrackCountingLoitering` của dự án.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Dự Án

### BƯỚC 1: Cấu hình Môi trường (Environment Variables)
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cd web
cp .env.example .env
```
Mở file `.env` vừa tạo và điền các thông số. **Lưu ý quan trọng:**
*   `AI_ENGINE=mock`: Chạy chế độ giả lập (Dữ liệu sinh ngẫu nhiên, nhẹ máy, dùng để test giao diện).
*   `AI_ENGINE=crowd_tracking`: Chạy chế độ AI THẬT (Sẽ gọi mô hình YOLOv8 từ thư mục gốc, yêu cầu máy phải có thư viện AI).
*   Điền `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` nếu muốn nhận thông báo.

### BƯỚC 2: Khởi chạy dự án

Có 2 cách để chạy hệ thống: dùng Docker (Khuyên dùng) hoặc chạy Local.

#### Cách A: Chạy bằng Docker Compose (Dễ nhất, khuyên dùng)
Yêu cầu máy bạn đã cài đặt sẵn Docker và Docker Compose. Mọi dependency như Redis, Python, Celery sẽ tự động được xử lý.

```bash
cd web
docker-compose up --build -d
```
*Hệ thống sẽ chạy ở: `http://localhost:8000`*

*(Lưu ý: Nếu sử dụng `AI_ENGINE=crowd_tracking` trong Docker, hãy đảm bảo bạn đã cấu hình GPU trong file docker-compose.yml và đã bỏ comment các thư viện AI trong `web/requirements.txt`)*

#### Cách B: Chạy thủ công (Local Development)
Dành cho việc debug hoặc phát triển thêm tính năng. Yêu cầu máy bạn đã cài đặt sẵn **Redis Server** và đang chạy.

1.  **Cài đặt thư viện:**
    ```bash
    # (Khuyên dùng môi trường ảo - venv)
    pip install -r web/requirements.txt
    ```
2.  **Khởi động Backend API (Mở Terminal 1):**
    ```bash
    # Chạy lệnh này từ THƯ MỤC GỐC của dự án (crowd-tracking)
    uvicorn web.app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
3.  **Khởi động Celery Worker (Mở Terminal 2):**
    ```bash
    # Chạy lệnh này từ THƯ MỤC GỐC của dự án (crowd-tracking)
    celery -A web.app.core.celery_app worker --loglevel=INFO -Q stream_queue,alert_queue,notification_queue,hardware_queue,stats_queue --concurrency=2
    ```
*Hệ thống sẽ chạy ở: `http://localhost:8000`*

---

## 🎮 Hướng Dẫn Sử Dụng Dashboard

1.  **Truy cập Web:** Mở trình duyệt vào `http://localhost:8000`.
2.  **Thêm Camera:**
    *   Vào tab **Cameras** -> Nhấn **+ Thêm camera**.
    *   Bạn có thể nhập URL của File Video MP4 hoặc link RTSP từ IP Camera.
    *   (Gợi ý: Chọn Loại nguồn `Mock (Demo)` nếu chỉ muốn test luồng stream).
3.  **Bật Giám Sát:**
    *   Tại tab **Dashboard**, nhập ID của camera vừa tạo vào ô `Camera ID` phía trên cùng bên phải.
    *   Nhấn nút **Start**. Đợi vài giây để Celery Worker khởi động AI, Video Live và biểu đồ sẽ bắt đầu nhảy số liệu.
4.  **Vẽ Vùng Giám Sát Động (Dynamic Zone):**
    *   Khi Camera đang phát Live, nhấn nút **"Vẽ Zone"** trên thanh công cụ của video.
    *   Click chuột lên khung video để đánh dấu các điểm tọa độ đa giác.
    *   Nhấn **"Lưu Zone"**. Hệ thống sẽ lập tức cập nhật tọa độ mới xuống cho Model AI chạy dưới nền mà không cần khởi động lại.
5.  **Dừng Hệ Thống:** Nhấn nút **Stop** trên cùng bên phải để giải phóng RAM và kết thúc luồng theo dõi.

---

## 🛠 Cấu trúc thư mục `web/`

*   `app/api/`: Các endpoint RESTful API phục vụ frontend (Camera, Alert, Stats...).
*   `app/core/`: Cấu hình nền tảng (Database, Redis, Config, Celery settings).
*   `app/models/`: Định nghĩa các bảng CSDL (SQLAlchemy ORM).
*   `app/tasks/`: Mã nguồn của các Celery Background Worker (Nhận diện luồng Video, xử lý Alert...).
*   `app/ws/`: Máy chủ WebSocket đẩy số liệu realtime.
*   `app/adapters/`: Nơi kết nối Web với các module bên ngoài (Module AI lõi, Phần cứng IoT, Telegram). Thiết kế theo chuẩn Interface.
*   `frontend/` & `app/static/`: Chứa file HTML, JavaScript và CSS.
