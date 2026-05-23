import sys
import uuid
import asyncio

# Thêm đường dẫn project vào sys.path để có thể import các module
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.app.adapters.notifier.telegram_notifier import create_notifier

async def send_mock_alert():
    print("Đang khởi tạo Telegram Notifier...")
    notifier = create_notifier()
    
    # Giả lập payload giống như khi có camera thật
    camera_id = "MOCK_CAM_01"
    alert_type = "crowding_detected"
    count = 15
    severity = "high"
    message = "Phát hiện đám đông tập trung bất thường tại khu vực cổng chính."
    
    text = (
        f"🚨 <b>CROWD ALERT</b>\n"
        f"Camera: <code>{camera_id}</code>\n"
        f"Type: {alert_type}\n"
        f"Severity: {severity.upper()}\n"
        f"People count: {count}\n"
        f"Message: {message}"
    )

    print("Đang gửi tin nhắn mock lên Telegram...")
    # Nếu bạn có 1 file ảnh test, hãy sửa đường dẫn và dùng dòng dưới
    # result = await notifier.send_photo("duong_dan_anh_test.jpg", caption=text)
    
    # Ở đây chúng ta test gửi text thông thường trước
    result = await notifier.send_message(text)
    
    print("Kết quả trả về từ Telegram:", result)
    if result.get("status") == "sent":
        print("✅ Đã gửi thành công! Hãy kiểm tra Telegram của bạn.")
    else:
        print("❌ Gửi thất bại. Hãy kiểm tra lại Bot Token và Chat ID trong file .env")

if __name__ == "__main__":
    asyncio.run(send_mock_alert())
