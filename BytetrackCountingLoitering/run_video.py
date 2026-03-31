import cv2
import time
import argparse
import numpy as np
from main import process_frame
from loitering import loiter_dict

def select_polygon(frame):
    points = []
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    window_name = 'Draw Polygon (Press ENTER to finish, R to reset)'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_polygon)

    while True:
        temp_frame = frame.copy()
        if len(points) > 0:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
            for p in points:
                cv2.circle(temp_frame, p, 4, (0, 0, 255), -1)
        if len(points) > 2:
            cv2.polylines(temp_frame, [np.array(points)], isClosed=True, color=(255, 0, 0), thickness=2)
            
        cv2.putText(temp_frame, "Click left mouse to add points.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(temp_frame, "Press ENTER to confirm, R to reset.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # ENTER
            break
        elif key == ord('r'):
            points = []

    cv2.destroyWindow(window_name)
    if len(points) < 3:
        return None
    return np.array(points, np.int32)

def run_on_video(video_path, output_path=None, mode="loitering", draw_roi=False):
    # Nếu truyền vào "0" thì chuyển thành số nguyên 0 để mở Webcam
    if video_path.isdigit():
        video_path = int(video_path)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Lấy thông số FPS và kích thước để hiển thị / lưu video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 25 # Mặc định nếu không lấy được fps
    
    # Thiết lập VideoWriter nếu muốn lưu video (output_path không rỗng)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    print("Đang xử lý video... Nhấn 'q' để thoát.")
    
    # Initialize flow tracking
    flow_dict = {}
    enter_count = 0
    exit_count = 0
    
    # Lấy frame đầu tiên để vẽ vùng nếu draw_roi=True
    ret, first_frame = cap.read()
    if not ret:
        print("Không thể đọc frame đầu tiên của video.")
        return
        
    dynamic_polygon = None
    if draw_roi and mode in ["geofence", "loitering"]:
        dynamic_polygon = select_polygon(first_frame)
        if dynamic_polygon is not None:
            print("Đã vẽ xong polygon tùy chỉnh!")
        else:
            print("Đã hủy bỏ hoặc chưa vẽ đủ 3 điểm, sử dụng polygon mặc định.")
            
    processed_frame, count, alerts, total_tracks, enter_count, exit_count = process_frame(first_frame, frame_id, loiter_dict, mode=mode, dynamic_polygon=dynamic_polygon, flow_dict=flow_dict, enter_count=enter_count, exit_count=exit_count)
    cv2.putText(processed_frame, f"FPS: {fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow("Crowd Tracking & Loitering Detection", processed_frame)
    if output_path: out.write(processed_frame)
    frame_id += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã phát video xong hoặc không thể đọc frame.")
            break
            
        processed_frame, count, alerts, total_tracks, enter_count, exit_count = process_frame(frame, frame_id, loiter_dict, mode=mode, dynamic_polygon=dynamic_polygon, flow_dict=flow_dict, enter_count=enter_count, exit_count=exit_count)
        
        cv2.putText(processed_frame, f"FPS: {fps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
        cv2.imshow("Crowd Tracking & Loitering Detection", processed_frame)
        
        if output_path:
            out.write(processed_frame)
            
        frame_id += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print("Hoàn thành!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy phát hiện và theo dõi đám đông trên video")
    parser.add_argument("--video", type=str, required=True, help="Đường dẫn đến file video đầu vào")
    parser.add_argument("--output", type=str, default="", help="Đường dẫn lưu file video đầu ra (VD: output.mp4)")
    parser.add_argument("--mode", type=str, choices=["tracking", "geofence", "loitering"], 
                        default="loitering", help="Chọn chế độ chạy: tracking, geofence, hoặc loitering")
    parser.add_argument("--draw-roi", action="store_true", help="Bật chế độ tự vẽ hình đa giác (dùng chuột) ở frame đầu tiên.")
    
    args = parser.parse_args()
    
    run_on_video(args.video, args.output if args.output else None, mode=args.mode, draw_roi=args.draw_roi)
