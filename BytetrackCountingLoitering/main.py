# main.py
import cv2
import numpy as np
import time

from config import polygon, CONF_THRESHOLD
from detection import model, detect
from tracking import convert_to_bytetrack, tracker, update_tracks

from utils import count_in_polygon, check_flow_crossing
from loitering import check_loitering, loiter_dict

def process_frame(
    frame,
    frame_id,
    loiter_dict,
    mode="loitering",
    dynamic_polygon=None,
    flow_dict=None,
    enter_count=0,
    exit_count=0,
    loitering_time_threshold=None,
    velocity_threshold=None,
    use_default_polygon=True,
):
    current_polygon = dynamic_polygon if dynamic_polygon is not None else (polygon if use_default_polygon else None)
    
    yolo_output = detect(model, frame, frame_id)
    detections = convert_to_bytetrack(yolo_output, tracked_class_ids=(0,))
    tracks = update_tracks(tracker, detections, frame.shape)

    # Draw polygon for debug and count
    count = 0
    if mode in ["geofence", "loitering"] and current_polygon is not None:
        cv2.polylines(frame, [current_polygon], True, (255,0,0), 2)
        # Count only persons (class 0) in polygon
        count = count_in_polygon(tracks, current_polygon)

    # Flow counting
    if flow_dict is not None and current_polygon is not None:
        for track in tracks:
            x1, y1, x2, y2, track_id, class_id = track
            if class_id == 0:  # Only for persons
                current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                previous_center = flow_dict.get(track_id)
                crossing = check_flow_crossing(track_id, current_center, previous_center, current_polygon)
                if crossing == "enter":
                    enter_count += 1
                elif crossing == "exit":
                    exit_count += 1
                flow_dict[track_id] = current_center

    # Loitering
    current_time = time.time()
    alerts = []
    if mode == "loitering" and current_polygon is not None:
        alerts = check_loitering(
            tracks,
            current_polygon,
            loiter_dict,
            current_time,
            threshold_time=loitering_time_threshold,
            velocity_threshold=velocity_threshold,
        )

    # Draw tracks with color based on loitering
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track
        color = (0,0,255) if track_id in alerts else (0,255,0)
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(frame, f"ID {track_id} C{class_id}", (int(x1),int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display count if applicable
    if mode in ["geofence", "loitering"]:
        # Đã đổi màu sang Vàng (B=0, G=255, R=255)
        cv2.putText(frame, f"Count: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        if flow_dict is not None:
            cv2.putText(frame, f"Enter: {enter_count} Exit: {exit_count}", (10,70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    return frame, count, alerts, len(tracks), enter_count, exit_count

# Example usage with image
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load image from Shanghai dataset
    image_path = r"Shanghaidataset\shanghaitech\shanghaitech\testing\frames\01_0014\186.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found!")
    else:
        print("Image loaded successfully!")
        processed_frame, count, alerts, total_tracks, _, _ = process_frame(frame, 0, loiter_dict)
        plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
        plt.title("Processed Image with Detections, Tracks, and Count")
        plt.show()

        print(f"Total detections: {total_tracks}")
        print(f"Active tracks: {total_tracks}")
        print(f"Count in polygon (persons only): {count}")
        print(f"Loitering alerts: {alerts}")
