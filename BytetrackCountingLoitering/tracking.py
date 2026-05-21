# tracking.py
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

# ByteTrack goc dung np.float, alias nay bi go trong NumPy moi.
if not hasattr(np, "float"):
    np.float = float

# Add original ByteTrack repo path.
# Expected structure:
# crowd-tracking_Do_An/
#   ByteTrack/
#     yolox/
#       tracker/
#         byte_tracker.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BYTETRACK_ROOT = PROJECT_ROOT / "ByteTrack"
sys.path.insert(0, str(BYTETRACK_ROOT))

from yolox.tracker.byte_tracker import BYTETracker


def create_tracker(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False, frame_rate=30):
    args = Namespace(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        mot20=mot20,
    )
    return BYTETracker(args, frame_rate=frame_rate)


tracker = create_tracker()


def convert_to_bytetrack(yolo_output, tracked_class_ids=(0,)):
    detections = []
    tracked_class_ids = set(tracked_class_ids)

    for i in range(len(yolo_output.boxes_xyxy)):
        class_id = int(yolo_output.class_ids[i])
        if class_id not in tracked_class_ids:
            continue

        x1, y1, x2, y2 = yolo_output.boxes_xyxy[i]
        conf = yolo_output.confidences[i]
        detections.append([x1, y1, x2, y2, conf])

    if not detections:
        return np.empty((0, 5), dtype=np.float32)

    return np.asarray(detections, dtype=np.float32)


def update_tracks(byte_tracker, detections, frame_shape):
    height, width = frame_shape[:2]
    online_targets = byte_tracker.update(detections, [height, width], (height, width))

    tracks = []
    for target in online_targets:
        x1, y1, x2, y2 = target.tlbr
        track_id = int(target.track_id)
        class_id = 0
        tracks.append([x1, y1, x2, y2, track_id, class_id])

    return tracks
