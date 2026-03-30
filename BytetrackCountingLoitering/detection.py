# detection.py
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from config import CONF_THRESHOLD

@dataclass
class YoloOutput:
    frame_id: int
    boxes_xyxy: np.ndarray
    confidences: np.ndarray
    class_ids: np.ndarray
    centers: List[Tuple[float, float]]

model = YOLO("yolov8s.pt")  # Use YOLOv8s for edge devices

def detect(model, frame, frame_id):
    results = model(frame, classes=[0, 1, 3], conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    conf = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    output = YoloOutput(
        frame_id=frame_id,
        boxes_xyxy=boxes,
        confidences=conf,
        class_ids=class_ids,
        centers=centers
    )
    return output