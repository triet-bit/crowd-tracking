"""
Mock AI Engine - dùng khi chưa chạy model thật.
Sinh dữ liệu ngẫu nhiên để test frontend và pipeline.
"""
import random
import numpy as np
import cv2
from .base import BaseAIEngine, FrameResult


class MockAIEngine(BaseAIEngine):
    """
    AI engine giả - sinh dữ liệu random.
    Dùng khi AI_ENGINE=mock trong .env.
    """

    def process(self, frame, camera_config: dict, runtime_state: dict) -> FrameResult:
        camera_id = camera_config.get("id", "cam_unknown")
        frame_id = runtime_state.get("frame_id", 0)

        # Sinh count ngẫu nhiên (tăng dần nhẹ theo thời gian)
        base_count = random.randint(0, 15)
        people_count = base_count
        total_tracks = people_count + random.randint(0, 3)

        # Cập nhật enter/exit count
        prev_count = runtime_state.get("prev_count", 0)
        enter_delta = max(0, people_count - prev_count)
        exit_delta = max(0, prev_count - people_count)
        enter_count = runtime_state.get("enter_count", 0) + enter_delta
        exit_count = runtime_state.get("exit_count", 0) + exit_delta
        runtime_state["prev_count"] = people_count
        runtime_state["enter_count"] = enter_count
        runtime_state["exit_count"] = exit_count

        # Vẽ thông tin lên frame (nếu có)
        processed_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(processed_frame, f"[MOCK] Count: {people_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Enter:{enter_count} Exit:{exit_count}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Alerts nếu count vượt ngưỡng
        threshold = camera_config.get("max_people_threshold", 10)
        alerts = []
        if people_count > threshold:
            alerts.append(f"threshold_exceeded:{people_count}>{threshold}")

        # Fake boxes
        h, w = processed_frame.shape[:2]
        boxes = [
            [random.randint(0, w // 2), random.randint(0, h // 2),
             random.randint(w // 2, w), random.randint(h // 2, h)]
            for _ in range(min(people_count, 5))
        ]

        return FrameResult(
            camera_id=camera_id,
            frame_id=frame_id,
            processed_frame=processed_frame,
            people_count=people_count,
            total_tracks=total_tracks,
            enter_count=enter_count,
            exit_count=exit_count,
            boxes=boxes,
            alerts=alerts,
        )
