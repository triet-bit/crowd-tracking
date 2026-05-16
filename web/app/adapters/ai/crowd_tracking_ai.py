"""
CrowdTrackingAIEngine - tích hợp AI thật từ BytetrackCountingLoitering.

Adapter này wrap process_frame() từ repo hiện tại và convert
kết quả sang FrameResult chuẩn để phần còn lại của hệ thống
không cần biết chi tiết bên trong AI pipeline.

Kích hoạt bằng: AI_ENGINE=crowd_tracking trong .env
"""
import sys
import os
import json
import numpy as np
from .base import BaseAIEngine, FrameResult

# BytetrackCountingLoitering/main.py dùng `from config import ...`
# Cần thêm thư mục vào sys.path để Python tìm thấy config.py
_bytetrack_dir = os.path.join(os.path.dirname(__file__), "../../../../BytetrackCountingLoitering")
_bytetrack_dir = os.path.abspath(_bytetrack_dir)
if _bytetrack_dir not in sys.path:
    sys.path.insert(0, _bytetrack_dir)

# Import pipeline AI thật từ repo
from BytetrackCountingLoitering.main import process_frame as _process_frame


class CrowdTrackingAIEngine(BaseAIEngine):
    """
    AI Engine sử dụng pipeline ByteTrack + YOLO từ repo hiện tại.

    Pipeline:
        frame (numpy ndarray)
            -> YOLO detection
            -> ByteTrack tracking
            -> Geofence / Loitering check
        Returns: (processed_frame, count, alerts, total_tracks, enter_count, exit_count)

    runtime_state phải chứa:
        - frame_id: int
        - loiter_dict: dict (lưu thời gian từng track ở trong polygon)
        - flow_dict: dict (lưu vị trí trước của từng track để đếm enter/exit)
        - enter_count: int
        - exit_count: int
    """

    def process(self, frame, camera_config: dict, runtime_state: dict) -> FrameResult:
        camera_id = camera_config.get("id", "cam_unknown")
        frame_id = runtime_state.get("frame_id", 0)
        mode = camera_config.get("mode", "loitering")

        # Lấy polygon từ camera_config (JSON string hoặc list)
        polygon_raw = camera_config.get("polygon_json") or camera_config.get("polygon")
        dynamic_polygon = None
        if polygon_raw:
            try:
                pts = json.loads(polygon_raw) if isinstance(polygon_raw, str) else polygon_raw
                import numpy as np
                dynamic_polygon = np.array(pts, dtype=np.int32)
            except Exception:
                dynamic_polygon = None

        # Lấy state liên frame
        loiter_dict = runtime_state.setdefault("loiter_dict", {})
        flow_dict = runtime_state.setdefault("flow_dict", {})
        enter_count = runtime_state.get("enter_count", 0)
        exit_count = runtime_state.get("exit_count", 0)

        # Gọi process_frame() từ repo gốc
        processed_frame, count, alerts, total_tracks, enter_count, exit_count = _process_frame(
            frame=frame,
            frame_id=frame_id,
            loiter_dict=loiter_dict,
            mode=mode,
            dynamic_polygon=dynamic_polygon,
            flow_dict=flow_dict,
            enter_count=enter_count,
            exit_count=exit_count,
        )

        # Cập nhật runtime state
        runtime_state["enter_count"] = enter_count
        runtime_state["exit_count"] = exit_count

        # Convert alerts từ list track_id sang list string
        alert_list = [f"loitering:track_{a}" for a in alerts] if alerts else []

        # Thêm alert nếu threshold_exceeded
        threshold = camera_config.get("max_people_threshold", 10)
        if count > threshold:
            alert_list.append(f"threshold_exceeded:{count}>{threshold}")

        return FrameResult(
            camera_id=camera_id,
            frame_id=frame_id,
            processed_frame=processed_frame,
            people_count=count,
            total_tracks=total_tracks,
            enter_count=enter_count,
            exit_count=exit_count,
            boxes=[],          # boxes đã được vẽ trực tiếp lên processed_frame
            alerts=alert_list,
        )
