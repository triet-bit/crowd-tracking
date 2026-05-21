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
from .base import BaseAIEngine, FrameResult

# BytetrackCountingLoitering/main.py dùng `from config import ...`
# Cần thêm thư mục vào sys.path để Python tìm thấy config.py
_bytetrack_dir = os.path.join(os.path.dirname(__file__), "../../../../BytetrackCountingLoitering")
_bytetrack_dir = os.path.abspath(_bytetrack_dir)
if _bytetrack_dir not in sys.path:
    sys.path.insert(0, _bytetrack_dir)

_crowd_counting_dir = os.path.join(os.path.dirname(__file__), "../../../../CrowdCounting")
_crowd_counting_dir = os.path.abspath(_crowd_counting_dir)
if _crowd_counting_dir not in sys.path:
    sys.path.insert(0, _crowd_counting_dir)

# Import pipeline AI thật từ repo
from BytetrackCountingLoitering.main import process_frame as _process_frame


class CrowdTrackingAIEngine(BaseAIEngine):
    _crowd_counting_model = None
    _crowd_counting_device = None

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
        visual_mode = camera_config.get("visual_mode", "bytetrack")
        loitering_time_threshold = camera_config.get("loitering_time_threshold")
        velocity_threshold = camera_config.get("velocity_threshold")

        previous_visual_mode = runtime_state.get("visual_mode")
        if previous_visual_mode != visual_mode:
            runtime_state["visual_mode"] = visual_mode
            runtime_state["loiter_dict"] = {}
            runtime_state["flow_dict"] = {}
            runtime_state["enter_count"] = 0
            runtime_state["exit_count"] = 0

        if visual_mode == "heatmap":
            try:
                return self._process_heatmap(frame, camera_id, frame_id)
            except Exception as exc:
                return self._build_heatmap_error_result(frame, camera_id, frame_id, exc)

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
        polygon_signature = json.dumps(polygon_raw, sort_keys=True) if polygon_raw is not None else None
        previous_signature = runtime_state.get("polygon_signature")

        if polygon_signature != previous_signature:
            loiter_dict.clear()
            flow_dict.clear()
            runtime_state["polygon_signature"] = polygon_signature

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
            loitering_time_threshold=loitering_time_threshold,
            velocity_threshold=velocity_threshold,
            use_default_polygon=False,
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

    @classmethod
    def _load_crowd_counting_model(cls):
        if cls._crowd_counting_model is not None:
            return cls._crowd_counting_model, cls._crowd_counting_device

        import torch
        from huggingface_hub import snapshot_download
        from model import CSRNet

        checkpoint_dir = os.path.join(_crowd_counting_dir, "checkpoints")
        checkpoint_name = "xinnguoihayvenoiday105.pth"
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)

        if not os.path.exists(checkpoint_file):
            snapshot_download(
                repo_id="b1nswing/CSRNET_config_B",
                local_dir=checkpoint_dir,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CSRNet().to(device)
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        cls._crowd_counting_model = model
        cls._crowd_counting_device = device
        return cls._crowd_counting_model, cls._crowd_counting_device

    def _process_heatmap(self, frame, camera_id: str, frame_id: int) -> FrameResult:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as func
        import torchvision.transforms.functional as TF

        model, device = self._load_crowd_counting_model()
        original_h, original_w = frame.shape[:2]

        img_resized = cv2.resize(frame, (432, 240))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = TF.to_tensor(img_rgb)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            output = torch.relu(output)
            pred_count = torch.sum(output).item()
            output_upscaled = func.interpolate(
                output,
                size=(img_tensor.size(2), img_tensor.size(3)),
                mode="bilinear",
                align_corners=False,
            )

        density_map = output_upscaled.cpu().squeeze().numpy()
        heatmap_norm = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_norm = np.uint8(heatmap_norm)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
        processed_frame = cv2.resize(blended, (original_w, original_h))

        cv2.putText(
            processed_frame,
            f"Heatmap Count: {pred_count:.1f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return FrameResult(
            camera_id=camera_id,
            frame_id=frame_id,
            processed_frame=processed_frame,
            people_count=int(round(pred_count)),
            total_tracks=0,
            enter_count=0,
            exit_count=0,
            boxes=[],
            alerts=[],
        )

    def _build_heatmap_error_result(self, frame, camera_id: str, frame_id: int, exc: Exception) -> FrameResult:
        import cv2

        processed_frame = frame.copy()
        cv2.putText(
            processed_frame,
            "Heatmap unavailable",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            processed_frame,
            str(exc)[:80],
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        return FrameResult(
            camera_id=camera_id,
            frame_id=frame_id,
            processed_frame=processed_frame,
            people_count=0,
            total_tracks=0,
            enter_count=0,
            exit_count=0,
            boxes=[],
            alerts=[],
        )
