"""Video file camera source - dùng file video MP4/AVI."""
import cv2
from .base import BaseCameraSource


class VideoFileCameraSource(BaseCameraSource):
    """Đọc từ video file để test AI pipeline."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._cap = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.source_url}")

    def read(self):
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            # Loop video khi hết
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


class OpenCVCameraSource(BaseCameraSource):
    """Webcam local qua OpenCV."""

    def __init__(self, source_url: str = "0"):
        self.device_index = int(source_url) if source_url.isdigit() else 0
        self._cap = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)

    def read(self):
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


def create_camera_source(camera_config: dict):
    """Factory tạo camera source từ camera_config."""
    source_type = camera_config.get("source_type", "mock")
    source_url = camera_config.get("source_url", "")

    if source_type == "mock":
        from .mock_camera import MockCameraSource
        return MockCameraSource()
    elif source_type in ["video_file", "file"]:
        return VideoFileCameraSource(source_url)
    elif source_type == "webcam":
        return OpenCVCameraSource(source_url)
    elif source_type == "rtsp":
        return VideoFileCameraSource(source_url)  # OpenCV hỗ trợ RTSP
    else:
        raise ValueError(f"Unknown source_type: {source_type}")
