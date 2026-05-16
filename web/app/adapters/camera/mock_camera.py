"""Mock camera source - sinh frame giả dùng numpy."""
import numpy as np
import cv2
import time
from .base import BaseCameraSource


class MockCameraSource(BaseCameraSource):
    """Sinh frame giả để test pipeline khi không có camera thật."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._frame_count = 0
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def read(self):
        if not self._opened:
            return False, None
        self._frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Vẽ nền gradient
        frame[:, :, 0] = 30
        frame[:, :, 1] = 40
        frame[:, :, 2] = 60
        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, f"MOCK CAMERA - {ts}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(frame, f"Frame: {self._frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        time.sleep(0.033)  # ~30 FPS
        return True, frame

    def release(self) -> None:
        self._opened = False

    def is_opened(self) -> bool:
        return self._opened
