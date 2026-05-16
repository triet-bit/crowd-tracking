"""
Base AI Engine interface.
All AI engines must implement this interface.
"""
from dataclasses import dataclass, field
from typing import Any, List
from datetime import datetime


@dataclass
class FrameResult:
    """Kết quả trả về từ AI engine sau mỗi frame."""
    camera_id: str
    frame_id: int
    processed_frame: Any          # numpy ndarray
    people_count: int
    total_tracks: int
    enter_count: int
    exit_count: int
    boxes: List[Any] = field(default_factory=list)
    alerts: List[Any] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_metrics_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "people_count": self.people_count,
            "total_tracks": self.total_tracks,
            "enter_count": self.enter_count,
            "exit_count": self.exit_count,
            "timestamp": self.timestamp,
        }

    def to_alert_payload(self, zone_id: str = "default", threshold: int = 10) -> dict:
        """Build alert payload nếu count vượt ngưỡng."""
        severity = "critical" if self.people_count > threshold * 2 else \
                   "high" if self.people_count > threshold else "medium"
        return {
            "camera_id": self.camera_id,
            "zone_id": zone_id,
            "alert_type": "threshold_exceeded",
            "severity": severity,
            "message": f"People count exceeded threshold: {self.people_count} > {threshold}",
            "people_count": self.people_count,
            "frame_id": self.frame_id,
        }

    def to_loitering_payload(self, zone_id: str = "default") -> dict:
        return {
            "camera_id": self.camera_id,
            "zone_id": zone_id,
            "alert_type": "loitering",
            "severity": "medium",
            "message": f"Loitering detected: tracks {self.alerts}",
            "people_count": self.people_count,
            "frame_id": self.frame_id,
        }


class BaseAIEngine:
    """Abstract base class for all AI engines."""

    def process(self, frame, camera_config: dict, runtime_state: dict) -> FrameResult:
        """
        Process a single video frame.

        Args:
            frame: numpy ndarray - ảnh frame từ camera
            camera_config: dict - thông tin camera (id, mode, polygon, threshold...)
            runtime_state: dict - trạng thái liên frame (loiter_dict, flow_dict, counters...)

        Returns:
            FrameResult
        """
        raise NotImplementedError
