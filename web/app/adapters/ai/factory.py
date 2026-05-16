"""
AI Adapter Factory - chọn AI engine theo config .env.
"""
from .base import BaseAIEngine
from .mock_ai import MockAIEngine


def create_ai_engine(settings=None) -> BaseAIEngine:
    """
    Factory function để tạo AI engine dựa trên AI_ENGINE trong .env.

    AI_ENGINE=mock           -> MockAIEngine (random data, không cần model)
    AI_ENGINE=crowd_tracking -> CrowdTrackingAIEngine (YOLO + ByteTrack thật)
    """
    if settings is None:
        from web.app.core.config import settings as _settings
        settings = _settings

    engine_type = settings.AI_ENGINE.lower()

    if engine_type == "mock":
        return MockAIEngine()

    if engine_type == "crowd_tracking":
        try:
            from .crowd_tracking_ai import CrowdTrackingAIEngine
            return CrowdTrackingAIEngine()
        except ImportError as e:
            import logging
            logging.warning(
                f"[AI Factory] Không thể load CrowdTrackingAIEngine: {e}. "
                f"Fallback về MockAIEngine."
            )
            return MockAIEngine()

    raise ValueError(f"Unknown AI_ENGINE: '{settings.AI_ENGINE}'. Dùng 'mock' hoặc 'crowd_tracking'.")
