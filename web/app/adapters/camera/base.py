"""Base camera source interface."""


class BaseCameraSource:
    def open(self) -> None:
        raise NotImplementedError

    def read(self):
        """Returns (success: bool, frame: ndarray)"""
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    def is_opened(self) -> bool:
        raise NotImplementedError
