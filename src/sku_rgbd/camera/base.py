from __future__ import annotations

from abc import ABC, abstractmethod

from sku_rgbd.models import RGBDFrame


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> None:
        """Start acquisition."""

    @abstractmethod
    def read(self) -> RGBDFrame:
        """Return the next synchronized RGB-D frame."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition and release resources."""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

