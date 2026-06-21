from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from sku_rgbd.measurement import MeasurementEngine
from sku_rgbd.models import MeasurementResult, RGBDFrame, SegmentationResult
from sku_rgbd.segmentation import RemoteSegmentationClient
from sku_rgbd.ui import draw_measurements


@dataclass
class BackendSnapshot:
    segmentation: SegmentationResult | None = None
    measurements: list[MeasurementResult] | None = None
    overlay: object | None = None
    status: str = "starting"


class BackendWorker(threading.Thread):
    def __init__(
        self,
        client: RemoteSegmentationClient,
        measurement: MeasurementEngine,
        target_fps: float,
    ):
        super().__init__(daemon=True)
        self.client = client
        self.measurement = measurement
        self.target_fps = target_fps
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.latest_frame: RGBDFrame | None = None
        self.latest = BackendSnapshot(measurements=[])

    def submit(self, frame: RGBDFrame) -> None:
        with self.lock:
            self.latest_frame = frame

    def snapshot(self) -> BackendSnapshot:
        with self.lock:
            return BackendSnapshot(
                segmentation=self.latest.segmentation,
                measurements=list(self.latest.measurements or []),
                overlay=None if self.latest.overlay is None else self.latest.overlay.copy(),
                status=self.latest.status,
            )

    def run(self) -> None:
        interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_run = 0.0
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if now < next_run:
                time.sleep(min(0.01, next_run - now))
                continue
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            next_run = now + interval
            try:
                segmentation = self.client.segment(frame.frame_id, frame.color_bgr)
                measurements = self.measurement.measure_all(frame, segmentation.instances)
                overlay = draw_measurements(frame.color_bgr, segmentation, measurements)
                status = (
                    f"seg {segmentation.infer_ms:.0f} ms | "
                    f"roundtrip {segmentation.roundtrip_ms:.0f} ms | "
                    f"{len(measurements)} measured"
                )
                with self.lock:
                    self.latest = BackendSnapshot(segmentation, measurements, overlay, status)
            except Exception as exc:
                self.client.close()
                with self.lock:
                    self.latest.status = f"retry: {type(exc).__name__}: {exc}"
                time.sleep(1.0)

    def stop(self) -> None:
        self.stop_event.set()
        self.client.close()

