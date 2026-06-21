from __future__ import annotations

import time

import cv2
import numpy as np

from sku_rgbd.camera.base import CameraSource
from sku_rgbd.models import CameraIntrinsics, RGBDFrame


def _intrinsics(profile) -> CameraIntrinsics:
    intr = profile.as_video_stream_profile().get_intrinsics()
    return CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        model=str(intr.model),
        coeffs=[float(value) for value in intr.coeffs],
    )


class RealSenseSource(CameraSource):
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 15, serial: str | None = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.pipeline = None
        self.profile = None
        self.align = None
        self.depth_scale = 0.001
        self.frame_id = 0

    def start(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("Install the realsense extra: pip install -e '.[realsense]'") from exc

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = float(self.profile.get_device().first_depth_sensor().get_depth_scale())
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def read(self) -> RGBDFrame:
        if self.pipeline is None or self.align is None:
            raise RuntimeError("RealSense source is not started")

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        aligned_depth_frame = aligned.get_depth_frame()
        raw_depth_frame = frames.get_depth_frame()
        if not color_frame or not aligned_depth_frame or not raw_depth_frame:
            raise RuntimeError("Incomplete RealSense RGB-D frameset")

        self.frame_id += 1
        color_rgb = np.asanyarray(color_frame.get_data())
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        depth_raw = np.asanyarray(raw_depth_frame.get_data()).copy()
        depth_aligned = np.asanyarray(aligned_depth_frame.get_data()).copy()
        device = self.profile.get_device()
        return RGBDFrame(
            frame_id=self.frame_id,
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_aligned_to_color=depth_aligned,
            depth_scale_m=self.depth_scale,
            color_intrinsics=_intrinsics(color_frame.profile),
            depth_intrinsics=_intrinsics(raw_depth_frame.profile),
            metadata={
                "source": "realsense",
                "device_name": device.get_info(__import__("pyrealsense2").camera_info.name),
                "serial": device.get_info(__import__("pyrealsense2").camera_info.serial_number),
                "usb_type": device.get_info(__import__("pyrealsense2").camera_info.usb_type_descriptor),
            },
        )

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None

