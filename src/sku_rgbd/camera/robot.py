from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from sku_rgbd.camera.base import CameraSource
from sku_rgbd.models import CameraIntrinsics, RGBDFrame


class RobotCameraSource(CameraSource):
    def __init__(
        self,
        robot_ip: str,
        port: int = 8088,
        camera_name: str = "Camera-000",
        api_path: str | None = None,
        retries: int = 5,
        retry_sleep: float = 0.2,
    ):
        self.robot_ip = robot_ip
        self.port = port
        self.camera_name = camera_name
        self.api_path = api_path
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.api = None
        self.frame_id = 0

    def start(self) -> None:
        if self.api_path:
            path = str(Path(self.api_path).expanduser().resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            from robotcontrol_api import RobotcontrolAPI
        except ImportError as exc:
            raise RuntimeError(
                "RobotcontrolAPI not found. Set --robot-api-path to the folder containing robotcontrol_api.py."
            ) from exc
        self.api = RobotcontrolAPI(self.robot_ip, port=self.port)

    def _get_camera_data(self):
        last_state = None
        for _ in range(max(1, self.retries)):
            state, data = self.api.getCameraData(camera_name=self.camera_name)
            last_state = state
            if data is not None and getattr(data, "rgb", None) and getattr(data, "depth", None):
                return data
            time.sleep(self.retry_sleep)
        raise RuntimeError(f"Robot camera read failed, state={last_state}")

    def read(self) -> RGBDFrame:
        if self.api is None:
            raise RuntimeError("Robot camera source is not started")
        data = self._get_camera_data()
        rgb_width = int(data.rgb.width)
        rgb_height = int(data.rgb.height)
        depth_width = int(data.depth.width)
        depth_height = int(data.depth.height)

        color_bgr = np.frombuffer(data.rgb.data, dtype=np.uint8).reshape(rgb_height, rgb_width, 3).copy()
        depth_raw = np.frombuffer(data.depth.data, dtype=np.uint16).reshape(depth_height, depth_width).copy()
        intrinsic = data.calib.intrinsic
        color_intr = CameraIntrinsics(
            width=rgb_width,
            height=rgb_height,
            fx=float(intrinsic.fx),
            fy=float(intrinsic.fy),
            cx=float(intrinsic.cx),
            cy=float(intrinsic.cy),
        )
        depth_intrinsic = getattr(data.calib, "intrinsic_ir", None) or intrinsic
        depth_intr = CameraIntrinsics(
            width=depth_width,
            height=depth_height,
            fx=float(depth_intrinsic.fx),
            fy=float(depth_intrinsic.fy),
            cx=float(depth_intrinsic.cx),
            cy=float(depth_intrinsic.cy),
        )

        # The current Seer API stream is treated as registered RGB-D, matching the existing working client.
        depth_aligned = depth_raw.copy()
        self.frame_id += 1
        return RGBDFrame(
            frame_id=self.frame_id,
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_raw=depth_raw,
            depth_aligned_to_color=depth_aligned,
            depth_scale_m=float(getattr(data.depth, "scale", 0.001)),
            color_intrinsics=color_intr,
            depth_intrinsics=depth_intr,
            metadata={
                "source": "robot",
                "camera_name": self.camera_name,
                "rgb_resolution": [rgb_width, rgb_height],
                "depth_resolution": [depth_width, depth_height],
            },
        )

    def stop(self) -> None:
        self.api = None
