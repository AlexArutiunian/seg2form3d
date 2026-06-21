from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str = ""
    coeffs: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "model": self.model,
            "coeffs": self.coeffs,
        }


@dataclass(slots=True)
class RGBDFrame:
    frame_id: int
    timestamp: float
    color_bgr: np.ndarray
    depth_raw: np.ndarray
    depth_aligned_to_color: np.ndarray
    depth_scale_m: float
    color_intrinsics: CameraIntrinsics
    depth_intrinsics: CameraIntrinsics
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SegmentationInstance:
    instance_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float = 0.0
    source_class_name: str = ""
    track_id: int | None = None

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass(slots=True)
class SegmentationResult:
    backend: str
    frame_id: int
    instances: list[SegmentationInstance]
    infer_ms: float
    roundtrip_ms: float
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MeasurementResult:
    instance_id: int
    length_mm: float | None
    width_mm: float | None
    height_mm: float | None
    center_xyz_mm: tuple[float | None, float | None, float | None]
    decision: str
    reasons: list[str]
    shape_class: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        x, y, z = self.center_xyz_mm
        return {
            "instance_id": self.instance_id,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "center_x_mm": x,
            "center_y_mm": y,
            "center_z_mm": z,
            "decision": self.decision,
            "reasons": self.reasons,
            "shape_class": self.shape_class,
            "metrics": self.metrics,
        }

