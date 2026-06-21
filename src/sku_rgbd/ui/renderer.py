from __future__ import annotations

import cv2
import numpy as np

from sku_rgbd.models import MeasurementResult, RGBDFrame, SegmentationResult


COLORS = {
    "accept": (60, 220, 70),
    "reject": (45, 55, 245),
    "unknown": (0, 220, 255),
    "robot": (255, 180, 0),
}


def depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = depth > 0
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], (2, 98))
        if hi <= lo:
            hi = lo + 1
        normalized = np.clip((depth.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def panel(image: np.ndarray, title: str, status: str = "") -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (0, 255, 0), 2, cv2.LINE_AA)
    if status:
        cv2.putText(out, status[:100], (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_measurements(
    frame: np.ndarray,
    segmentation: SegmentationResult,
    measurements: list[MeasurementResult],
    alpha: float = 0.42,
) -> np.ndarray:
    out = frame.copy()
    fill = frame.copy()
    measured = {item.instance_id: item for item in measurements}
    for instance in segmentation.instances:
        result = measured.get(instance.instance_id)
        if result is None:
            continue
        color = COLORS.get(result.decision, COLORS["unknown"])
        fill[instance.mask] = color
        contours, _ = cv2.findContours(instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
        x1, y1, x2, y2 = instance.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        dims = "/".join("?" if value is None else f"{value:.0f}" for value in (
            result.length_mm,
            result.width_mm,
            result.height_mm,
        ))
        line1 = f"#{instance.instance_id} {result.decision.upper()} {dims}mm"
        line2 = ",".join(result.reasons) if result.reasons else result.shape_class
        cv2.putText(out, line1, (x1, max(20, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
        cv2.putText(out, line2[:55], (x1, max(38, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
    blended = cv2.addWeighted(out, 1 - alpha, fill, alpha, 0)
    return blended


def build_grid(
    frame: RGBDFrame,
    sam3_overlay: np.ndarray | None,
    sam3_status: str,
    rtdm_overlay: np.ndarray | None,
    rtdm_status: str,
    save_status: str,
) -> np.ndarray:
    color = frame.color_bgr
    blank = np.zeros_like(color)
    depth = depth_preview(frame.depth_aligned_to_color)
    if depth.shape[:2] != color.shape[:2]:
        depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    top = np.hstack((panel(color, "RGB", save_status), panel(depth, "Aligned depth")))
    bottom = np.hstack(
        (
            panel(sam3_overlay if sam3_overlay is not None else blank, "SAM3", sam3_status),
            panel(rtdm_overlay if rtdm_overlay is not None else blank, "RTMDet / RTDM", rtdm_status),
        )
    )
    return np.vstack((top, bottom))

