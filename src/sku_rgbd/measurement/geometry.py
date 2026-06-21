from __future__ import annotations

import math

import cv2
import numpy as np

from sku_rgbd.models import CameraIntrinsics


def largest_contour(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def contour_metrics(mask: np.ndarray) -> dict:
    contour = largest_contour(mask)
    if contour is None or len(contour) < 3:
        return {}
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    rect_area = float(rw * rh)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return {
        "contour_area_px": area,
        "perimeter_px": perimeter,
        "circularity": 4.0 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else None,
        "solidity": area / hull_area if hull_area > 0 else None,
        "extent": float(mask.sum()) / (w * h) if w > 0 and h > 0 else None,
        "rectangularity": area / rect_area if rect_area > 0 else None,
        "approx_vertices": int(len(approx)),
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
    }


def resize_depth_context(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, CameraIntrinsics]:
    target_h, target_w = target_shape
    if depth.shape == target_shape:
        return depth, intrinsics
    scale_x = target_w / depth.shape[1]
    scale_y = target_h / depth.shape[0]
    resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    adjusted = CameraIntrinsics(
        width=target_w,
        height=target_h,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        model=intrinsics.model,
        coeffs=list(intrinsics.coeffs),
    )
    return resized, adjusted


def unproject_mask(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    depth, intrinsics = resize_depth_context(depth, intrinsics, mask.shape)
    valid = mask & (depth > 0)
    ys, xs = np.where(valid)
    if not len(xs):
        return np.empty((0, 3), dtype=np.float64)
    z_mm = depth[ys, xs].astype(np.float64) * depth_scale_m * 1000.0
    finite = np.isfinite(z_mm) & (z_mm > 0)
    xs = xs[finite].astype(np.float64)
    ys = ys[finite].astype(np.float64)
    z_mm = z_mm[finite]
    x_mm = (xs - intrinsics.cx) * z_mm / intrinsics.fx
    y_mm = (ys - intrinsics.cy) * z_mm / intrinsics.fy
    return np.column_stack((x_mm, y_mm, z_mm))


def robust_pca_extents(points: np.ndarray, low: float = 2.0, high: float = 98.0):
    if len(points) < 4:
        return None, None
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes.T
    extents = np.percentile(projected, high, axis=0) - np.percentile(projected, low, axis=0)
    extents = sorted((float(value) for value in extents if np.isfinite(value)), reverse=True)
    if len(extents) != 3:
        return None, None
    return tuple(extents), axes


def mask_ring(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask


def local_background_height(
    mask: np.ndarray,
    object_points: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float | None, dict]:
    for radius in (12, 20, 32):
        ring_points = unproject_mask(mask_ring(mask, radius), depth, depth_scale_m, intrinsics)
        if len(ring_points) < 80:
            continue
        if len(ring_points) > 3000:
            ring_points = ring_points[:: max(1, len(ring_points) // 3000)]
        xy1 = np.column_stack((ring_points[:, 0], ring_points[:, 1], np.ones(len(ring_points))))
        try:
            coef, *_ = np.linalg.lstsq(xy1, ring_points[:, 2], rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = np.abs(xy1 @ coef - ring_points[:, 2])
        keep = residual <= max(12.0, float(np.percentile(residual, 70)))
        if int(keep.sum()) < 80:
            continue
        coef, *_ = np.linalg.lstsq(xy1[keep], ring_points[keep, 2], rcond=None)
        residual_mm = float(np.median(np.abs(xy1[keep] @ coef - ring_points[keep, 2])))
        if residual_mm > 35.0:
            continue
        plane_z = coef[0] * object_points[:, 0] + coef[1] * object_points[:, 1] + coef[2]
        positive = plane_z - object_points[:, 2]
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if len(positive) < max(20, int(0.02 * len(object_points))):
            continue
        return float(np.percentile(positive, 95)), {
            "local_plane_radius_px": radius,
            "local_plane_residual_mm": residual_mm,
            "local_plane_support_px": int(keep.sum()),
        }
    return None, {
        "local_plane_radius_px": None,
        "local_plane_residual_mm": None,
        "local_plane_support_px": 0,
    }

