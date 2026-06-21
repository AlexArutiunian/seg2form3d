from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sku_rgbd.measurement.geometry import resize_depth_context
from sku_rgbd.models import CameraIntrinsics


@dataclass(slots=True)
class SurfaceAnalysis:
    shape_class: str
    reject: bool
    metrics: dict


def _organized_xyz(
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    target_shape: tuple[int, int],
) -> np.ndarray:
    depth, intrinsics = resize_depth_context(depth, intrinsics, target_shape)
    z = depth.astype(np.float64) * depth_scale_m * 1000.0
    ys, xs = np.indices(target_shape, dtype=np.float64)
    x = (xs - intrinsics.cx) * z / intrinsics.fx
    y = (ys - intrinsics.cy) * z / intrinsics.fy
    return np.dstack((x, y, z))


def _points_and_normals(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    depth, adjusted_intrinsics = resize_depth_context(depth, intrinsics, mask.shape)
    filtered_depth = cv2.medianBlur(depth.astype(np.uint16), 5)
    xyz = _organized_xyz(
        filtered_depth, depth_scale_m, adjusted_intrinsics, mask.shape
    )
    valid_depth = xyz[:, :, 2] > 0
    radius = 3
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    valid = eroded & valid_depth
    valid[:radius, :] = False
    valid[-radius:, :] = False
    valid[:, :radius] = False
    valid[:, -radius:] = False

    dx = np.roll(xyz, -radius, axis=1) - np.roll(xyz, radius, axis=1)
    dy = np.roll(xyz, -radius, axis=0) - np.roll(xyz, radius, axis=0)
    normals = np.cross(dx, dy)
    lengths = np.linalg.norm(normals, axis=2)
    neighbor_jump = np.maximum(np.abs(dx[:, :, 2]), np.abs(dy[:, :, 2]))
    valid &= np.isfinite(lengths) & (lengths > 1e-6) & (neighbor_jump <= 60.0)
    normals[valid] /= lengths[valid, None]
    points = xyz[valid]
    normals = normals[valid]
    finite = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(normals), axis=1)
    return points[finite], normals[finite]


def _plane_coverage(
    points: np.ndarray,
    normals: np.ndarray,
    seed: int,
    max_planes: int = 6,
) -> tuple[float, list[float], np.ndarray]:
    count = len(points)
    remaining = np.ones(count, dtype=bool)
    plane_ratios = []
    rng = np.random.default_rng(seed)
    cos_threshold = np.cos(np.deg2rad(25.0))

    for _ in range(max_planes):
        indices = np.flatnonzero(remaining)
        if len(indices) < max(50, int(0.06 * count)):
            break
        if len(indices) > 2500:
            indices = rng.choice(indices, 2500, replace=False)
        sample_points = points[indices]
        sample_normals = normals[indices]

        best_mask = None
        best_count = 0
        for _ in range(72):
            anchor_index = int(rng.integers(len(sample_points)))
            normal = sample_normals[anchor_index]
            anchor = sample_points[anchor_index]
            distances = np.abs((sample_points - anchor) @ normal)
            alignment = np.abs(sample_normals @ normal)
            inliers = (distances <= 8.0) & (alignment >= cos_threshold)
            inlier_count = int(inliers.sum())
            if inlier_count > best_count:
                best_count = inlier_count
                best_mask = inliers
        if best_mask is None or best_count < max(35, int(0.06 * count)):
            break

        selected = indices[best_mask]
        selected_points = points[selected]
        centered = selected_points - np.mean(selected_points, axis=0)
        try:
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        normal = axes[-1]
        anchor = np.median(selected_points, axis=0)
        remaining_indices = np.flatnonzero(remaining)
        distances = np.abs((points[remaining_indices] - anchor) @ normal)
        alignment = np.abs(normals[remaining_indices] @ normal)
        full_inliers = remaining_indices[(distances <= 10.0) & (alignment >= cos_threshold)]
        ratio = len(full_inliers) / max(count, 1)
        if ratio < 0.05:
            break
        plane_ratios.append(float(ratio))
        remaining[full_inliers] = False

    coverage = 1.0 - float(np.mean(remaining))
    return coverage, plane_ratios, remaining


def _normal_direction_coverage(normals: np.ndarray) -> tuple[float, int]:
    if len(normals) < 20:
        return 0.0, 0
    directions = normals.copy()
    flip = directions[:, 2] < 0
    directions[flip] *= -1.0
    unused = np.ones(len(directions), dtype=bool)
    covered = 0
    clusters = 0
    cos_threshold = np.cos(np.deg2rad(15.0))
    while clusters < 6:
        indices = np.flatnonzero(unused)
        if not len(indices):
            break
        subset = directions[indices]
        candidate_ids = np.linspace(0, len(subset) - 1, min(80, len(subset))).astype(int)
        best = None
        best_count = 0
        for candidate_id in candidate_ids:
            alignment = np.abs(subset @ subset[candidate_id])
            count = int(np.sum(alignment >= cos_threshold))
            if count > best_count:
                best_count = count
                best = alignment >= cos_threshold
        if best is None or best_count < max(20, int(0.04 * len(directions))):
            break
        unused[indices[best]] = False
        covered += best_count
        clusters += 1
    return covered / len(directions), clusters


def analyze_surface_normals(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    instance_id: int,
    object_metrics: dict | None = None,
) -> SurfaceAnalysis:
    object_metrics = object_metrics or {}
    points, normals = _points_and_normals(
        mask, depth, depth_scale_m, intrinsics
    )
    metrics = {
        "surface_backend": "surface_normals_v1",
        "normal_points": int(len(points)),
    }
    if len(points) < 120:
        metrics.update(
            planar_coverage=0.0,
            curved_surface_ratio=1.0,
            normal_cluster_coverage=0.0,
            normal_cluster_count=0,
            roll_risk_score=1.0,
        )
        return SurfaceAnalysis("unknown_roll_risk", True, metrics)

    if len(points) > 6000:
        step = max(1, len(points) // 6000)
        points = points[::step][:6000]
        normals = normals[::step][:6000]

    planar_coverage, plane_ratios, curved_mask = _plane_coverage(
        points, normals, seed=104729 + instance_id
    )
    cluster_coverage, cluster_count = _normal_direction_coverage(normals)
    curved_ratio = float(np.mean(curved_mask))
    largest_plane = max(plane_ratios, default=0.0)

    roll_risk_score = float(
        np.clip(
            0.45 * curved_ratio
            + 0.25 * (1.0 - planar_coverage)
            + 0.20 * (1.0 - cluster_coverage)
            + 0.10 * (1.0 - largest_plane),
            0.0,
            1.0,
        )
    )
    metrics.update(
        planar_coverage=float(planar_coverage),
        curved_surface_ratio=curved_ratio,
        plane_count=len(plane_ratios),
        plane_ratios=plane_ratios,
        largest_plane_ratio=float(largest_plane),
        normal_cluster_coverage=float(cluster_coverage),
        normal_cluster_count=int(cluster_count),
        roll_risk_score=roll_risk_score,
    )

    stable_planar = (
        planar_coverage >= 0.68
        and cluster_coverage >= 0.72
        and curved_ratio <= 0.32
        and len(plane_ratios) >= 1
    )
    box_supported_planar = (
        planar_coverage >= 0.50
        and cluster_coverage >= 0.60
        and float(object_metrics.get("solidity") or 0.0) >= 0.86
        and float(object_metrics.get("rectangularity") or 0.0) >= 0.68
        and 4 <= int(object_metrics.get("approx_vertices") or 0) <= 10
        and float(object_metrics.get("pca_to_bbox_projection_ratio") or 99.0) <= 2.5
    )
    clear_curved = (
        curved_ratio >= 0.48
        or planar_coverage <= 0.50
        or cluster_coverage <= 0.55
        or roll_risk_score >= 0.52
    )
    metrics["box_supported_planar"] = bool(box_supported_planar)
    if stable_planar or box_supported_planar:
        return SurfaceAnalysis("planar_stable_shape", False, metrics)
    if clear_curved:
        return SurfaceAnalysis("roll_risk_curved_surface", True, metrics)
    return SurfaceAnalysis("unknown_roll_risk", True, metrics)
