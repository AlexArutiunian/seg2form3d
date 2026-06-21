from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sku_rgbd.measurement.geometry import resize_depth_context
from sku_rgbd.measurement.shape3d import fit_primitives
from sku_rgbd.models import CameraIntrinsics


@dataclass(slots=True)
class SurfaceAnalysis:
    shape_class: str
    reject: bool
    metrics: dict
    reasons: list[str]


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


def _depth_curvature_metrics(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
) -> dict:
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = mask.astype(bool) & (depth > 0)
    ys, xs = np.where(valid)
    if len(xs) < 120:
        return {
            "depth_fit_points": int(len(xs)),
            "depth_plane_residual_p80_mm": 0.0,
            "depth_quadratic_improvement": 0.0,
        }
    z = depth[valid].astype(np.float64) * depth_scale_m * 1000.0
    x = (xs - np.mean(xs)) / max(float(np.std(xs)), 1.0)
    y = (ys - np.mean(ys)) / max(float(np.std(ys)), 1.0)
    plane = np.column_stack((np.ones(len(x)), x, y))
    quadratic = np.column_stack((plane, x * x, x * y, y * y))
    try:
        plane_residual = z - plane @ np.linalg.lstsq(plane, z, rcond=None)[0]
        quadratic_residual = z - quadratic @ np.linalg.lstsq(quadratic, z, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {
            "depth_fit_points": int(len(xs)),
            "depth_plane_residual_p80_mm": 0.0,
            "depth_quadratic_improvement": 0.0,
        }
    plane_error = float(np.mean(plane_residual**2))
    quadratic_error = float(np.mean(quadratic_residual**2))
    improvement = 1.0 - quadratic_error / max(plane_error, 1e-9)
    return {
        "depth_fit_points": int(len(xs)),
        "depth_plane_residual_p80_mm": float(np.percentile(np.abs(plane_residual), 80.0)),
        "depth_quadratic_improvement": float(np.clip(improvement, 0.0, 1.0)),
    }


def _cross_section_metrics(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
) -> dict:
    if depth.shape != mask.shape:
        depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    valid = mask.astype(bool) & (depth > 0)
    ys, xs = np.where(valid)
    defaults = {
        "cross_section_r2_median": 0.0,
        "cross_section_curved_slice_ratio": 0.0,
        "cross_section_sign_consistency": 0.0,
        "cross_section_abs_depth_correlation_median": 0.0,
        "cross_section_valid_slices": 0,
        "cross_section_rotational_support": False,
    }
    if len(xs) < 120:
        return defaults
    pixels = np.column_stack((xs, ys)).astype(np.float64)
    centered = pixels - np.mean(pixels, axis=0)
    try:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return defaults
    coordinates = centered @ axes.T
    major, transverse = coordinates[:, 0], coordinates[:, 1]
    z = depth[valid].astype(np.float64) * depth_scale_m * 1000.0
    low, high = np.percentile(major, (10.0, 90.0))
    edges = np.linspace(low, high, 7)
    r2_values, signs, correlations = [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        selected = (major >= start) & (major < stop)
        if int(selected.sum()) < 30:
            continue
        local_x = transverse[selected]
        local_z = z[selected]
        if float(np.ptp(local_x)) < 6.0:
            continue
        local_x = local_x - np.mean(local_x)
        design = np.column_stack((local_x * local_x, local_x, np.ones(len(local_x))))
        try:
            coefficients = np.linalg.lstsq(design, local_z, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        predicted = design @ coefficients
        linear = np.column_stack((local_x, np.ones(len(local_x))))
        linear_residual_std = float(
            np.std(local_z - linear @ np.linalg.lstsq(linear, local_z, rcond=None)[0])
        )
        if linear_residual_std < 0.5:
            continue
        total = float(np.sum((local_z - np.mean(local_z)) ** 2))
        r2 = 1.0 - float(np.sum((local_z - predicted) ** 2)) / max(total, 1e-9)
        correlation = np.corrcoef(np.abs(local_x), local_z)[0, 1]
        r2_values.append(float(np.clip(r2, 0.0, 1.0)))
        signs.append(float(np.sign(coefficients[0])))
        correlations.append(abs(float(correlation)) if np.isfinite(correlation) else 0.0)
    if not r2_values:
        return defaults
    r2_median = float(np.median(r2_values))
    curved_ratio = float(np.mean(np.asarray(r2_values) >= 0.45))
    positive = float(np.mean(np.asarray(signs) > 0))
    sign_consistency = max(positive, 1.0 - positive)
    correlation_median = float(np.median(correlations))
    support = (
        r2_median >= 0.45
        and curved_ratio >= 0.50
        and sign_consistency >= 0.80
        and correlation_median >= 0.55
    )
    return {
        "cross_section_r2_median": r2_median,
        "cross_section_curved_slice_ratio": curved_ratio,
        "cross_section_sign_consistency": sign_consistency,
        "cross_section_abs_depth_correlation_median": correlation_median,
        "cross_section_valid_slices": len(r2_values),
        "cross_section_rotational_support": bool(support),
    }


def _value(metrics: dict, name: str, default=0.0):
    value = metrics.get(name, default)
    return default if value is None else value


def evaluate_roll_risk(
    metrics: dict,
    length: float,
    width: float,
    height: float,
    source_class_name: str = "",
) -> list[str]:
    circularity = float(_value(metrics, "circularity"))
    solidity = float(_value(metrics, "solidity"))
    rectangularity = float(_value(metrics, "rectangularity"))
    vertices = int(_value(metrics, "approx_vertices", 0))
    bbox = _value(metrics, "bbox_xywh", [0, 0, 0, 0])
    bbox_aspect = max(float(bbox[2]), float(bbox[3])) / max(1.0, min(float(bbox[2]), float(bbox[3])))
    elongation = length / max(width, 1e-6)
    height_width = height / max(width, 1e-6)
    curved = float(_value(metrics, "curved_surface_ratio"))
    planar = float(_value(metrics, "planar_coverage"))
    normal_coverage = float(_value(metrics, "normal_cluster_coverage"))
    largest_plane = float(_value(metrics, "largest_plane_ratio"))
    depth_improvement = float(_value(metrics, "depth_quadratic_improvement"))
    depth_residual = float(_value(metrics, "depth_plane_residual_p80_mm"))

    very_strong_planar_face = planar >= 0.90 and largest_plane >= 0.60 and curved <= 0.12
    box_like_planar_face = (
        planar >= 0.82 and largest_plane >= 0.60 and curved <= 0.18
        and rectangularity >= 0.78 and elongation <= 2.50 and 4 <= vertices <= 8
    )
    planar_suppression = very_strong_planar_face or box_like_planar_face
    reasons = []

    confidence = float(_value(metrics, "axis_rotational_confidence"))
    radial_inliers = float(_value(metrics, "axis_rotational_radial_inlier_ratio"))
    radial_residual = float(_value(metrics, "axis_rotational_radial_residual_p80_norm", 1.0))
    angular_coverage = float(_value(metrics, "axis_rotational_angular_coverage_deg"))
    radius_cv = float(_value(metrics, "axis_rotational_slice_radius_cv", 1.0))
    valid_slices = int(_value(metrics, "axis_rotational_valid_slices", 0))
    support_min = float(_value(metrics, "axis_rotational_radial_support_ratio_min"))
    support_max = float(_value(metrics, "axis_rotational_radial_support_ratio_max", 99.0))

    if elongation >= 2.2:
        strong_axis = (
            confidence >= 0.50 and radial_inliers >= 0.72 and radial_residual <= 0.18
            and angular_coverage >= 120.0 and radius_cv <= 0.10 and valid_slices >= 4
            and support_min >= 0.55 and support_max <= 1.40
        )
        if strong_axis and not planar_suppression:
            reasons.append("axis_rotational_roll_risk")
        weak_axis = (
            depth_improvement >= 0.38 and depth_residual >= 3.0
            and angular_coverage >= 120.0 and radius_cv <= 0.10 and valid_slices >= 4
            and 5 <= vertices <= 12
        )
        weak_confirmation = (
            confidence >= 0.35 and radial_inliers >= 0.60 and radial_residual <= 0.22
            and support_min >= 0.50 and support_max <= 1.45
        )
        if weak_axis and weak_confirmation and not planar_suppression:
            reasons.append("axis_supported_rotational_roll_risk")
        partial_axis = (
            confidence >= 0.48 and radial_inliers >= 0.80 and radial_residual <= 0.16
            and angular_coverage >= 120.0 and support_min >= 0.20 and support_max <= 1.45
        )
        if partial_axis and not planar_suppression:
            reasons.append("partial_axis_rotational_roll_risk")

    cross_section = bool(_value(metrics, "cross_section_rotational_support", False))
    if cross_section and height_width >= 0.25 and solidity >= 0.80 and not planar_suppression:
        reasons.append("cross_section_rotational_roll_risk")

    compact_candidate = (
        circularity >= 0.70 and solidity >= 0.93 and 6 <= vertices <= 12
        and 1.25 <= bbox_aspect <= 2.40 and height_width >= 0.55
        and depth_improvement >= 0.40 and depth_residual >= 3.0 and largest_plane <= 0.65
    )
    if compact_candidate and not planar_suppression:
        reasons.append("compact_rotational_roll_risk")
    strong_compact = circularity >= 0.84 and solidity >= 0.93 and elongation <= 1.35 and height_width >= 0.35
    if strong_compact and not planar_suppression:
        reasons.append("strong_compact_round_roll_risk")
    compact_round = (
        circularity >= 0.72 and solidity >= 0.94 and 6 <= vertices <= 12
        and elongation <= 1.35 and height_width >= 0.45 and rectangularity <= 0.90
        and largest_plane <= 0.55
        and (curved >= 0.18 or depth_improvement >= 0.15 or depth_residual >= 6.0)
    )
    if compact_round and not planar_suppression:
        reasons.append("compact_round_object_roll_risk")

    semantic_tokens = ("bottle", "can", "cylinder", "tube", "roll", "ball", "sphere")
    semantic = any(token in source_class_name.lower() for token in semantic_tokens)
    if semantic and (curved >= 0.25 or depth_improvement >= 0.15) and not planar_suppression:
        reasons.append("semantic_rotational_roll_risk")
    return list(dict.fromkeys(reasons))


def analyze_surface_normals(
    mask: np.ndarray,
    depth: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    instance_id: int,
    object_metrics: dict | None = None,
    source_class_name: str = "",
) -> SurfaceAnalysis:
    object_metrics = object_metrics or {}
    points, normals = _points_and_normals(
        mask, depth, depth_scale_m, intrinsics
    )
    metrics = {
        "surface_backend": "surface_normals_v2",
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
        return SurfaceAnalysis("unknown_roll_risk", True, metrics, ["unknown_roll_risk"])

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
    metrics.update(_depth_curvature_metrics(mask, depth, depth_scale_m))
    metrics.update(_cross_section_metrics(mask, depth, depth_scale_m))

    length = float(_value(object_metrics, "shape_length_mm"))
    width = float(_value(object_metrics, "shape_width_mm"))
    height = float(_value(object_metrics, "shape_height_mm"))
    elongation = length / max(width, 1e-6)
    if elongation >= 2.2:
        primitive = fit_primitives(points)
        primitive_metrics = primitive.metrics
        mapping = {
            "axis_rotational_confidence": "primitive_cylinder_score",
            "axis_rotational_radial_inlier_ratio": "cylinder_radial_inlier_ratio",
            "axis_rotational_radial_residual_p80_norm": "cylinder_radial_residual_p80_norm",
            "axis_rotational_angular_coverage_deg": "cylinder_angular_coverage_deg",
            "axis_rotational_slice_radius_cv": "cylinder_slice_radius_cv",
            "axis_rotational_valid_slices": "cylinder_valid_slices",
            "axis_rotational_radial_support_ratio_min": "cylinder_radial_support_ratio_min",
            "axis_rotational_radial_support_ratio_max": "cylinder_radial_support_ratio_max",
        }
        for target, source in mapping.items():
            metrics[target] = primitive_metrics.get(source)
        metrics["axis_rotational_primitive_class"] = primitive.shape_class

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
    metrics["stable_planar"] = bool(stable_planar)
    metrics["box_supported_planar"] = bool(box_supported_planar)
    metrics["clear_curved"] = bool(clear_curved)
    reasons = evaluate_roll_risk(metrics, length, width, height, source_class_name)
    metrics["roll_risk_reasons"] = reasons
    if reasons:
        return SurfaceAnalysis(reasons[0], True, metrics, reasons)
    if stable_planar or box_supported_planar:
        return SurfaceAnalysis("planar_stable_shape", False, metrics, [])
    return SurfaceAnalysis("unknown_roll_risk", False, metrics, [])
