from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PrimitiveFit:
    shape_class: str
    confidence: float
    metrics: dict


def _fit_circle(points_2d: np.ndarray) -> tuple[np.ndarray, float] | None:
    if len(points_2d) < 12:
        return None
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points_2d))))
    rhs = x * x + y * y
    try:
        solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = solution[:2]
    radius_sq = float(solution[2] + center @ center)
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        return None
    return center, math.sqrt(radius_sq)


def _circle_from_three(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    first, second, third = points
    matrix = 2.0 * np.array((second - first, third - first))
    rhs = np.array(
        (
            float(second @ second - first @ first),
            float(third @ third - first @ first),
        )
    )
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-6:
        return None
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return None
    radius = float(np.linalg.norm(first - center))
    if not np.isfinite(radius) or radius <= 2.0 or radius > 1000.0:
        return None
    return center, radius


def _fit_circle_ransac(
    points_2d: np.ndarray,
    seed: int,
    iterations: int = 48,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(points_2d) < 30:
        return None
    if len(points_2d) > 1200:
        step = max(1, len(points_2d) // 1200)
        sample_points = points_2d[::step][:1200]
    else:
        sample_points = points_2d

    rng = np.random.default_rng(seed)
    best = None
    best_rank = (-1.0, float("-inf"))
    for _ in range(iterations):
        indices = rng.choice(len(sample_points), size=3, replace=False)
        circle = _circle_from_three(sample_points[indices])
        if circle is None:
            continue
        center, radius = circle
        tolerance = max(4.0, 0.06 * radius)
        residuals = np.abs(np.linalg.norm(sample_points - center, axis=1) - radius)
        inliers = residuals <= tolerance
        inlier_ratio = float(np.mean(inliers))
        if int(inliers.sum()) < 20:
            continue
        residual_p80 = float(np.percentile(residuals[inliers], 80.0))
        rank = (inlier_ratio, -residual_p80 / max(radius, 1e-6))
        if rank > best_rank:
            best_rank = rank
            best = (center, radius)

    if best is None:
        return None
    center, radius = best
    tolerance = max(4.0, 0.06 * radius)
    residuals = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
    inliers = residuals <= tolerance
    refined = _fit_circle(points_2d[inliers])
    if refined is not None:
        center, radius = refined
        tolerance = max(4.0, 0.06 * radius)
        residuals = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
        inliers = residuals <= tolerance
    return center, radius, inliers


def _angular_coverage(angles: np.ndarray) -> float:
    if len(angles) < 3:
        return 0.0
    values = np.sort(np.mod(angles, 2.0 * math.pi))
    gaps = np.diff(np.concatenate((values, values[:1] + 2.0 * math.pi)))
    return float(2.0 * math.pi - gaps.max())


def _box_surface_score(projected: np.ndarray) -> tuple[float, dict]:
    low = np.percentile(projected, 2.0, axis=0)
    high = np.percentile(projected, 98.0, axis=0)
    extents = np.maximum(high - low, 1e-6)
    face_distance = np.minimum(np.abs(projected - low), np.abs(high - projected))
    nearest_face = np.min(face_distance, axis=1)
    scale = max(5.0, float(np.min(extents)))
    normalized = nearest_face / scale
    p80 = float(np.percentile(normalized, 80.0))
    inlier_ratio = float(np.mean(normalized <= 0.08))
    score = float(np.clip(0.65 * inlier_ratio + 0.35 * (1.0 - p80 / 0.20), 0.0, 1.0))
    return score, {
        "box_surface_inlier_ratio": inlier_ratio,
        "box_surface_residual_p80_norm": p80,
    }


def _cylinder_candidate(
    projected: np.ndarray,
    axis_index: int,
    seed: int,
    robust: bool = False,
) -> dict | None:
    radial_indices = [index for index in range(3) if index != axis_index]
    radial_points = projected[:, radial_indices]
    if robust:
        fitted = _fit_circle_ransac(radial_points, seed)
        if fitted is None:
            return None
        center, radius, initial_inliers = fitted
    else:
        circle = _fit_circle(radial_points)
        if circle is None:
            return None
        center, radius = circle
        initial_inliers = None
    radial_vectors = radial_points - center
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    axis_values = projected[:, axis_index]
    axis_low, axis_high = np.percentile(axis_values, (2.0, 98.0))
    axis_length = float(axis_high - axis_low)
    if axis_length <= 0 or not np.isfinite(axis_length):
        return None

    fixed_residuals = np.abs(radial_distances - radius)
    fixed_tolerance = max(4.0, 0.06 * radius)
    fixed_inliers = (
        fixed_residuals <= fixed_tolerance if initial_inliers is None else initial_inliers
    )

    design = np.column_stack((axis_values, np.ones(len(axis_values))))
    keep = np.ones(len(axis_values), dtype=bool)
    slope = 0.0
    intercept = radius
    for _ in range(3):
        try:
            coefficients, *_ = np.linalg.lstsq(
                design[keep], radial_distances[keep], rcond=None
            )
        except np.linalg.LinAlgError:
            break
        slope, intercept = (float(value) for value in coefficients)
        predicted = design @ coefficients
        conical_residuals = np.abs(radial_distances - predicted)
        threshold = max(4.0, float(np.percentile(conical_residuals, 65.0)))
        keep = conical_residuals <= threshold
        if int(keep.sum()) < 30:
            break
    predicted_radii = slope * axis_values + intercept
    median_radius = float(np.median(predicted_radii))
    conical_residuals = np.abs(radial_distances - predicted_radii)
    conical_tolerance = max(4.0, 0.06 * max(median_radius, 1.0))
    conical_inliers = conical_residuals <= conical_tolerance

    def fit_quality(residuals: np.ndarray, inliers: np.ndarray, reference_radius: float):
        residual_p80 = float(np.percentile(residuals, 80.0))
        residual_norm = residual_p80 / max(reference_radius, 1e-6)
        inlier_ratio = float(np.mean(inliers))
        score = float(
            0.58 * np.clip((inlier_ratio - 0.40) / 0.55, 0.0, 1.0)
            + 0.42 * np.clip(1.0 - residual_norm / 0.18, 0.0, 1.0)
        )
        return score, residual_p80, residual_norm, inlier_ratio

    fixed_quality = fit_quality(fixed_residuals, fixed_inliers, radius)
    conical_quality = fit_quality(
        conical_residuals, conical_inliers, max(median_radius, 1.0)
    )
    use_conical = (
        conical_quality[0] >= fixed_quality[0] + 0.08
        and abs(slope) * axis_length / max(median_radius, 1e-6) <= 1.5
        and np.min(predicted_radii) > 2.0
    )
    if use_conical:
        radius = median_radius
        residuals = conical_residuals
        inliers = conical_inliers
        rotational_fit = "linear_radius"
    else:
        residuals = fixed_residuals
        inliers = fixed_inliers
        rotational_fit = "constant_radius"

    angles = np.arctan2(radial_vectors[:, 1], radial_vectors[:, 0])
    coverage = _angular_coverage(angles[inliers])

    slice_radii = []
    slice_positions = []
    edges = np.linspace(axis_low, axis_high, 7)
    for start, stop in zip(edges[:-1], edges[1:]):
        selected = (axis_values >= start) & (axis_values < stop) & inliers
        if int(selected.sum()) >= 20:
            slice_radii.append(float(np.median(radial_distances[selected])))
            slice_positions.append(float(np.median(axis_values[selected])))
    if len(slice_radii) >= 3:
        if use_conical:
            expected = slope * np.asarray(slice_positions) + intercept
            radius_cv = float(
                np.std(np.asarray(slice_radii) - expected)
                / max(np.mean(expected), 1e-6)
            )
        else:
            radius_cv = float(np.std(slice_radii) / max(np.mean(slice_radii), 1e-6))
    else:
        radius_cv = 1.0

    residual_p80 = float(np.percentile(residuals, 80.0))
    residual_norm = residual_p80 / max(radius, 1e-6)
    inlier_ratio = float(np.mean(inliers))
    coverage_deg = math.degrees(coverage)
    radius_ratio = radius / axis_length
    radial_spans = np.percentile(radial_points, 98.0, axis=0) - np.percentile(
        radial_points, 2.0, axis=0
    )
    diameter = 2.0 * radius
    radial_support_ratios = radial_spans / max(diameter, 1e-6)

    residual_score = np.clip(1.0 - residual_norm / 0.10, 0.0, 1.0)
    inlier_score = np.clip((inlier_ratio - 0.50) / 0.40, 0.0, 1.0)
    coverage_score = np.clip((coverage_deg - 55.0) / 145.0, 0.0, 1.0)
    stability_score = np.clip(1.0 - radius_cv / 0.18, 0.0, 1.0)
    confidence = float(
        0.35 * residual_score
        + 0.30 * inlier_score
        + 0.20 * coverage_score
        + 0.15 * stability_score
    )
    return {
        "axis_index": axis_index,
        "radius_mm": float(radius),
        "axis_length_mm": axis_length,
        "radius_axis_ratio": float(radius_ratio),
        "radial_span_1_mm": float(radial_spans[0]),
        "radial_span_2_mm": float(radial_spans[1]),
        "radial_support_ratio_min": float(np.min(radial_support_ratios)),
        "radial_support_ratio_max": float(np.max(radial_support_ratios)),
        "radial_residual_p80_mm": residual_p80,
        "radial_residual_p80_norm": float(residual_norm),
        "radial_inlier_ratio": inlier_ratio,
        "angular_coverage_deg": float(coverage_deg),
        "slice_radius_cv": radius_cv,
        "valid_slices": len(slice_radii),
        "confidence": confidence,
        "circle_fit": "ransac" if robust else "least_squares",
        "rotational_fit": rotational_fit,
        "radius_slope_mm_per_mm": slope if use_conical else 0.0,
        "radius_change_ratio": (
            abs(slope) * axis_length / max(radius, 1e-6) if use_conical else 0.0
        ),
    }


def _basis_from_axis(axis: np.ndarray) -> np.ndarray:
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ helper)) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    radial_1 = np.cross(axis, helper)
    radial_1 /= max(np.linalg.norm(radial_1), 1e-12)
    radial_2 = np.cross(axis, radial_1)
    return np.vstack((axis, radial_1, radial_2))


def _axis_candidates(pca_axes: np.ndarray) -> list[np.ndarray]:
    candidates = []
    offsets = (-0.45, -0.22, 0.0, 0.22, 0.45)
    for index in range(3):
        axis = pca_axes[index]
        tangent_1 = pca_axes[(index + 1) % 3]
        tangent_2 = pca_axes[(index + 2) % 3]
        for first in offsets:
            for second in offsets:
                direction = axis + first * tangent_1 + second * tangent_2
                direction /= max(np.linalg.norm(direction), 1e-12)
                candidates.append(direction)
    return candidates


def fit_primitives(points: np.ndarray, max_points: int = 8000) -> PrimitiveFit:
    metrics: dict = {"primitive_backend": "strict_3d"}
    if len(points) < 120:
        return PrimitiveFit("unknown_shape", 0.0, metrics)

    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step][:max_points]

    center = np.median(points, axis=0)
    centered = points - center
    try:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return PrimitiveFit("unknown_shape", 0.0, metrics)
    projected = centered @ axes.T

    low = np.percentile(projected, 1.0, axis=0)
    high = np.percentile(projected, 99.0, axis=0)
    keep = np.all((projected >= low) & (projected <= high), axis=1)
    projected = projected[keep]
    if len(projected) < 120:
        return PrimitiveFit("unknown_shape", 0.0, metrics)

    box_score, box_metrics = _box_surface_score(projected)
    fit_points = centered[keep]
    if len(fit_points) > 2500:
        fit_points = fit_points[:: max(1, len(fit_points) // 2500)][:2500]
    rough_candidates = []
    for candidate_index, axis in enumerate(_axis_candidates(axes)):
        candidate_projection = fit_points @ _basis_from_axis(axis).T
        candidate = _cylinder_candidate(candidate_projection, 0, candidate_index)
        if candidate is not None:
            candidate["axis_candidate_index"] = candidate_index
            candidate["axis_abs_camera_z"] = abs(float(axis[2]))
            candidate["_axis"] = axis
            rough_candidates.append(candidate)
    metrics.update(box_metrics)
    if not rough_candidates:
        metrics["primitive_box_score"] = box_score
        return PrimitiveFit("unknown_shape", 0.0, metrics)

    rough_candidates.sort(key=lambda item: item["confidence"], reverse=True)
    candidates = []
    for rough in rough_candidates[:12]:
        axis = rough.pop("_axis")
        candidate_projection = fit_points @ _basis_from_axis(axis).T
        candidate = _cylinder_candidate(
            candidate_projection,
            0,
            seed=7919 + int(rough["axis_candidate_index"]),
            robust=True,
        )
        if candidate is not None:
            candidate["axis_candidate_index"] = rough["axis_candidate_index"]
            candidate["axis_abs_camera_z"] = abs(float(axis[2]))
            candidates.append(candidate)
    if not candidates:
        metrics["primitive_box_score"] = box_score
        return PrimitiveFit("unknown_shape", 0.0, metrics)

    best = max(candidates, key=lambda item: item["confidence"])
    metrics.update({f"cylinder_{key}": value for key, value in best.items()})
    metrics["primitive_box_score"] = box_score
    metrics["primitive_cylinder_score"] = best["confidence"]
    metrics["primitive_score_margin"] = best["confidence"] - box_score

    strict_cylinder = (
        best["radial_inlier_ratio"] >= 0.72
        and best["radial_residual_p80_norm"] <= 0.08
        and best["angular_coverage_deg"] >= 75.0
        and best["slice_radius_cv"] <= 0.12
        and best["valid_slices"] >= 3
        and 0.08 <= best["radius_axis_ratio"] <= 1.25
        and best["radial_support_ratio_min"] >= 0.35
        and best["radial_support_ratio_max"] <= 1.35
        and best["axis_abs_camera_z"] <= 0.90
        and best["confidence"] >= 0.72
        and best["confidence"] >= box_score + 0.08
    )
    if strict_cylinder:
        metrics["cylinder_decision_tier"] = "strict"
        return PrimitiveFit("cylindrical_shape", best["confidence"], metrics)

    clear_box = box_score >= 0.68 and box_score >= best["confidence"] - 0.02
    if clear_box:
        return PrimitiveFit("box_or_rectangular_shape", box_score, metrics)
    return PrimitiveFit("unknown_shape", max(box_score, best["confidence"]), metrics)
