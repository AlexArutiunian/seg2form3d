import numpy as np

from sku_rgbd.measurement.shape_normals import (
    _cross_section_metrics,
    evaluate_roll_risk,
)


def base_metrics():
    return {
        "circularity": 0.60,
        "solidity": 0.96,
        "rectangularity": 0.82,
        "approx_vertices": 6,
        "bbox_xywh": [10, 10, 160, 80],
        "curved_surface_ratio": 0.10,
        "planar_coverage": 0.92,
        "normal_cluster_coverage": 0.85,
        "largest_plane_ratio": 0.70,
        "depth_quadratic_improvement": 0.05,
        "depth_plane_residual_p80_mm": 1.0,
    }


def test_planar_box_is_not_rejected():
    assert evaluate_roll_risk(base_metrics(), 200.0, 120.0, 70.0) == []


def test_strong_axis_cylinder_is_rejected():
    metrics = base_metrics()
    metrics.update(
        planar_coverage=0.35,
        largest_plane_ratio=0.20,
        curved_surface_ratio=0.60,
        approx_vertices=10,
        axis_rotational_confidence=0.70,
        axis_rotational_radial_inlier_ratio=0.85,
        axis_rotational_radial_residual_p80_norm=0.10,
        axis_rotational_angular_coverage_deg=160.0,
        axis_rotational_slice_radius_cv=0.06,
        axis_rotational_valid_slices=6,
        axis_rotational_radial_support_ratio_min=0.70,
        axis_rotational_radial_support_ratio_max=1.20,
    )
    reasons = evaluate_roll_risk(metrics, 240.0, 80.0, 65.0)
    assert "axis_rotational_roll_risk" in reasons


def test_planar_face_suppresses_false_cross_section_candidate():
    metrics = base_metrics()
    metrics["cross_section_rotational_support"] = True
    assert evaluate_roll_risk(metrics, 200.0, 120.0, 70.0) == []


def test_compact_round_object_is_rejected():
    metrics = base_metrics()
    metrics.update(
        circularity=0.88,
        solidity=0.97,
        rectangularity=0.70,
        approx_vertices=10,
        bbox_xywh=[10, 10, 120, 90],
        curved_surface_ratio=0.35,
        planar_coverage=0.40,
        largest_plane_ratio=0.25,
        depth_quadratic_improvement=0.45,
        depth_plane_residual_p80_mm=7.0,
    )
    reasons = evaluate_roll_risk(metrics, 110.0, 90.0, 60.0)
    assert "strong_compact_round_roll_risk" in reasons
    assert "compact_round_object_roll_risk" in reasons


def test_parabolic_depth_has_cross_section_rotational_support():
    height, width = 100, 220
    mask = np.zeros((height, width), bool)
    mask[20:80, 20:200] = True
    ys, _ = np.indices(mask.shape)
    depth_mm = 900.0 + 0.025 * (ys - 50.0) ** 2
    metrics = _cross_section_metrics(mask, depth_mm.astype(np.uint16), 0.001)
    assert metrics["cross_section_valid_slices"] >= 4
    assert metrics["cross_section_rotational_support"] is True
