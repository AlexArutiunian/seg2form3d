import numpy as np

from sku_rgbd.measurement import MeasurementEngine
from sku_rgbd.measurement.shape3d import fit_primitives


def test_box_priority_over_circularity():
    engine = MeasurementEngine()
    metrics = {
        "circularity": 0.77,
        "solidity": 0.98,
        "extent": 0.72,
        "rectangularity": 0.83,
        "approx_vertices": 6,
    }
    shape, reasons = engine._classify_shape(metrics, 220.0, 155.0, 95.0)
    assert shape == "box_or_rectangular_shape"
    assert reasons == []


def test_true_round_candidate():
    engine = MeasurementEngine()
    metrics = {
        "circularity": 0.91,
        "solidity": 0.98,
        "extent": 0.78,
        "rectangularity": 0.60,
        "approx_vertices": 12,
    }
    shape, reasons = engine._classify_shape(metrics, 105.0, 95.0, 80.0)
    assert shape == "spherical_or_round_shape"
    assert reasons == ["spherical_or_round_shape"]


def test_cylindrical_candidate_is_not_box():
    engine = MeasurementEngine()
    metrics = {
        "circularity": 0.798,
        "solidity": 0.97,
        "extent": 0.79,
        "rectangularity": 0.80,
        "approx_vertices": 8,
    }
    shape, reasons = engine._classify_shape(metrics, 102.0, 67.0, 56.0)
    assert shape == "cylindrical_shape"
    assert reasons == ["cylindrical_shape"]


def test_size_limits():
    engine = MeasurementEngine()
    assert engine._size_reasons(100.0, 80.0, 40.0) == []
    assert engine._size_reasons(500.0, 80.0, 5.0) == [
        "dim1_above_450mm",
        "dim3_below_10mm",
    ]


def test_strict_3d_detects_cylinder():
    rng = np.random.default_rng(7)
    axis = rng.uniform(-90.0, 90.0, 5000)
    angle = rng.uniform(-1.35, 1.35, 5000)
    radius = 42.0 + rng.normal(0.0, 1.0, 5000)
    points = np.column_stack(
        (
            axis,
            radius * np.cos(angle),
            radius * np.sin(angle),
        )
    )
    fitted = fit_primitives(points)
    assert fitted.shape_class == "cylindrical_shape"
    assert fitted.metrics["cylinder_radial_inlier_ratio"] > 0.9


def test_strict_3d_does_not_call_box_cylinder():
    rng = np.random.default_rng(11)
    count = 1800
    top = np.column_stack(
        (
            rng.uniform(-100.0, 100.0, count),
            rng.uniform(-70.0, 70.0, count),
            np.full(count, 45.0) + rng.normal(0.0, 0.8, count),
        )
    )
    front = np.column_stack(
        (
            rng.uniform(-100.0, 100.0, count),
            np.full(count, -70.0) + rng.normal(0.0, 0.8, count),
            rng.uniform(-45.0, 45.0, count),
        )
    )
    side = np.column_stack(
        (
            np.full(count, 100.0) + rng.normal(0.0, 0.8, count),
            rng.uniform(-70.0, 70.0, count),
            rng.uniform(-45.0, 45.0, count),
        )
    )
    fitted = fit_primitives(np.vstack((top, front, side)))
    assert fitted.shape_class != "cylindrical_shape"


def test_partial_cylinder_can_use_3d_plus_silhouette_support():
    rng = np.random.default_rng(23)
    axis = rng.uniform(-45.0, 45.0, 4500)
    angle = rng.uniform(-1.7, 1.7, 4500)
    radius = 36.0 + rng.normal(0.0, 4.0, 4500)
    points = np.column_stack((axis, radius * np.cos(angle), radius * np.sin(angle)))
    engine = MeasurementEngine()
    metrics = {
        "circularity": 0.82,
        "solidity": 0.97,
        "extent": 0.78,
        "rectangularity": 0.79,
        "approx_vertices": 8,
    }
    shape, reasons = engine._classify_shape(metrics, 100.0, 70.0, 58.0, points)
    assert shape == "cylindrical_shape"
    assert reasons == ["cylindrical_shape"]
