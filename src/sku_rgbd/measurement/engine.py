from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sku_rgbd.measurement.geometry import (
    contour_metrics,
    local_background_height,
    robust_pca_extents,
    unproject_mask,
)
from sku_rgbd.measurement.shape3d import fit_primitives
from sku_rgbd.measurement.shape_normals import analyze_surface_normals
from sku_rgbd.models import MeasurementResult, RGBDFrame, SegmentationInstance


ROBOT_TOKENS = ("robot_hand", "robot hand", "robot_arm", "robot arm", "gripper", "manipulator")


@dataclass(slots=True)
class MeasurementConfig:
    min_dimension_mm: float = 10.0
    max_dimension_mm: float = 450.0
    max_center_z_mm: float = 1500.0
    min_mask_area_px: int = 80
    min_depth_points: int = 80
    min_depth_valid_ratio: float = 0.15
    reject_by_shape: bool = True
    reject_flat: bool = True
    reject_elongated: bool = True
    reject_round: bool = True
    pca_low_percentile: float = 2.0
    pca_high_percentile: float = 98.0
    shape_backend: str = "strict_3d"


class MeasurementEngine:
    def __init__(self, config: MeasurementConfig | None = None):
        self.config = config or MeasurementConfig()

    def measure(self, frame: RGBDFrame, instance: SegmentationInstance) -> MeasurementResult | None:
        if instance.area < self.config.min_mask_area_px:
            return None
        source_name = instance.source_class_name.lower().replace("-", "_")
        if any(token in source_name for token in ROBOT_TOKENS):
            return MeasurementResult(
                instance_id=instance.instance_id,
                length_mm=None,
                width_mm=None,
                height_mm=None,
                center_xyz_mm=(None, None, None),
                decision="robot",
                reasons=["robot_manipulator"],
                shape_class="robot_manipulator",
            )

        metrics = contour_metrics(instance.mask)
        points = unproject_mask(
            instance.mask,
            frame.depth_aligned_to_color,
            frame.depth_scale_m,
            frame.color_intrinsics,
        )
        valid_ratio = len(points) / max(1, instance.area)
        metrics["valid_depth_px"] = int(len(points))
        metrics["depth_valid_ratio"] = float(valid_ratio)
        if len(points) < self.config.min_depth_points:
            return self._unknown(instance, metrics, "insufficient_depth_points")

        center = tuple(float(value) for value in np.median(points, axis=0))
        if center[2] > self.config.max_center_z_mm:
            return None

        extents, _axes = robust_pca_extents(
            points,
            self.config.pca_low_percentile,
            self.config.pca_high_percentile,
        )
        if extents is None:
            return self._unknown(instance, metrics, "pca_failed", center)
        length, width, pca_height = extents
        plane_height, plane_metrics = local_background_height(
            instance.mask,
            points,
            frame.depth_aligned_to_color,
            frame.depth_scale_m,
            frame.color_intrinsics,
        )
        metrics.update(plane_metrics)
        metrics["pca_extents_mm"] = [length, width, pca_height]
        metrics["pca_height_mm"] = pca_height
        metrics["local_plane_height_mm"] = plane_height

        # Primary dimensions are the three orientation-independent robust PCA/OBB extents.
        # Local-plane height remains diagnostic because clutter around a mask can bias the fitted plane.
        dimensions = sorted((float(length), float(width), float(pca_height)), reverse=True)
        length, width, height = dimensions
        bbox = metrics.get("bbox_xywh") or [0, 0, 0, 0]
        projected_width_mm = float(bbox[2]) * center[2] / max(frame.color_intrinsics.fx, 1e-6)
        projected_height_mm = float(bbox[3]) * center[2] / max(frame.color_intrinsics.fy, 1e-6)
        projected_span_mm = max(projected_width_mm, projected_height_mm, 1e-6)
        metrics["bbox_projected_width_mm"] = projected_width_mm
        metrics["bbox_projected_height_mm"] = projected_height_mm
        metrics["pca_to_bbox_projection_ratio"] = length / projected_span_mm

        if self.config.shape_backend == "surface_normals":
            metrics.update(
                shape_length_mm=length,
                shape_width_mm=width,
                shape_height_mm=height,
            )
            surface = analyze_surface_normals(
                instance.mask,
                frame.depth_aligned_to_color,
                frame.depth_scale_m,
                frame.color_intrinsics,
                instance.instance_id,
                metrics,
                instance.source_class_name,
            )
            metrics.update(surface.metrics)
            shape_class = surface.shape_class
            shape_reasons = surface.reasons if surface.reject else []
        else:
            shape_class, shape_reasons = self._classify_shape(
                metrics,
                length,
                width,
                height,
                points=points,
            )
        size_reasons = self._size_reasons(length, width, height)
        reasons = list(size_reasons)
        if self.config.reject_by_shape:
            reasons.extend(shape_reasons)
        decision = "reject" if reasons else "accept"
        if valid_ratio < self.config.min_depth_valid_ratio:
            decision = "unknown"
            reasons.append("low_depth_validity")

        return MeasurementResult(
            instance_id=instance.instance_id,
            length_mm=length,
            width_mm=width,
            height_mm=height,
            center_xyz_mm=center,
            decision=decision,
            reasons=list(dict.fromkeys(reasons)),
            shape_class=shape_class,
            metrics=metrics,
        )

    def measure_all(self, frame: RGBDFrame, instances: list[SegmentationInstance]) -> list[MeasurementResult]:
        return [result for item in instances if (result := self.measure(frame, item)) is not None]

    def _size_reasons(self, length: float, width: float, height: float) -> list[str]:
        reasons = []
        for index, value in enumerate((length, width, height), start=1):
            if value < self.config.min_dimension_mm:
                reasons.append(f"dim{index}_below_{self.config.min_dimension_mm:g}mm")
            if value > self.config.max_dimension_mm:
                reasons.append(f"dim{index}_above_{self.config.max_dimension_mm:g}mm")
        return reasons

    def _classify_shape(
        self,
        metrics: dict,
        length: float,
        width: float,
        height: float,
        points: np.ndarray | None = None,
    ):
        circularity = float(metrics.get("circularity") or 0.0)
        solidity = float(metrics.get("solidity") or 0.0)
        extent = float(metrics.get("extent") or 0.0)
        rectangularity = float(metrics.get("rectangularity") or 0.0)
        vertices = int(metrics.get("approx_vertices") or 0)
        elongation = length / width if width > 0 else float("inf")
        flatness = height / length if length > 0 else 0.0
        compactness = height / width if width > 0 else 0.0
        metrics.update(
            {
                "shape_elongation_lw": elongation,
                "shape_flatness_hl": flatness,
                "shape_compactness_hw": compactness,
            }
        )

        if self.config.shape_backend == "strict_3d" and points is not None:
            primitive = fit_primitives(points)
            metrics.update(primitive.metrics)
            metrics["primitive_confidence"] = primitive.confidence
            strict_shape_consistency = (
                solidity >= 0.75
                and circularity >= 0.30
                and 0.08 <= extent <= 1.05
                and flatness >= 0.12
                and compactness >= 0.18
                and float(metrics.get("pca_to_bbox_projection_ratio") or 1.0) <= 2.5
            )
            if primitive.shape_class == "cylindrical_shape" and strict_shape_consistency:
                reasons = ["cylindrical_shape"] if self.config.reject_round else []
                return primitive.shape_class, reasons
            if primitive.shape_class == "box_or_rectangular_shape":
                return primitive.shape_class, []
            supported_cylinder = (
                circularity >= 0.78
                and solidity >= 0.90
                and compactness >= 0.55
                and flatness >= 0.12
                and float(metrics.get("cylinder_radial_inlier_ratio") or 0.0) >= 0.55
                and float(metrics.get("cylinder_radial_residual_p80_norm") or 1.0) <= 0.20
                and float(metrics.get("cylinder_angular_coverage_deg") or 0.0) >= 120.0
                and float(metrics.get("cylinder_slice_radius_cv") or 1.0) <= 0.08
                and int(metrics.get("cylinder_valid_slices") or 0) >= 4
                and 0.12 <= float(metrics.get("cylinder_radius_axis_ratio") or 0.0) <= 1.0
                and float(metrics.get("cylinder_radial_support_ratio_min") or 0.0) >= 0.35
                and float(metrics.get("cylinder_radial_support_ratio_max") or 99.0) <= 1.35
                and float(metrics.get("cylinder_axis_abs_camera_z") or 1.0) <= 0.90
                and float(metrics.get("pca_to_bbox_projection_ratio") or 1.0) <= 2.5
                and float(metrics.get("primitive_score_margin") or 0.0) >= 0.15
            )
            if supported_cylinder:
                metrics["cylinder_decision_tier"] = "3d_plus_silhouette"
                reasons = ["cylindrical_shape"] if self.config.reject_round else []
                return "cylindrical_shape", reasons
            # A strict backend never promotes a cylinder from 2D circularity alone.
            legacy_shape, legacy_reasons = self._classify_shape_legacy(
                metrics, length, width, height
            )
            if legacy_shape in ("cylindrical_shape", "spherical_or_round_shape"):
                return "unknown_shape", []
            return legacy_shape, legacy_reasons
        return self._classify_shape_legacy(metrics, length, width, height)

    def _classify_shape_legacy(
        self,
        metrics: dict,
        length: float,
        width: float,
        height: float,
    ):
        circularity = float(metrics.get("circularity") or 0.0)
        solidity = float(metrics.get("solidity") or 0.0)
        extent = float(metrics.get("extent") or 0.0)
        rectangularity = float(metrics.get("rectangularity") or 0.0)
        vertices = int(metrics.get("approx_vertices") or 0)
        elongation = length / width if width > 0 else float("inf")
        flatness = height / length if length > 0 else 0.0
        compactness = height / width if width > 0 else 0.0

        strong_box = (
            solidity >= 0.90
            and rectangularity >= 0.72
            and extent >= 0.50
            and 4 <= vertices <= 8
            and elongation <= 2.40
            and flatness >= 0.08
            and circularity <= 0.79
        )
        if strong_box:
            return "box_or_rectangular_shape", []
        if solidity < 0.68:
            return "irregular_or_occluded_shape", ["irregular_or_occluded_shape"]
        if flatness < 0.08:
            reasons = ["flat_shape"] if self.config.reject_flat else []
            return "flat_shape", reasons
        if elongation >= 3.0:
            reasons = ["elongated_shape"] if self.config.reject_elongated else []
            return "elongated_shape", reasons

        weak_box = solidity >= 0.86 and rectangularity >= 0.68 and 4 <= vertices <= 10
        if weak_box and circularity <= 0.79:
            return "box_or_rectangular_shape", []
        if circularity >= 0.84 and elongation <= 1.30:
            shape = "spherical_or_round_shape" if compactness >= 0.55 else "cylindrical_shape"
            return shape, [shape] if self.config.reject_round else []
        if circularity >= 0.79 and elongation <= 1.80:
            return "cylindrical_shape", ["cylindrical_shape"] if self.config.reject_round else []
        return "unknown_shape", []

    @staticmethod
    def _unknown(
        instance: SegmentationInstance,
        metrics: dict,
        reason: str,
        center=(None, None, None),
    ) -> MeasurementResult:
        return MeasurementResult(
            instance_id=instance.instance_id,
            length_mm=None,
            width_mm=None,
            height_mm=None,
            center_xyz_mm=center,
            decision="unknown",
            reasons=[reason],
            shape_class="unknown_shape",
            metrics=metrics,
        )
