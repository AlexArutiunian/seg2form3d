from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sku_rgbd.measurement import MeasurementConfig, MeasurementEngine
from sku_rgbd.models import CameraIntrinsics, RGBDFrame, SegmentationInstance, SegmentationResult
from sku_rgbd.ui import draw_measurements


def intrinsics_from_meta(value: dict) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=int(value["width"]),
        height=int(value["height"]),
        fx=float(value["fx"]),
        fy=float(value["fy"]),
        cx=float(value.get("cx", value.get("ppx"))),
        cy=float(value.get("cy", value.get("ppy"))),
        model=str(value.get("model", "")),
        coeffs=[float(item) for item in value.get("coeffs", [])],
    )


def find_depth_dir(run_dir: Path) -> Path:
    for relative in ("camera/depth_aligned_to_color", "camera/depth_aligned_to_rgb"):
        candidate = run_dir / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("No aligned depth folder found")


def load_instances(mask_path: Path, backend_meta: dict) -> list[SegmentationInstance]:
    mask_index = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask_index is None:
        raise RuntimeError(f"Cannot read {mask_path}")
    metadata = {
        int(item.get("instance_id", item.get("track_id", index))): item
        for index, item in enumerate(backend_meta.get("instances", []), start=1)
    }
    instances = []
    for mask_id in (int(value) for value in np.unique(mask_index) if value):
        mask = mask_index == mask_id
        ys, xs = np.where(mask)
        item = metadata.get(mask_id, {})
        instances.append(
            SegmentationInstance(
                instance_id=mask_id,
                mask=mask,
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                score=float(item.get("score", 0.0)),
                source_class_name=str(item.get("source_class_name", "")),
            )
        )
    return instances


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute measurements from saved RGB-D and masks, without GPU.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend", choices=("sam3", "rtdm"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-dimension-mm", type=float, default=10.0)
    parser.add_argument("--max-dimension-mm", type=float, default=450.0)
    parser.add_argument("--max-center-z-mm", type=float, default=1500.0)
    parser.add_argument("--no-shape-reject", action="store_true")
    parser.add_argument(
        "--shape-backend",
        choices=("strict_3d", "legacy_2d", "surface_normals"),
        default="strict_3d",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    overlay_dir = out_dir / "overlay"
    meta_out_dir = out_dir / "meta"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = find_depth_dir(run_dir)
    engine = MeasurementEngine(
        MeasurementConfig(
            min_dimension_mm=args.min_dimension_mm,
            max_dimension_mm=args.max_dimension_mm,
            max_center_z_mm=args.max_center_z_mm,
            reject_by_shape=not args.no_shape_reject,
            shape_backend=args.shape_backend,
        )
    )

    mask_dir = run_dir / args.backend / "mask_index"
    mask_files = sorted(mask_dir.glob("*.png"))
    if args.limit:
        mask_files = mask_files[: args.limit]
    started = time.perf_counter()
    for index, mask_path in enumerate(mask_files, start=1):
        sample_id = mask_path.stem
        rgb_path = run_dir / "camera/rgb" / f"{sample_id}.png"
        depth_path = depth_dir / f"{sample_id}.png"
        camera_meta_path = run_dir / "camera/meta" / f"{sample_id}.json"
        backend_meta_path = run_dir / args.backend / "meta" / f"{sample_id}.json"
        if not all(path.exists() for path in (rgb_path, depth_path, camera_meta_path, backend_meta_path)):
            continue
        color = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        camera_meta = json.loads(camera_meta_path.read_text(encoding="utf-8"))
        backend_meta = json.loads(backend_meta_path.read_text(encoding="utf-8"))
        color_intr = intrinsics_from_meta(camera_meta["color_intrinsics"])
        depth_intr = intrinsics_from_meta(
            camera_meta.get("depth_intrinsics")
            or camera_meta.get("raw_depth_intrinsics")
            or camera_meta["aligned_depth_intrinsics"]
        )
        frame = RGBDFrame(
            frame_id=int(camera_meta.get("frame_id", index)),
            timestamp=float(camera_meta.get("timestamp", 0.0)),
            color_bgr=color,
            depth_raw=depth,
            depth_aligned_to_color=depth,
            depth_scale_m=float(camera_meta.get("depth_scale_m", camera_meta.get("depth_scale_m_per_unit", 0.001))),
            color_intrinsics=color_intr,
            depth_intrinsics=depth_intr,
            metadata=camera_meta.get("metadata", {}),
        )
        instances = load_instances(mask_path, backend_meta)
        measurements = engine.measure_all(frame, instances)
        segmentation = SegmentationResult(
            backend=args.backend,
            frame_id=frame.frame_id,
            instances=instances,
            infer_ms=float(backend_meta.get("infer_ms", 0.0)),
            roundtrip_ms=float(backend_meta.get("roundtrip_ms", 0.0)),
        )
        overlay = draw_measurements(color, segmentation, measurements)
        cv2.imwrite(str(overlay_dir / f"{sample_id}.png"), overlay)
        (meta_out_dir / f"{sample_id}.json").write_text(
            json.dumps(
                {
                    "source_run": str(run_dir),
                    "backend": args.backend,
                    "sample_id": sample_id,
                    "measurements": [item.as_dict() for item in measurements],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        elapsed = max(0.001, time.perf_counter() - started)
        rate = index / elapsed
        eta = (len(mask_files) - index) / rate if rate > 0 else 0.0
        print(
            f"\r[{index}/{len(mask_files)}] {sample_id} | "
            f"{rate:.2f} frame/s | ETA {eta:.0f}s",
            end="",
            flush=True,
        )
    print(f"\n[OK] output={out_dir}")
