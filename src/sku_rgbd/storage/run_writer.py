from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from sku_rgbd.models import MeasurementResult, RGBDFrame, SegmentationResult


class RunWriter:
    def __init__(self, root: str | Path):
        self.run_dir = Path(root).expanduser() / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        for subdir in (
            "camera/rgb",
            "camera/depth_raw",
            "camera/depth_aligned_to_color",
            "camera/meta",
            "sam3/overlay",
            "sam3/mask_index",
            "sam3/meta",
            "rtdm/overlay",
            "rtdm/mask_index",
            "rtdm/meta",
            "grid",
        ):
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save(
        self,
        frame: RGBDFrame,
        grid,
        backend_data: dict[str, tuple[SegmentationResult | None, list[MeasurementResult], object | None]],
    ) -> str:
        sample_id = f"{frame.frame_id:06d}_{datetime.now():%Y%m%d_%H%M%S}"
        cv2.imwrite(str(self.run_dir / "camera/rgb" / f"{sample_id}.png"), frame.color_bgr)
        Image.fromarray(frame.depth_raw, mode="I;16").save(
            self.run_dir / "camera/depth_raw" / f"{sample_id}.png"
        )
        Image.fromarray(frame.depth_aligned_to_color, mode="I;16").save(
            self.run_dir / "camera/depth_aligned_to_color" / f"{sample_id}.png"
        )
        camera_meta = {
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "depth_scale_m": frame.depth_scale_m,
            "color_intrinsics": frame.color_intrinsics.as_dict(),
            "depth_intrinsics": frame.depth_intrinsics.as_dict(),
            "metadata": frame.metadata,
        }
        (self.run_dir / "camera/meta" / f"{sample_id}.json").write_text(
            json.dumps(camera_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if grid is not None:
            cv2.imwrite(str(self.run_dir / "grid" / f"{sample_id}.png"), grid)

        for backend, (segmentation, measurements, overlay) in backend_data.items():
            if segmentation is None:
                continue
            backend_dir = self.run_dir / backend
            if overlay is not None:
                cv2.imwrite(str(backend_dir / "overlay" / f"{sample_id}.png"), overlay)
            mask_index = frame.depth_raw.astype("uint16") * 0
            if mask_index.shape != frame.color_bgr.shape[:2]:
                import numpy as np

                mask_index = np.zeros(frame.color_bgr.shape[:2], dtype=np.uint16)
            for instance in segmentation.instances:
                mask_index[instance.mask] = int(instance.instance_id)
            cv2.imwrite(str(backend_dir / "mask_index" / f"{sample_id}.png"), mask_index)
            meta = {
                "backend": backend,
                "frame_id": segmentation.frame_id,
                "infer_ms": segmentation.infer_ms,
                "roundtrip_ms": segmentation.roundtrip_ms,
                "instances": [
                    {
                        "instance_id": item.instance_id,
                        "score": item.score,
                        "bbox": list(item.bbox),
                        "area": item.area,
                        "source_class_name": item.source_class_name,
                    }
                    for item in segmentation.instances
                ],
                "measurements": [item.as_dict() for item in measurements],
            }
            (backend_dir / "meta" / f"{sample_id}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return sample_id

