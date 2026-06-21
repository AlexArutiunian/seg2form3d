#!/usr/bin/env python3
"""Group near-duplicate masked crops using visual and silhouette fingerprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def fingerprint(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], crop_mask, [16, 8], [0, 180, 0, 256]).reshape(-1)
    histogram /= max(float(histogram.sum()), 1.0)
    silhouette = cv2.resize(crop_mask, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)
    silhouette /= max(float(np.linalg.norm(silhouette)), 1e-6)
    return np.concatenate((histogram * 2.0, silhouette))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.24)
    args = parser.parse_args()
    coco = json.loads(args.annotations.read_text())
    images = {item["id"]: item for item in coco["images"]}
    groups: list[dict] = []
    for annotation in coco["annotations"]:
        image_info = images[annotation["image_id"]]
        image = cv2.imread(str(args.images / image_info["file_name"]))
        rle = annotation["segmentation"]
        rle["counts"] = rle["counts"] if isinstance(rle["counts"], list) else rle["counts"].encode()
        mask = mask_utils.decode(rle).astype(bool)
        vector = fingerprint(image, mask)
        best = None
        for group in groups:
            distance = float(np.linalg.norm(vector - group["centroid"]))
            if best is None or distance < best[0]:
                best = (distance, group)
        if best is None or best[0] > args.distance:
            groups.append({"centroid": vector, "members": [annotation["id"]]})
        else:
            group = best[1]
            n = len(group["members"])
            group["centroid"] = (group["centroid"] * n + vector) / (n + 1)
            group["members"].append(annotation["id"])
    result = {"method": "HSV histogram + normalized silhouette", "distance_threshold": args.distance, "unique_groups": len(groups), "total_masks": len(coco["annotations"]), "groups": [group["members"] for group in groups]}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"unique={len(groups)} total={len(coco['annotations'])}")


if __name__ == "__main__":
    main()
