#!/usr/bin/env python3
"""Convert saved RGB + mask-index runs into a reproducible COCO dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="sam3")
    parser.add_argument("--selection", choices=("all", "center"), default="all")
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--max-area-ratio", type=float, default=0.50)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def rle_counts(mask: np.ndarray) -> list[int]:
    pixels = mask.astype(np.uint8).T.reshape(-1)
    counts: list[int] = []
    previous = 0
    length = 0
    for value in pixels:
        value = int(value)
        if value == previous:
            length += 1
        else:
            counts.append(length)
            length = 1
            previous = value
    counts.append(length)
    return counts


def selected_ids(mask_index: np.ndarray, mode: str) -> list[int]:
    ids = [int(value) for value in np.unique(mask_index) if value]
    if mode == "all" or len(ids) < 2:
        return ids
    h, w = mask_index.shape
    center = np.array([w / 2.0, h / 2.0])
    candidates = []
    for instance_id in ids:
        ys, xs = np.where(mask_index == instance_id)
        centroid = np.array([xs.mean(), ys.mean()])
        distance = float(np.linalg.norm((centroid - center) / [w, h]))
        candidates.append((distance, -len(xs), instance_id))
    best_distance = min(item[0] for item in candidates)
    near = [item for item in candidates if item[0] <= best_distance + 0.03]
    return [min(near, key=lambda item: item[1])[2]]


def discover(run: Path, backend: str) -> list[tuple[Path, Path]]:
    rgb_dir = run / "camera" / "rgb"
    mask_dir = run / backend / "mask_index"
    if not rgb_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected camera/rgb and {backend}/mask_index in {run}")
    rgb_by_stem = {path.stem: path for path in rgb_dir.glob("*.png")}
    return [(rgb_by_stem[path.stem], path) for path in sorted(mask_dir.glob("*.png")) if path.stem in rgb_by_stem]


def split_runs(items: list[tuple[Path, Path, str]], args: argparse.Namespace) -> dict[str, list]:
    by_run: dict[str, list] = defaultdict(list)
    for item in items:
        by_run[item[2]].append(item)
    groups = list(by_run.values())
    random.Random(args.seed).shuffle(groups)
    flattened = [item for group in groups for item in group]
    val_count = round(len(flattened) * args.val_ratio)
    test_count = round(len(flattened) * args.test_ratio)
    # Runs stay intact when enough independent runs exist; otherwise split frames deterministically.
    if len(groups) >= 5:
        targets = {"train": [], "val": [], "test": []}
        for group in sorted(groups, key=len, reverse=True):
            target = min(targets, key=lambda key: len(targets[key]) / max(1, {"train": len(flattened)-val_count-test_count, "val": val_count, "test": test_count}[key]))
            targets[target].extend(group)
        return targets
    rng = random.Random(args.seed)
    rng.shuffle(flattened)
    return {
        "test": flattened[:test_count],
        "val": flattened[test_count:test_count + val_count],
        "train": flattened[test_count + val_count:],
    }


def write_split(name: str, items: list, args: argparse.Namespace) -> dict:
    image_dir = args.output / name
    image_dir.mkdir(parents=True, exist_ok=True)
    coco = {"info": {"description": "SAM3 teacher masks for SKU segmentation"}, "images": [], "annotations": [], "categories": [{"id": 1, "name": "sku"}]}
    annotation_id = 1
    rejected = 0
    for image_id, (rgb_path, mask_path, run_name) in enumerate(items, start=1):
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask_index = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask_index is None:
            rejected += 1
            continue
        if mask_index.ndim == 3:
            mask_index = mask_index[:, :, 0]
        if mask_index.shape != image.shape[:2]:
            rejected += 1
            continue
        filename = f"{run_name}__{rgb_path.name}"
        shutil.copy2(rgb_path, image_dir / filename)
        h, w = image.shape[:2]
        coco["images"].append({"id": image_id, "file_name": filename, "width": w, "height": h, "run": run_name})
        for instance_id in selected_ids(mask_index, args.selection):
            mask = mask_index == instance_id
            area = int(mask.sum())
            if area < args.min_area or area / (h * w) > args.max_area_ratio:
                continue
            ys, xs = np.where(mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            coco["annotations"].append({
                "id": annotation_id, "image_id": image_id, "category_id": 1,
                "segmentation": {"size": [h, w], "counts": rle_counts(mask)},
                "area": area, "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1], "iscrowd": 0,
            })
            annotation_id += 1
    (args.output / f"annotations_{name}.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    return {"images": len(coco["images"]), "annotations": len(coco["annotations"]), "rejected": rejected}


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = [(rgb, mask, run.name) for run in args.runs for rgb, mask in discover(run, args.backend)]
    stats = {name: write_split(name, split, args) for name, split in split_runs(items, args).items()}
    manifest = {"seed": args.seed, "backend": args.backend, "selection": args.selection, "runs": [str(path.resolve()) for path in args.runs], "splits": stats}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
