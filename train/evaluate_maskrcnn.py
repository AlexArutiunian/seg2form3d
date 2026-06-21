#!/usr/bin/env python3
"""Evaluate a trained checkpoint with standard COCO bbox and mask AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from train_maskrcnn import CocoMasks, model_for


def encode_binary_mask(mask):
    from pycocotools import mask as mask_utils
    import numpy as np

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset = CocoMasks(args.dataset / args.split, args.dataset / f"annotations_{args.split}.json", False)
    loader = DataLoader(dataset, 1, shuffle=False, collate_fn=lambda batch: tuple(zip(*batch)))
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_for(2, pretrained=False).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            output = model([images[0].to(device)])[0]
            image_id = int(targets[0]["image_id"])
            for box, mask, score in zip(output["boxes"], output["masks"], output["scores"]):
                score = float(score)
                if score < args.score:
                    continue
                x0, y0, x1, y1 = box.cpu().tolist()
                predictions.append({"image_id": image_id, "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": encode_binary_mask(mask[0].cpu().numpy() >= 0.5), "score": score})
    results = {}
    if predictions:
        detections = dataset.coco.loadRes(predictions)
        for kind in ("bbox", "segm"):
            evaluator = COCOeval(dataset.coco, detections, kind)
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            results[kind] = {"AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]), "AP75": float(evaluator.stats[2]), "AR100": float(evaluator.stats[8])}
    results["predictions"] = len(predictions)
    output = args.output or args.checkpoint.with_suffix(f".{args.split}.metrics.json")
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
