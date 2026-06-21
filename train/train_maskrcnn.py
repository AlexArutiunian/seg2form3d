#!/usr/bin/env python3
"""Fine-tune torchvision Mask R-CNN on COCO teacher masks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor


class CocoMasks(Dataset):
    def __init__(self, images: Path, annotations: Path, train: bool):
        self.images, self.coco, self.train = images, COCO(str(annotations)), train
        self.ids = sorted(self.coco.imgs)

    def __len__(self): return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image = pil_to_tensor(Image.open(self.images / info["file_name"]).convert("RGB")).float() / 255.0
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=False))
        boxes, masks = [], []
        for item in annotations:
            x, y, w, h = item["bbox"]
            boxes.append([x, y, x + w, y + h])
            masks.append(self.coco.annToMask(item))
        mask_tensor = torch.as_tensor(np.asarray(masks), dtype=torch.uint8)
        if not masks:
            mask_tensor = torch.zeros((0, image.shape[-2], image.shape[-1]), dtype=torch.uint8)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "masks": mask_tensor,
            "image_id": torch.tensor(image_id),
        }
        if self.train and torch.rand(()) < 0.5:
            image = image.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            if len(target["boxes"]):
                width = image.shape[-1]
                old = target["boxes"].clone()
                target["boxes"][:, 0], target["boxes"][:, 2] = width - old[:, 2], width - old[:, 0]
        return image, target


def model_for(num_classes: int, pretrained: bool):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_channels = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 256, num_classes)
    return model


def evaluate_loss(model, loader, device):
    model.train()
    values = []
    with torch.no_grad():
        for images, targets in loader:
            loss = sum(model([x.to(device) for x in images], [{k: v.to(device) for k, v in y.items()} for y in targets]).values())
            values.append(float(loss))
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260618)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train = CocoMasks(args.dataset / "train", args.dataset / "annotations_train.json", True)
    val = CocoMasks(args.dataset / "val", args.dataset / "annotations_val.json", False)
    collate = lambda batch: tuple(zip(*batch))
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val, 1, shuffle=False, num_workers=args.workers, collate_fn=collate)
    model = model_for(2, pretrained=args.resume is None).to(device)
    start_epoch = 1
    state = None
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        start_epoch = int(state.get("epoch", 0)) + 1 if isinstance(state, dict) else 1
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    if isinstance(state, dict) and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if isinstance(state, dict) and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    args.output.mkdir(parents=True, exist_ok=True)
    history = list(state.get("history", [])) if isinstance(state, dict) else []
    best_val = min((item["val_loss"] for item in history), default=float("inf"))
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for images, targets in train_loader:
            loss = sum(model([x.to(device) for x in images], [{k: v.to(device) for k, v in y.items()} for y in targets]).values())
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, device)
        elapsed = time.time() - started
        eta = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch)
        record = {"epoch": epoch, "train_loss": sum(losses) / max(1, len(losses)), "val_loss": val_loss, "eta_seconds": eta}
        history.append(record); print(json.dumps(record), flush=True)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "args": vars(args), "history": history}
        torch.save(checkpoint, args.output / f"maskrcnn_epoch_{epoch:03d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, args.output / "maskrcnn_best.pt")
        (args.output / "history.json").write_text(json.dumps(history, indent=2, default=str))


if __name__ == "__main__":
    main()
