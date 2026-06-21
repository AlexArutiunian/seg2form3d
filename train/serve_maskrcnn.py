#!/usr/bin/env python3
"""Loopback-only inference server compatible with RemoteSegmentationClient."""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

from sku_rgbd.segmentation.protocol import decode_message, encode_message, recv_message, send_message
from train_maskrcnn import model_for


def response_for(model, device, request, score_threshold):
    image = cv2.imdecode(np.frombuffer(request["jpg"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("invalid JPEG")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model([to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device)])[0]
    infer_ms = (time.perf_counter() - started) * 1000.0
    mask_index = np.zeros(image.shape[:2], np.uint16)
    instances = []
    next_id = 1
    order = torch.argsort(output["scores"], descending=True)
    for index in order.tolist():
        score = float(output["scores"][index])
        if score < score_threshold or next_id >= np.iinfo(np.uint16).max:
            continue
        mask = output["masks"][index, 0].detach().cpu().numpy() >= 0.5
        mask &= mask_index == 0
        if int(mask.sum()) < 20:
            continue
        mask_index[mask] = next_id
        instances.append({"id": next_id, "score": score, "source_class_name": "sku"})
        next_id += 1
    ok, encoded = cv2.imencode(".png", mask_index)
    if not ok:
        raise RuntimeError("mask PNG encoding failed")
    return {"ok": True, "backend": "maskrcnn_sam3_teacher", "infer_ms": infer_ms, "mask_index_png": encoded.tobytes(), "instances": instances}


def serve_connection(conn, model, device, threshold):
    with conn:
        while True:
            try:
                request = decode_message(recv_message(conn))
            except EOFError:
                return
            try:
                if request.get("cmd") == "ping":
                    response = {"ok": True, "backend": "maskrcnn_sam3_teacher"}
                elif request.get("cmd") == "segment":
                    response = response_for(model, device, request, threshold)
                else:
                    response = {"ok": False, "error": "unsupported command"}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            send_message(conn, encode_message(response))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score", type=float, default=0.35)
    args = parser.parse_args()
    if args.bind not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("Refusing a public bind. Use an authenticated reverse proxy if remote access is required.")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_for(2, pretrained=False).to(device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port)); server.listen(4)
        print(f"listening on {args.bind}:{args.port} device={device}", flush=True)
        while True:
            conn, _ = server.accept()
            serve_connection(conn, model, device, args.score)


if __name__ == "__main__":
    main()
