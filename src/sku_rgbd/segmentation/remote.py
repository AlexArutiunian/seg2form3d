from __future__ import annotations

import socket
import time

import cv2
import numpy as np

from sku_rgbd.models import SegmentationInstance, SegmentationResult
from sku_rgbd.segmentation.protocol import decode_message, encode_message, recv_message, send_message


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    endpoint = endpoint.removeprefix("tcp://")
    host, port = endpoint.rsplit(":", 1)
    return host, int(port)


class RemoteSegmentationClient:
    def __init__(
        self,
        backend: str,
        endpoint: str,
        jpg_quality: int = 85,
        max_mask_area_ratio: float = 1 / 3,
        max_bbox_area_ratio: float = 1 / 3,
        timeout: float = 120.0,
    ):
        self.backend = backend
        self.endpoint = endpoint
        self.jpg_quality = jpg_quality
        self.max_mask_area_ratio = max_mask_area_ratio
        self.max_bbox_area_ratio = max_bbox_area_ratio
        self.timeout = timeout
        self.conn: socket.socket | None = None

    def connect(self) -> dict:
        host, port = parse_endpoint(self.endpoint)
        self.conn = socket.create_connection((host, port), timeout=self.timeout)
        self.conn.settimeout(self.timeout)
        send_message(self.conn, encode_message({"cmd": "ping"}))
        response = decode_message(recv_message(self.conn))
        if not response.get("ok"):
            raise RuntimeError(f"Bad ping response from {self.endpoint}: {response}")
        return response

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def segment(self, frame_id: int, frame_bgr: np.ndarray) -> SegmentationResult:
        if self.conn is None:
            self.connect()
        ok, jpg = cv2.imencode(
            ".jpg",
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpg_quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode segmentation request JPEG")

        request = {
            "cmd": "segment",
            "request_id": f"{self.backend}_{frame_id}",
            "jpg": jpg.tobytes(),
            "height": int(frame_bgr.shape[0]),
            "width": int(frame_bgr.shape[1]),
        }
        started = time.perf_counter()
        try:
            send_message(self.conn, encode_message(request))
            response = decode_message(recv_message(self.conn))
        except Exception:
            self.close()
            raise
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown segmentation server error"))

        mask_png = np.frombuffer(response["mask_index_png"], dtype=np.uint8)
        mask_index = cv2.imdecode(mask_png, cv2.IMREAD_UNCHANGED)
        if mask_index is None:
            raise RuntimeError("Failed to decode remote mask index")
        if mask_index.ndim == 3:
            mask_index = mask_index[:, :, 0]
        if mask_index.shape != frame_bgr.shape[:2]:
            mask_index = cv2.resize(
                mask_index,
                (frame_bgr.shape[1], frame_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        instances = []
        for item in response.get("instances", []):
            mask = mask_index == int(item["id"])
            area = int(mask.sum())
            if not area:
                continue
            ys, xs = np.where(mask)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if self.max_mask_area_ratio > 0 and area / frame_area > self.max_mask_area_ratio:
                continue
            if self.max_bbox_area_ratio > 0 and bbox_area / frame_area > self.max_bbox_area_ratio:
                continue
            instances.append(
                SegmentationInstance(
                    instance_id=len(instances) + 1,
                    mask=mask,
                    bbox=bbox,
                    score=float(item.get("score", 0.0)),
                    source_class_name=str(item.get("source_class_name", "")),
                )
            )

        return SegmentationResult(
            backend=self.backend,
            frame_id=frame_id,
            instances=instances,
            infer_ms=float(response.get("infer_ms", 0.0)),
            roundtrip_ms=roundtrip_ms,
            raw_metadata={"server_backend": response.get("backend", "")},
        )

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

