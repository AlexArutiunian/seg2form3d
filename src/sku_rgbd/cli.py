from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from sku_rgbd.camera import RealSenseSource, RobotCameraSource
from sku_rgbd.measurement import MeasurementConfig, MeasurementEngine
from sku_rgbd.pipeline import BackendWorker
from sku_rgbd.segmentation import RemoteSegmentationClient
from sku_rgbd.storage import RunWriter
from sku_rgbd.ui import build_grid


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Modular RGB-D remote segmentation and SKU measurement.")
    root.add_argument("--camera", choices=("realsense", "robot"), required=True)
    root.add_argument("--sam3-server", default="tcp://127.0.0.1:5558")
    root.add_argument("--rtdm-server", default="tcp://127.0.0.1:5557")
    root.add_argument("--sam3-fps", type=float, default=1.0)
    root.add_argument("--rtdm-fps", type=float, default=10.0)
    root.add_argument("--jpg-quality", type=int, default=85)
    root.add_argument("--max-mask-area-ratio", type=float, default=1 / 3)
    root.add_argument("--max-bbox-area-ratio", type=float, default=1 / 3)
    root.add_argument("--measurement-config", default=None)
    root.add_argument("--min-dimension-mm", type=float, default=None)
    root.add_argument("--max-dimension-mm", type=float, default=None)
    root.add_argument("--max-center-z-mm", type=float, default=None)
    root.add_argument("--min-depth-points", type=int, default=None)
    root.add_argument("--no-shape-reject", action="store_true")
    root.add_argument(
        "--shape-backend",
        choices=("strict_3d", "legacy_2d", "surface_normals"),
        default=None,
    )
    root.add_argument("--save-mode", choices=("manual", "continuous", "off"), default="manual")
    root.add_argument("--save-every-n", type=int, default=15)
    root.add_argument("--save-root", default="data")
    root.add_argument("--width", type=int, default=1280)
    root.add_argument("--height", type=int, default=720)
    root.add_argument("--camera-fps", type=int, default=15)
    root.add_argument("--serial", default=None)
    root.add_argument("--robot-ip", default=None)
    root.add_argument("--robot-port", type=int, default=8088)
    root.add_argument("--robot-camera-name", default="Camera-000")
    root.add_argument(
        "--robot-api-path",
        default=str(Path(__file__).resolve().parents[2] / "third_party" / "seer_camera"),
    )
    return root


def make_camera(args):
    if args.camera == "realsense":
        return RealSenseSource(args.width, args.height, args.camera_fps, args.serial)
    if not args.robot_ip:
        raise ValueError("--robot-ip is required for the robot camera")
    return RobotCameraSource(
        robot_ip=args.robot_ip,
        port=args.robot_port,
        camera_name=args.robot_camera_name,
        api_path=args.robot_api_path,
    )


def make_worker(name: str, endpoint: str, fps: float, args, measurement):
    client = RemoteSegmentationClient(
        backend=name,
        endpoint=endpoint,
        jpg_quality=args.jpg_quality,
        max_mask_area_ratio=args.max_mask_area_ratio,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
    )
    return BackendWorker(client, measurement, fps)


def public_run_config(args) -> dict:
    values = vars(args).copy()
    for key in ("robot_ip", "robot_api_path"):
        if values.get(key):
            values[key] = "<redacted>"
    return values


def main() -> None:
    args = parser().parse_args()
    config_values = {}
    if args.measurement_config:
        config_values = json.loads(Path(args.measurement_config).expanduser().read_text(encoding="utf-8"))
    measurement_config = MeasurementConfig(**config_values)
    if args.min_dimension_mm is not None:
        measurement_config.min_dimension_mm = args.min_dimension_mm
    if args.max_dimension_mm is not None:
        measurement_config.max_dimension_mm = args.max_dimension_mm
    if args.max_center_z_mm is not None:
        measurement_config.max_center_z_mm = args.max_center_z_mm
    if args.min_depth_points is not None:
        measurement_config.min_depth_points = args.min_depth_points
    if args.no_shape_reject:
        measurement_config.reject_by_shape = False
    if args.shape_backend is not None:
        measurement_config.shape_backend = args.shape_backend
    measurement = MeasurementEngine(measurement_config)
    camera = make_camera(args)
    writer = RunWriter(args.save_root)
    (writer.run_dir / "run_config.json").write_text(
        json.dumps(public_run_config(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sam3 = make_worker("sam3", args.sam3_server, args.sam3_fps, args, measurement)
    rtdm = make_worker("rtdm", args.rtdm_server, args.rtdm_fps, args, measurement)

    camera.start()
    sam3.start()
    rtdm.start()
    cv2.namedWindow("SKU RGB-D Pipeline", cv2.WINDOW_NORMAL)
    saved = 0
    print(f"[OK] run_dir={writer.run_dir}")
    print("[INFO] s=save, q/ESC=quit")
    try:
        while True:
            frame = camera.read()
            sam3.submit(frame)
            rtdm.submit(frame)
            sam3_state = sam3.snapshot()
            rtdm_state = rtdm.snapshot()
            status = f"camera={args.camera} save={args.save_mode} saved={saved} | s=save q=quit"
            grid = build_grid(
                frame,
                sam3_state.overlay,
                sam3_state.status,
                rtdm_state.overlay,
                rtdm_state.status,
                status,
            )
            cv2.imshow("SKU RGB-D Pipeline", grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            do_save = args.save_mode == "continuous" and frame.frame_id % max(1, args.save_every_n) == 0
            do_save = do_save or (args.save_mode == "manual" and key == ord("s"))
            if do_save and args.save_mode != "off":
                sample_id = writer.save(
                    frame,
                    grid,
                    {
                        "sam3": (sam3_state.segmentation, sam3_state.measurements or [], sam3_state.overlay),
                        "rtdm": (rtdm_state.segmentation, rtdm_state.measurements or [], rtdm_state.overlay),
                    },
                )
                saved += 1
                print(f"[SAVE] {sample_id}")
    finally:
        sam3.stop()
        rtdm.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
