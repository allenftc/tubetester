from __future__ import annotations

import argparse
from pathlib import Path

from controller.config.settings import load_settings
from controller.vision.qr_decoder import QrDecoder
from controller.web.server import serve_control_server
from controller.workflow.state_machine import TubeScanWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test tube scanner controller")
    default_config_dir = Path(__file__).resolve().parents[1] / "calibration"
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir,
        help="Directory containing calibration JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned workflow and exit.",
    )
    parser.add_argument(
        "--serve-web",
        action="store_true",
        help="Start the built-in web control surface.",
    )
    parser.add_argument(
        "--scan-camera",
        action="store_true",
        help="Open the configured camera and decode any QR code visible in the stream.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open a lightweight live camera preview that highlights QR codes until interrupted.",
    )
    parser.add_argument(
        "--decode-once",
        action="store_true",
        help="Capture one frame, attempt a QR decode, and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings(args.config_dir)
    workflow = TubeScanWorkflow(settings)

    if args.dry_run:
        for line in workflow.describe():
            print(line)
        return 0

    if args.serve_web:
        serve_control_server(settings)
        return 0

    if args.scan_camera:
        decoder = QrDecoder(library=settings.camera.qr_library)
        result = decoder.scan_camera(
            camera_index=settings.camera.device_index,
            width=settings.camera.resolution.width,
            height=settings.camera.resolution.height,
            fps=120,
        )
        if result.payload is None:
            print("No QR code detected.")
        else:
            print(f"Decoded QR payload: {result.payload}")
        return 0

    if args.preview:
        decoder = QrDecoder(library=settings.camera.qr_library)
        decoder.preview_stream(
            camera_index=settings.camera.device_index,
            width=640,
            height=480,
            fps=30,
        )
        return 0

    if args.decode_once:
        decoder = QrDecoder(library=settings.camera.qr_library)
        capture = cv2.VideoCapture(settings.camera.device_index)
        if not capture.isOpened():
            print("Unable to open camera.")
            return 1

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640.0)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)
        capture.set(cv2.CAP_PROP_FPS, 30.0)
        grabbed, frame = capture.read()
        capture.release()
        if not grabbed or frame is None:
            print("Unable to read camera frame.")
            return 1

        result = decoder.decode_from_frame(frame)
        if result.payload is None:
            print("No QR code detected.")
        else:
            print(f"Decoded QR payload: {result.payload}")
        return 0

    print("Controller scaffold is loaded.")
    print("Use --dry-run to inspect the planned scan sequence.")
    print("Use --serve-web to start the browser control surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())