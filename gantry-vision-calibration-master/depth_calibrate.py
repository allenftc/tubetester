#!/usr/bin/env python3
"""Calibrate depth from horizontal camera motion and a tracked circle.

Each sample is two images of the same stationary circle:
    python depth_calibrate.py ^
        --sample left_300.jpg right_300.jpg 300 10 ^
        --sample left_500.jpg right_500.jpg 500 10 ^
        --sample left_700.jpg right_700.jpg 700 10 ^
        --camera-calibration camera_calibration.npz ^
        --output depth_calibration.json

For height samples relative to a reference plane:
    python depth_calibrate.py ^
        --reference-depth 500 ^
        --height-sample left_h0.jpg right_h0.jpg 0 10 ^
        --height-sample left_h50.jpg right_h50.jpg 50 10 ^
        --height-sample left_h100.jpg right_h100.jpg 100 10

For several heights visible in one image pair:
    python depth_calibrate.py ^
        --reference-depth 500 ^
        --multi-height-sample left.jpg right.jpg 0,50,100,150 10

The last two values are known object depth and camera movement, in the same
units. Click the circle center in the left image, then in the right image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate depth from camera translation.")
    parser.add_argument(
        "--sample",
        nargs=4,
        action="append",
        metavar=("LEFT", "RIGHT", "DEPTH", "BASELINE"),
        help="Image pair followed by known object depth and camera movement.",
    )
    parser.add_argument(
        "--height-sample",
        nargs=4,
        action="append",
        metavar=("LEFT", "RIGHT", "HEIGHT", "BASELINE"),
        help=(
            "Image pair followed by object height above the reference plane "
            "and camera movement."
        ),
    )
    parser.add_argument(
        "--multi-height-sample",
        nargs=4,
        action="append",
        metavar=("LEFT", "RIGHT", "HEIGHTS", "BASELINE"),
        help=(
            "One image pair containing several objects. HEIGHTS is a comma-separated "
            "list in click order, such as 0,50,100,150."
        ),
    )
    parser.add_argument(
        "--reference-depth",
        type=float,
        help=(
            "Camera-to-reference-plane distance. Required with --height-sample; "
            "depth is calculated as reference-depth minus height."
        ),
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=Path("camera_calibration.npz"),
        help="OpenCV calibration file containing camera_matrix, distortion_coefficients, image_size.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("depth_calibration.json"),
        help="Output depth calibration JSON.",
    )
    return parser.parse_args()


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    try:
        with np.load(path) as data:
            camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(data["distortion_coefficients"], dtype=np.float64)
            image_size = tuple(map(int, data["image_size"]))
    except (OSError, KeyError, ValueError) as exc:
        raise RuntimeError(f"Could not load camera calibration {path}: {exc}") from exc
    if camera_matrix.shape != (3, 3) or distortion.size == 0:
        raise ValueError("Camera calibration must contain a 3x3 matrix and distortion coefficients.")
    return camera_matrix, distortion, image_size


def click_center(image: np.ndarray, title: str) -> tuple[float, float]:
    point: list[tuple[float, float]] = []
    display = image.copy()

    def redraw() -> None:
        nonlocal display
        display = image.copy()
        if point:
            x, y = map(int, np.round(point[0]))
            cv2.drawMarker(display, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.circle(display, (x, y), 8, (0, 255, 255), 2)
        cv2.putText(
            display,
            "Click circle center | Enter=accept | R=reset | Esc=cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Click circle center | Enter=accept | R=reset | Esc=cancel",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def callback(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [(float(x), float(y))]
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, callback)
    redraw()
    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and point:
            cv2.destroyWindow(title)
            return point[0]
        if key in (ord("r"), ord("R")):
            point.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("Point selection cancelled.")


def click_centers(
    image: np.ndarray, title: str, count: int
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    display = image.copy()

    def redraw() -> None:
        nonlocal display
        display = image.copy()
        for index, (x, y) in enumerate(points):
            center = (int(round(x)), int(round(y)))
            cv2.drawMarker(display, center, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(
                display, str(index), (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                display, str(index), (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
        text = f"Click object {len(points) + 1}/{count} in matching order | R=reset | Esc=cancel"
        cv2.putText(display, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    def callback(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((float(x), float(y)))
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, callback)
    redraw()
    while True:
        cv2.imshow(title, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and len(points) == count:
            cv2.destroyWindow(title)
            return points
        if key in (ord("r"), ord("R")):
            points.clear()
            redraw()
        if key == 27:
            cv2.destroyWindow(title)
            raise RuntimeError("Point selection cancelled.")


def normalized_x(
    pixel: tuple[float, float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    calibration_size: tuple[int, int],
    image: np.ndarray,
) -> float:
    height, width = image.shape[:2]
    calibration_width, calibration_height = calibration_size
    matrix = camera_matrix.copy()
    matrix[0, :] *= width / calibration_width
    matrix[1, :] *= height / calibration_height
    point = np.asarray([[pixel]], dtype=np.float32)
    return float(cv2.undistortPoints(point, matrix, distortion)[0, 0, 0])


def main() -> int:
    args = parse_args()
    depth_samples = args.sample or []
    height_samples = args.height_sample or []
    multi_height_samples = args.multi_height_sample or []
    modes = sum(bool(values) for values in (depth_samples, height_samples, multi_height_samples))
    if modes != 1:
        raise ValueError(
            "Provide exactly one of --sample, --height-sample, or --multi-height-sample."
        )
    if (height_samples or multi_height_samples) and args.reference_depth is None:
        raise ValueError(
            "--reference-depth is required with --height-sample or --multi-height-sample."
        )
    if args.reference_depth is not None and args.reference_depth <= 0:
        raise ValueError("--reference-depth must be positive.")
    raw_samples = depth_samples or height_samples
    camera_matrix, distortion, calibration_size = load_intrinsics(
        args.camera_calibration
    )
    samples = []
    if multi_height_samples:
        for index, raw in enumerate(multi_height_samples):
            left_path, right_path, heights_raw, baseline_raw = raw
            try:
                heights = [float(value.strip()) for value in heights_raw.split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid height list in multi sample {index}: {heights_raw}"
                ) from exc
            if not heights:
                raise ValueError(f"Multi sample {index} has no heights.")
            baseline = float(baseline_raw)
            if baseline <= 0:
                raise ValueError("Baseline must be positive.")
            left = cv2.imread(left_path, cv2.IMREAD_COLOR)
            right = cv2.imread(right_path, cv2.IMREAD_COLOR)
            if left is None or right is None:
                raise RuntimeError(f"Could not read multi sample {index}: {left_path}, {right_path}")
            print(f"Select {len(heights)} objects in LEFT image, in height-list order.")
            left_pixels = click_centers(left, f"Multi sample {index} left", len(heights))
            print(f"Select the same {len(heights)} objects in RIGHT image, in the same order.")
            right_pixels = click_centers(right, f"Multi sample {index} right", len(heights))
            for height, left_pixel, right_pixel in zip(heights, left_pixels, right_pixels):
                depth = args.reference_depth - height
                if depth <= 0:
                    raise ValueError(
                        f"Height {height} is not below reference depth {args.reference_depth}."
                    )
                left_x = normalized_x(left_pixel, camera_matrix, distortion, calibration_size, left)
                right_x = normalized_x(right_pixel, camera_matrix, distortion, calibration_size, right)
                disparity = left_x - right_x
                if disparity <= 0:
                    raise ValueError("A multi-sample disparity is non-positive; check click order.")
                samples.append({
                    "depth": depth,
                    "height": height,
                    "baseline": baseline,
                    "disparity": disparity,
                    "left_pixel": list(left_pixel),
                    "right_pixel": list(right_pixel),
                })
        input_mode = "height"
    else:
        input_mode = "height" if height_samples else "depth"
    for index, raw in enumerate(raw_samples):
        left_path, right_path, depth_raw, baseline_raw = raw
        measured_value = float(depth_raw)
        depth = (
            args.reference_depth - measured_value
            if height_samples
            else measured_value
        )
        baseline = float(baseline_raw)
        if depth <= 0 or baseline <= 0:
            if height_samples:
                raise ValueError(
                    f"Sample {index} produces non-positive depth. "
                    "Height must be less than --reference-depth."
                )
            raise ValueError("Depth and baseline must be positive.")
        left = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError(f"Could not read sample {index}: {left_path}, {right_path}")
        left_pixel = click_center(left, f"Sample {index} left")
        right_pixel = click_center(right, f"Sample {index} right")
        left_x = normalized_x(left_pixel, camera_matrix, distortion, calibration_size, left)
        right_x = normalized_x(right_pixel, camera_matrix, distortion, calibration_size, right)
        disparity = left_x - right_x
        if disparity <= 0:
            raise ValueError(
                f"Sample {index} has non-positive disparity ({disparity:.6g}); "
                "check left/right order and camera motion direction."
            )
        samples.append(
            {
                "depth": depth,
                "height": measured_value if height_samples else None,
                "baseline": baseline,
                "disparity": disparity,
                "left_pixel": list(left_pixel),
                "right_pixel": list(right_pixel),
            }
        )

    if len(samples) < 3:
        raise ValueError("Use at least three samples at different depths.")
    depths = np.asarray([sample["depth"] for sample in samples])
    baselines = np.asarray([sample["baseline"] for sample in samples])
    disparities = np.asarray([sample["disparity"] for sample in samples])
    design = np.column_stack((baselines / depths, np.ones(len(samples))))
    scale, offset = np.linalg.lstsq(design, disparities, rcond=None)[0]
    predicted = design @ np.array([scale, offset])
    rmse = float(np.sqrt(np.mean((predicted - disparities) ** 2)))
    if scale <= 0:
        raise ValueError("Fitted scale is non-positive; verify the sample geometry.")

    result = {
        "version": 1,
        "model": "normalized_disparity = scale * baseline / depth + offset",
        "input_mode": input_mode,
        "reference_depth": args.reference_depth if (height_samples or multi_height_samples) else None,
        "scale": float(scale),
        "offset": float(offset),
        "rmse_normalized_disparity": rmse,
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved depth calibration to {args.output}")
    print(f"scale={scale:.9f}, offset={offset:.9f}, RMSE={rmse:.9f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
