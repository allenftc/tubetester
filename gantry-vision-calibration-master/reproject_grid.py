#!/usr/bin/env python3
"""
Project a real-world grid into an image using checkerboard_mapper.py output.

Examples:
    python reproject_grid.py --mapping mapping.json --image workarea.jpg --spacing 25
    python reproject_grid.py --mapping mapping.json --camera 0 --spacing 10 --show

The spacing uses the same units as the calibration pitch. Grid coordinates are
world coordinates, so the generated grid is aligned with the calibration axes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a real-world coordinate grid using a saved homography."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Image on which to draw the grid.")
    source.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Camera index for live projection.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="JSON file produced by checkerboard_mapper.py.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        required=True,
        help="Distance between grid lines/points in calibration units.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reprojected_grid.jpg"),
        help="Output image path for still-image mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the result in a window.",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Draw world X,Y labels beside each projected grid point.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Radius of each projected grid point in pixels.",
    )
    parser.add_argument(
        "--min-pixel-separation",
        type=float,
        default=5.0,
        help="Minimum screen-pixel separation between drawn markers; use 0 for all.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=1,
        help="Grid line width in pixels.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=1000000,
        help="Safety limit for projected points.",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pixel_to_world = np.asarray(data["pixel_to_world_homography"], dtype=np.float64)
        world_to_pixel = np.asarray(
            data.get("world_to_pixel_homography", np.linalg.inv(pixel_to_world)),
            dtype=np.float64,
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read calibration mapping {path}: {exc}") from exc
    if pixel_to_world.shape != (3, 3) or world_to_pixel.shape != (3, 3):
        raise RuntimeError("The mapping file must contain 3x3 homography matrices.")
    return pixel_to_world, world_to_pixel


def load_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image is not None:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        return frame
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("The camera did not return a frame.")
    return frame


def open_camera(index: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera index {index}")
    return capture


def pixel_to_world(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float64), matrix
    ).reshape(-1, 2)


def world_to_pixel(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float64), matrix
    ).reshape(-1, 2)


def visible_world_bounds(
    pixel_to_world_matrix: np.ndarray, width: int, height: int
) -> tuple[float, float, float, float]:
    corners = pixel_to_world(
        pixel_to_world_matrix,
        np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
    )
    return (
        float(np.min(corners[:, 0])),
        float(np.max(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 1])),
    )


def grid_values(low: float, high: float, spacing: float) -> np.ndarray:
    first = np.ceil(low / spacing - 1e-9) * spacing
    last = np.floor(high / spacing + 1e-9) * spacing
    if last < first:
        return np.empty(0, dtype=np.float64)
    return np.arange(first, last + spacing * 0.5, spacing, dtype=np.float64)


def draw_grid(
    frame: np.ndarray,
    pixel_to_world_matrix: np.ndarray,
    world_to_pixel_matrix: np.ndarray,
    spacing: float,
    point_radius: int,
    line_width: int,
    labels: bool,
    max_points: int,
    min_pixel_separation: float,
) -> tuple[np.ndarray, int, int, float]:
    height, width = frame.shape[:2]
    x_min, x_max, y_min, y_max = visible_world_bounds(
        pixel_to_world_matrix, width, height
    )
    xs = grid_values(x_min, x_max, spacing)
    ys = grid_values(y_min, y_max, spacing)
    if len(xs) * len(ys) > max_points:
        raise RuntimeError(
            f"The requested spacing creates {len(xs) * len(ys)} points; "
            f"increase --spacing or set --max-points to at least "
            f"{len(xs) * len(ys)}."
        )

    output = frame.copy()
    # Draw projected grid lines, clipping naturally at the image boundary.
    for x in xs:
        world_line = np.array([[x, y_min], [x, y_max]], dtype=np.float64)
        pixels = world_to_pixel(world_to_pixel_matrix, world_line)
        cv2.line(
            output,
            tuple(np.round(pixels[0]).astype(int)),
            tuple(np.round(pixels[1]).astype(int)),
            (255, 180, 0),
            line_width,
            cv2.LINE_AA,
        )
    for y in ys:
        world_line = np.array([[x_min, y], [x_max, y]], dtype=np.float64)
        pixels = world_to_pixel(world_to_pixel_matrix, world_line)
        cv2.line(
            output,
            tuple(np.round(pixels[0]).astype(int)),
            tuple(np.round(pixels[1]).astype(int)),
            (0, 180, 255),
            line_width,
            cv2.LINE_AA,
        )

    world_points = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    pixel_points = world_to_pixel(world_to_pixel_matrix, world_points)
    visible = (
        (pixel_points[:, 0] >= 0)
        & (pixel_points[:, 0] < width)
        & (pixel_points[:, 1] >= 0)
        & (pixel_points[:, 1] < height)
    )
    visible_world = world_points[visible]
    visible_pixels = pixel_points[visible]
    if len(visible_pixels) == 0:
        return output, 0, 0, 0.0
    center_world = np.mean(visible_world, axis=0)
    sample_world = np.vstack((center_world, center_world + [spacing, 0.0]))
    sample_pixels = world_to_pixel(world_to_pixel_matrix, sample_world)
    projected_spacing = float(np.linalg.norm(sample_pixels[1] - sample_pixels[0]))

    count = 0
    drawn_pixels: list[np.ndarray] = []
    for world, pixel in zip(visible_world, visible_pixels):
        if min_pixel_separation > 0 and any(
            np.linalg.norm(pixel - drawn) < min_pixel_separation
            for drawn in drawn_pixels
        ):
            continue
        px, py = np.round(pixel).astype(int)
        cv2.circle(output, (px, py), point_radius, (0, 0, 255), -1, cv2.LINE_AA)
        if labels:
            text = f"({world[0]:g},{world[1]:g})"
            cv2.putText(
                output, text, (px + 6, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                output, text, (px + 6, py - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1, cv2.LINE_AA
            )
        drawn_pixels.append(pixel)
        count += 1
    return output, count, len(visible_pixels), projected_spacing


def main() -> int:
    args = parse_args()
    if args.spacing <= 0:
        raise ValueError("--spacing must be greater than zero.")
    if args.point_radius < 1 or args.line_width < 1:
        raise ValueError("--point-radius and --line-width must be positive.")
    pixel_to_world_matrix, world_to_pixel_matrix = load_mapping(args.mapping)
    if args.image is not None:
        frame = load_frame(args)
        output, count, visible_count, projected_spacing = draw_grid(
            frame,
            pixel_to_world_matrix,
            world_to_pixel_matrix,
            args.spacing,
            args.point_radius,
            args.line_width,
            args.labels,
            args.max_points,
            args.min_pixel_separation,
        )
        if not cv2.imwrite(str(args.output), output):
            raise RuntimeError(f"Could not write output image: {args.output}")
        print(
            f"Saved {count} visible markers to {args.output} "
            f"({visible_count} visible points before overlap filtering)."
        )
    else:
        capture = open_camera(args.camera)
        print("Live projection started. Press Q or Esc to close.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise RuntimeError("The camera stopped returning frames.")
            output, count, visible_count, projected_spacing = draw_grid(
                frame,
                pixel_to_world_matrix,
                world_to_pixel_matrix,
                args.spacing,
                args.point_radius,
                args.line_width,
                args.labels,
                args.max_points,
                args.min_pixel_separation,
            )
            cv2.imshow("Reprojected grid", output)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
        capture.release()
        cv2.destroyAllWindows()
    print(f"Approximate projected spacing near image center: {projected_spacing:.2f} pixels")
    if args.show and args.image is not None:
        cv2.imshow("Reprojected grid", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
