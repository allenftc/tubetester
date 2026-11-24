#!/usr/bin/env python3
"""
Create a pixel -> real-world coordinate mapping from one checkerboard or
regular-grid image.

The mapping is valid on the checkerboard plane (and on a work surface that is
coplanar with it). Coordinates are in the same units as --pitch.

Example:
    python checkerboard_mapper.py ^
        --image board.png ^
        --corners 9 6 ^
        --pitch 10 ^
        --origin-offset 125 80 ^
        --output mapping.json ^
        --query 640 360

For a non-checkerboard grid, first create a numbered candidate image:
    python checkerboard_mapper.py --image crate.jpg --detect-grid

Then use selected candidate IDs and their integer grid coordinates:
    python checkerboard_mapper.py --image crate.jpg --detect-grid ^
        --grid-pixel-ids 12 27 31 44 ^
        --grid-indices "0,0;1,0;0,1;1,1" --pitch 25
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a checkerboard and create a pixel-to-plane mapping."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image containing the checkerboard.")
    source.add_argument(
        "--camera",
        type=int,
        metavar="INDEX",
        help="Capture one frame from this camera index instead of reading an image.",
    )
    parser.add_argument(
        "--manual-grid",
        action="store_true",
        help=(
            "Open an image window and let you click exact grid intersections. "
            "Press Enter when finished; coordinates are entered in the terminal."
        ),
    )
    parser.add_argument(
        "--corners",
        type=int,
        nargs=2,
        metavar=("COLUMNS", "ROWS"),
        help="Number of inner checkerboard corners, not number of squares.",
    )
    parser.add_argument(
        "--detect-grid",
        action="store_true",
        help=(
            "Detect generic grid-intersection candidates with Shi-Tomasi. "
            "Use --grid-pixel-ids and --grid-indices to create a mapping."
        ),
    )
    parser.add_argument(
        "--grid-pixel-ids",
        type=int,
        nargs="+",
        metavar="ID",
        help="Candidate IDs selected from the --detect-grid numbered image.",
    )
    parser.add_argument(
        "--grid-indices",
        type=str,
        help='Integer grid coordinates matching the IDs, e.g. "0,0;1,0;0,1;1,1".',
    )
    parser.add_argument(
        "--max-grid-candidates",
        type=int,
        default=100,
        help="Maximum number of structural grid intersections to display/save.",
    )
    parser.add_argument(
        "--grid-min-line-length",
        type=float,
        default=45.0,
        help="Minimum blue rail length in pixels for structural grid detection.",
    )
    parser.add_argument(
        "--grid-line-thickness",
        type=int,
        default=5,
        help="Morphological thickness used to join broken blue rail segments.",
    )
    parser.add_argument(
        "--grid-snap-radius",
        type=int,
        default=18,
        help="Radius in pixels for snapping a line crossing to the rail-junction center.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=1.0,
        help="Checkerboard square pitch in your desired real-world units.",
    )
    parser.add_argument(
        "--absolute-coordinates",
        action="store_true",
        help=(
            "In --manual-grid mode, treat entered coordinates as real-world "
            "coordinates directly instead of grid indices multiplied by --pitch."
        ),
    )
    parser.add_argument(
        "--origin-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("X", "Y"),
        help=(
            "Real-world coordinates assigned to the checkerboard center. "
            "For example, --origin-offset 125 80."
        ),
    )
    parser.add_argument(
        "--distance",
        type=float,
        help=(
            "Optional camera-to-board distance to record as metadata. "
            "It is not required for a planar pixel-to-coordinate mapping."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mapping.json"),
        help="Output JSON file containing the mapping matrix.",
    )
    parser.add_argument(
        "--query",
        type=float,
        nargs=2,
        action="append",
        metavar=("PIXEL_X", "PIXEL_Y"),
        help="Convert a pixel coordinate immediately; may be supplied more than once.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the detected corners and a coordinate-axis overlay.",
    )
    return parser.parse_args()


def load_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image is not None:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.image}")
        return frame

    capture = cv2.VideoCapture(args.camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {args.camera}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("The camera did not return a frame.")
        return frame
    finally:
        capture.release()


def find_checkerboard(
    frame: np.ndarray, columns: int, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pattern = (columns, rows)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern,
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

    if not found or corners is None:
        raise RuntimeError(
            f"Could not find a {columns} x {rows} inner-corner checkerboard. "
            "Check the corner count, lighting, focus, and that the full board is visible."
        )

    # Refine the legacy detector's output. SB corners are already highly accurate,
    # but refining both paths is harmless and improves consistency.
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-4,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return refined.reshape(-1, 2), gray


def find_grid_candidates(
    frame: np.ndarray,
    max_candidates: int,
    min_line_length: float,
    line_thickness: int,
    snap_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # The crate rails are blue. Restricting the detector to that color avoids
    # treating the label, wood grain, and shadows as grid intersections.
    mask = cv2.inRange(hsv, (90, 45, 25), (145, 255, 255))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (max(3, line_thickness), max(3, line_thickness))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 720,
        threshold=max(20, int(min_line_length * 0.55)),
        minLineLength=min_line_length,
        maxLineGap=max(10, int(min_line_length * 0.25)),
    )
    if lines is None:
        raise RuntimeError(
            "Could not detect structural blue rails. Try adjusting "
            "--grid-min-line-length or the HSV thresholds in find_grid_candidates."
        )

    segments = lines[:, 0].astype(np.float64)
    angles = np.mod(
        np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0]),
        np.pi,
    )
    # The crate has two rail families. A doubled angle makes each family
    # circularly clusterable even when the image is rotated.
    vectors = np.column_stack((np.cos(2 * angles), np.sin(2 * angles)))
    first_axis = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
    scores = vectors @ first_axis
    family_a = scores >= 0
    family_b = ~family_a
    if family_a.sum() < 2 or family_b.sum() < 2:
        raise RuntimeError("Could not separate the crate rails into two directions.")

    def representative_line(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p1 = segment[:2]
        p2 = segment[2:]
        direction = p2 - p1
        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        if normal[0] < 0 or (abs(normal[0]) < 1e-6 and normal[1] < 0):
            normal = -normal
        return normal, np.array([normal @ p1, direction[0], direction[1]])

    def merge_family(selected: np.ndarray) -> list[tuple[np.ndarray, float]]:
        family_segments = segments[selected]
        entries = []
        for segment in family_segments:
            normal, data = representative_line(segment)
            entries.append((normal, data[0]))
        entries.sort(key=lambda item: item[1])
        merged: list[tuple[np.ndarray, float]] = []
        merge_distance = max(10.0, min_line_length * 0.18)
        for normal, offset in entries:
            if merged and abs(offset - merged[-1][1]) < merge_distance:
                old_normal, old_offset = merged[-1]
                combined = old_normal + normal
                combined_norm = np.linalg.norm(combined)
                if combined_norm > 1e-6:
                    merged[-1] = (
                        combined / combined_norm,
                        (old_offset + offset) / 2.0,
                    )
            else:
                merged.append((normal, offset))
        return merged

    family_lines = [merge_family(family_a), merge_family(family_b)]
    candidates = []
    height, width = mask.shape
    for normal_a, offset_a in family_lines[0]:
        for normal_b, offset_b in family_lines[1]:
            matrix = np.vstack((normal_a, normal_b))
            if abs(np.linalg.det(matrix)) < 0.15:
                continue
            point = np.linalg.solve(matrix, np.array([offset_a, offset_b]))
            x, y = point
            if -5 <= x < width + 5 and -5 <= y < height + 5:
                # Hough lines describe rail centerlines independently, so
                # their infinite intersection can be several pixels away from
                # the visible junction. Snap to the local thickest blue region.
                x0 = max(0, int(round(x)) - snap_radius)
                x1 = min(width, int(round(x)) + snap_radius + 1)
                y0 = max(0, int(round(y)) - snap_radius)
                y1 = min(height, int(round(y)) + snap_radius + 1)
                local = distance[y0:y1, x0:x1]
                if local.size:
                    peak_y, peak_x = np.unravel_index(np.argmax(local), local.shape)
                    snapped = np.array([x0 + peak_x, y0 + peak_y], dtype=np.float32)
                    if distance[int(snapped[1]), int(snapped[0])] >= 2.0:
                        candidates.append(snapped)
    if len(candidates) < 4:
        raise RuntimeError(
            "Could not find at least four structural rail intersections. "
            "Try lowering --grid-min-line-length."
        )
    points = np.asarray(candidates, dtype=np.float32)
    # Several nearly parallel Hough segments can generate the same junction.
    # Collapse those duplicates after snapping.
    deduplicated = []
    for point in points:
        if not any(np.linalg.norm(point - existing) < 14 for existing in deduplicated):
            deduplicated.append(point)
    points = np.asarray(deduplicated, dtype=np.float32)
    if len(points) > max_candidates:
        center = np.array([width / 2, height / 2])
        keep = np.argsort(np.linalg.norm(points - center, axis=1))[:max_candidates]
        points = points[keep]
    return points, mask


def save_grid_candidates(
    frame: np.ndarray, candidates: np.ndarray, output: Path
) -> None:
    overlay = frame.copy()
    for index, point in enumerate(candidates):
        x, y = np.round(point).astype(int)
        label = str(index)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )
        text_x = min(max(2, x + 8), max(2, overlay.shape[1] - text_width - 2))
        text_y = min(
            max(text_height + 2, y - 8),
            max(text_height + 2, overlay.shape[0] - baseline - 2),
        )
        cv2.line(overlay, (x, y), (text_x, text_y - text_height // 2), (0, 0, 0), 3)
        cv2.circle(overlay, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            font,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output), overlay)


def select_grid_points_manually(frame: np.ndarray) -> np.ndarray:
    points: list[tuple[float, float]] = []
    window = (
        "Left click=add/drag | Wheel=zoom | Middle/right drag=pan | "
        "Enter=finish R=reset Esc=cancel"
    )
    height, width = frame.shape[:2]
    zoom = 1.0
    pan_x = 0.0
    pan_y = 0.0
    dragging_point: int | None = None
    panning = False
    pan_start = (0, 0)
    pan_origin = (0.0, 0.0)
    display = frame.copy()

    def image_to_screen(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((point[0] - pan_x) * zoom)),
            int(round((point[1] - pan_y) * zoom)),
        )

    def screen_to_image(x: int, y: int) -> tuple[float, float]:
        return (
            x / zoom + pan_x,
            y / zoom + pan_y,
        )

    def redraw() -> None:
        nonlocal display
        matrix = np.array(
            [[zoom, 0, -pan_x * zoom], [0, zoom, -pan_y * zoom]],
            dtype=np.float32,
        )
        display = cv2.warpAffine(
            frame,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(35, 35, 35),
        )
        for index, (x, y) in enumerate(points):
            center = image_to_screen((x, y))
            label = str(index)
            cv2.circle(display, center, 7, (0, 0, 255), 2)
            cv2.putText(
                display,
                label,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                label,
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def nearest_point(x: int, y: int) -> int | None:
        if not points:
            return None
        distances = [
            np.hypot(x - image_to_screen(point)[0], y - image_to_screen(point)[1])
            for point in points
        ]
        index = int(np.argmin(distances))
        return index if distances[index] <= max(12, 10 * zoom / 2) else None

    def on_click(event: int, x: int, y: int, flags: int, _data: object) -> None:
        nonlocal dragging_point, panning, pan_start, pan_origin, zoom, pan_x, pan_y
        if event == cv2.EVENT_LBUTTONDOWN:
            existing = nearest_point(x, y)
            if existing is not None:
                dragging_point = existing
            else:
                image_point = screen_to_image(x, y)
                if 0 <= image_point[0] < width and 0 <= image_point[1] < height:
                    points.append(image_point)
            redraw()
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_point is not None:
                points[dragging_point] = screen_to_image(x, y)
                redraw()
            elif panning:
                pan_x = pan_origin[0] - (x - pan_start[0]) / zoom
                pan_y = pan_origin[1] - (y - pan_start[1]) / zoom
                redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point = None
        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            panning = True
            pan_start = (x, y)
            pan_origin = (pan_x, pan_y)
        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_zoom = zoom
            zoom = min(12.0, max(0.25, zoom * (1.2 if flags > 0 else 1 / 1.2)))
            image_at_cursor = (x / old_zoom + pan_x, y / old_zoom + pan_y)
            pan_x = image_at_cursor[0] - x / zoom
            pan_y = image_at_cursor[1] - y / zoom
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_click)
    redraw()
    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key in (ord("r"), ord("R")):
            points.clear()
            zoom = 1.0
            pan_x = 0.0
            pan_y = 0.0
            redraw()
        if key == 27:  # Escape
            cv2.destroyWindow(window)
            raise RuntimeError("Manual grid selection cancelled.")
    cv2.destroyWindow(window)
    if len(points) < 4:
        raise ValueError("Select at least four exact grid intersections.")
    return np.asarray(points, dtype=np.float32)


def prompt_grid_indices(count: int, frame: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    window = "Selected points | keep this window open while entering coordinates"
    annotated = frame.copy()
    for index, point in enumerate(image_points):
        x, y = np.round(point).astype(int)
        label = str(index)
        cv2.circle(annotated, (x, y), 7, (0, 0, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    print(f"Enter {count} coordinates in click order.")
    values: list[tuple[float, float]] = []
    error: list[str] = []
    finished = threading.Event()

    def read_coordinates() -> None:
        for index in range(count):
            while True:
                try:
                    raw = input(f"Point {index}: ").strip().replace(",", " ")
                    x, y = raw.split()[:2]
                    values.append((float(x), float(y)))
                    break
                except (ValueError, EOFError):
                    print("Please enter two numbers, for example: 120.5 80.0")
        finished.set()

    thread = threading.Thread(target=read_coordinates, daemon=True)
    thread.start()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    while not finished.is_set():
        cv2.imshow(window, annotated)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            error.append("Coordinate entry cancelled.")
            break
    cv2.destroyWindow(window)
    if error:
        raise RuntimeError(error[0])
    thread.join(timeout=0.1)
    return np.asarray(values, dtype=np.float32)


def build_mapping(
    image_points: np.ndarray,
    columns: int,
    rows: int,
    pitch: float,
    origin_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if pitch <= 0:
        raise ValueError("--pitch must be greater than zero.")

    center_col = (columns - 1) / 2.0
    center_row = (rows - 1) / 2.0
    object_points = np.array(
        [
            [
                (col - center_col) * pitch + origin_offset[0],
                (row - center_row) * pitch + origin_offset[1],
            ]
            for row in range(rows)
            for col in range(columns)
        ],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(image_points, object_points, method=0)
    if homography is None:
        raise RuntimeError("Could not calculate the pixel-to-world homography.")
    homography /= homography[2, 2]
    return homography, object_points


def parse_grid_indices(value: str) -> np.ndarray:
    try:
        pairs = [item.strip().split(",") for item in value.split(";")]
        indices = np.array([[float(x), float(y)] for x, y in pairs], dtype=np.float32)
    except (ValueError, TypeError):
        raise ValueError(
            '--grid-indices must look like "0,0;1,0;0,1;1,1".'
        ) from None
    return indices


def build_grid_mapping(
    image_points: np.ndarray,
    grid_indices: np.ndarray,
    pitch: float,
    origin_offset: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if pitch <= 0:
        raise ValueError("--pitch must be greater than zero.")
    if len(image_points) != len(grid_indices) or len(image_points) < 4:
        raise ValueError("A grid mapping needs at least four matching point/index pairs.")
    world_points = grid_indices * pitch + np.asarray(origin_offset, dtype=np.float32)
    homography, _ = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
    if homography is None:
        raise RuntimeError("Could not calculate the grid pixel-to-world homography.")
    homography /= homography[2, 2]
    return homography, world_points


def pixel_to_world(mapping: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float]:
    point = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
    world = cv2.perspectiveTransform(point, mapping)[0, 0]
    return float(world[0]), float(world[1])


def world_to_pixel(mapping: np.ndarray, world: tuple[float, float]) -> tuple[float, float]:
    inverse = np.linalg.inv(mapping)
    point = np.array([[[world[0], world[1]]]], dtype=np.float64)
    pixel = cv2.perspectiveTransform(point, inverse)[0, 0]
    return float(pixel[0]), float(pixel[1])


def reprojection_error(
    mapping: np.ndarray, image_points: np.ndarray, world_points: np.ndarray
) -> float:
    predicted = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2).astype(np.float64), mapping
    ).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(predicted - world_points, axis=1)))


def show_overlay(
    frame: np.ndarray,
    image_points: np.ndarray,
    columns: int,
    rows: int,
    mapping: np.ndarray,
    origin_offset: tuple[float, float],
) -> None:
    overlay = frame.copy()
    points = image_points.astype(np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(
        overlay,
        (columns, rows),
        points,
        True,
    )

    origin = world_to_pixel(mapping, origin_offset)
    x_axis = world_to_pixel(mapping, (origin_offset[0] + 50, origin_offset[1]))
    y_axis = world_to_pixel(mapping, (origin_offset[0], origin_offset[1] + 50))
    origin_i = tuple(np.round(origin).astype(int))
    x_axis_i = tuple(np.round(x_axis).astype(int))
    y_axis_i = tuple(np.round(y_axis).astype(int))
    cv2.arrowedLine(overlay, origin_i, x_axis_i, (0, 0, 255), 2, tipLength=0.15)
    cv2.arrowedLine(overlay, origin_i, y_axis_i, (0, 255, 0), 2, tipLength=0.15)
    cv2.putText(overlay, "X", x_axis_i, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(overlay, "Y", y_axis_i, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Checkerboard mapping (press any key to close)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    if sum(bool(value) for value in (args.detect_grid, args.manual_grid)) > 1:
        raise ValueError("Use only one of --detect-grid or --manual-grid.")
    if not args.detect_grid and not args.manual_grid and args.corners is None:
        raise ValueError("--corners is required unless --detect-grid or --manual-grid is used.")

    frame = load_frame(args)
    if args.manual_grid:
        image_points = select_grid_points_manually(frame)
        coordinates = prompt_grid_indices(len(image_points), frame, image_points)
        if args.absolute_coordinates:
            world_points = coordinates
            mapping, _ = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
            if mapping is None:
                raise RuntimeError("Could not calculate the pixel-to-world homography.")
            mapping /= mapping[2, 2]
        else:
            mapping, world_points = build_grid_mapping(
                image_points,
                coordinates,
                args.pitch,
                tuple(args.origin_offset),
            )
        point_kind = "manually selected grid points"
        columns, rows = 0, 0
    elif args.detect_grid:
        candidates, _ = find_grid_candidates(
            frame,
            args.max_grid_candidates,
            args.grid_min_line_length,
            args.grid_line_thickness,
            args.grid_snap_radius,
        )
        candidate_image = args.output.with_name(args.output.stem + "_candidates.jpg")
        save_grid_candidates(frame, candidates, candidate_image)
        candidates_json = args.output.with_name(args.output.stem + "_candidates.json")
        candidates_json.write_text(
            json.dumps(
                [{"id": i, "pixel": point.tolist()} for i, point in enumerate(candidates)],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Detected {len(candidates)} grid-intersection candidates.")
        print(f"Numbered image: {candidate_image}")
        print(f"Candidate coordinates: {candidates_json}")
        if args.grid_pixel_ids is None or args.grid_indices is None:
            print(
                "Select at least four candidate IDs, then rerun with "
                "--grid-pixel-ids and --grid-indices."
            )
            return 0
        indices = parse_grid_indices(args.grid_indices)
        if len(args.grid_pixel_ids) != len(indices):
            raise ValueError("--grid-pixel-ids and --grid-indices must have equal lengths.")
        if any(index < 0 or index >= len(candidates) for index in args.grid_pixel_ids):
            raise ValueError("A --grid-pixel-ids value is outside the candidate list.")
        image_points = candidates[np.asarray(args.grid_pixel_ids, dtype=int)]
        mapping, world_points = build_grid_mapping(
            image_points,
            indices,
            args.pitch,
            tuple(args.origin_offset),
        )
        point_kind = "grid points"
        columns, rows = 0, 0
    else:
        columns, rows = args.corners
        if columns < 2 or rows < 2:
            raise ValueError("--corners must contain at least 2 columns and 2 rows.")
        image_points, _ = find_checkerboard(frame, columns, rows)
        mapping, world_points = build_mapping(
            image_points,
            columns,
            rows,
            args.pitch,
            tuple(args.origin_offset),
        )
        point_kind = "checkerboard corners"
    error = reprojection_error(mapping, image_points, world_points)

    result = {
        "version": 1,
        "coordinate_units": "same as pitch",
        "pixel_to_world_homography": mapping.tolist(),
        "world_to_pixel_homography": np.linalg.inv(mapping).tolist(),
        "checkerboard_inner_corners": [columns, rows] if not args.detect_grid else None,
        "grid_pixel_ids": args.grid_pixel_ids if args.detect_grid else None,
        "grid_indices": args.grid_indices if args.detect_grid else None,
        "manual_coordinates_are_absolute": (
            args.absolute_coordinates if args.manual_grid else None
        ),
        "pitch": args.pitch,
        "origin_offset": list(args.origin_offset),
        "camera_to_board_distance": args.distance,
        "mean_corner_reprojection_error": error,
        "mapping_scope": "the checkerboard/work-surface plane only",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Using {len(image_points)} {point_kind}.")
    print(f"Saved mapping to {args.output}")
    print(f"Mean corner reprojection error: {error:.6f} coordinate units")
    print("Pixel-to-world homography:")
    print(np.array2string(mapping, precision=10, suppress_small=False))

    for query in args.query or []:
        world = pixel_to_world(mapping, tuple(query))
        print(f"pixel ({query[0]:.3f}, {query[1]:.3f}) -> "
              f"world ({world[0]:.6f}, {world[1]:.6f})")

    if args.show and not args.detect_grid:
        show_overlay(
            frame,
            image_points,
            columns,
            rows,
            mapping,
            tuple(args.origin_offset),
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
