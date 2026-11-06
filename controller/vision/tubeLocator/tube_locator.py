"""HSV-based tube localisation inside a moving region of interest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import json


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]
Circle = Tuple[int, int, int]
STEREO_BASELINE_MM = 10.0


@dataclass(frozen=True)
class ColorRange:
    """HSV range for classifying a tube cap color.

    A hue segment whose lower bound exceeds its upper bound wraps through 180,
    which is where red sits. ``priority`` breaks ties between ranges that both
    match a sample; ``relative_value`` lets a range accept pixels that are only
    bright relative to the ROI background, which is how a white cap reads under
    dim light.
    """

    name: str
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]
    priority: int = 1
    relative_value: bool = False


NEUTRAL_PRIORITY = 0
NEUTRAL_RELATIVE_VALUE_GAIN = 1.25
MIN_SAMPLE_VALUE = 40
MIN_COLOR_SAMPLES = 8

DEFAULT_COLOR_RANGES: tuple[ColorRange, ...] = (
    ColorRange("white", (0, 0, 200), (180, 60, 255), priority=NEUTRAL_PRIORITY, relative_value=True),
    ColorRange("red", (170, 100, MIN_SAMPLE_VALUE), (10, 255, 255)),
    ColorRange("pink", (140, 50, MIN_SAMPLE_VALUE), (170, 255, 255)),
    ColorRange("green", (40, 50, MIN_SAMPLE_VALUE), (85, 255, 255)),
    ColorRange("cyan", (85, 100, MIN_SAMPLE_VALUE), (105, 255, 255)),
    ColorRange("gray", (0, 0, MIN_SAMPLE_VALUE), (180, 60, 200), priority=NEUTRAL_PRIORITY),
)

CALIBRATION_PATH = Path(__file__).resolve().with_name("grid_mapping.json")

with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
    calibration = json.load(file)

pixel_to_world = np.array(
    calibration["pixel_to_world_homography"],
    dtype=np.float64,
)


def pixel_to_real(pixel_x: float, pixel_y: float) -> tuple[float, float]:
    pixel = np.array(
        [[[pixel_x, pixel_y]]],
        dtype=np.float64,
    )

    real = cv2.perspectiveTransform(pixel, pixel_to_world)[0, 0]
    return float(real[0]), float(real[1])


def triangulate_stereo_circles(
	left_pixel: Point,
	right_pixel: Point,
	image_size: tuple[int, int],
	baseline_mm: float = STEREO_BASELINE_MM,
) -> tuple[float, float, float]:
	"""Estimate a circle's midpoint-camera XYZ from two horizontally shifted views."""
	width, height = image_size
	if width <= 0 or height <= 0:
		raise ValueError("image_size must contain positive width and height")
	if not np.isfinite(baseline_mm) or baseline_mm <= 0:
		raise ValueError("baseline_mm must be finite and greater than zero")
	if not np.isfinite((*left_pixel, *right_pixel)).all():
		raise ValueError("stereo pixel coordinates must be finite")

	calibration_path = Path(__file__).with_name("camera_calibration.npz")
	with np.load(calibration_path) as calibration:
		required = {"camera_matrix", "distortion_coefficients", "image_size"}
		missing = required.difference(calibration.files)
		if missing:
			raise ValueError(f"Camera calibration is missing arrays: {', '.join(sorted(missing))}")
		camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
		distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
		calibration_width, calibration_height = map(int, calibration["image_size"])

	if (
		camera_matrix.shape != (3, 3)
		or distortion.size == 0
		or not np.isfinite(camera_matrix).all()
		or not np.isfinite(distortion).all()
	):
		raise ValueError("Camera calibration has invalid camera matrix or distortion coefficients")
	if calibration_width <= 0 or calibration_height <= 0:
		raise ValueError("Camera calibration image_size must contain positive dimensions")

	scaled_matrix = camera_matrix.copy()
	scaled_matrix[0, :] *= width / calibration_width
	scaled_matrix[1, :] *= height / calibration_height
	left = np.asarray([[left_pixel]], dtype=np.float32)
	right = np.asarray([[right_pixel]], dtype=np.float32)
	left_normalized = cv2.undistortPoints(left, scaled_matrix, distortion)[0, 0]
	right_normalized = cv2.undistortPoints(right, scaled_matrix, distortion)[0, 0]
	disparity = float(left_normalized[0] - right_normalized[0])
	if not np.isfinite(disparity) or disparity <= 0:
		raise ValueError("Stereo circle disparity must be positive; verify left/right capture order and baseline")

	depth = baseline_mm / disparity
	x_from_left_camera = float(left_normalized[0]) * depth
	y_from_midpoint_camera = (
		float(left_normalized[1]) + float(right_normalized[1])
	) * depth / 2.0
	return x_from_left_camera - baseline_mm / 2.0, y_from_midpoint_camera, depth


def resize_to_resolution(
	image: np.ndarray,
	target_width: int,
	target_height: int,
) -> np.ndarray:
	"""Resize the entire image to the requested dimensions without cropping."""
	return cv2.resize(
		image,
		(target_width, target_height),
		interpolation=cv2.INTER_AREA,
	)


def resize_to_240p(image: np.ndarray) -> np.ndarray:
	"""Resize the entire image to 320x240 without cropping."""
	return resize_to_resolution(image, 320, 240)


def resize_to_480p(image: np.ndarray) -> np.ndarray:
	"""Resize the entire image to 640x480 without cropping."""
	return resize_to_resolution(image, 640, 480)


class TubeLocator:
	"""Locate the largest thresholded contour inside a rotated dynamic ROI.

	``roi`` is supplied for every frame as
	``(center_x, center_y, width, height)``. The rectangle is rotated around
	that center by ``roi_rotation_deg``. The returned point is in full-frame
	coordinates, rather than ROI coordinates.
	"""

	def __init__(
		self,
		lower_hsv: Tuple[int, int, int] = (100, 16, 0),
		upper_hsv: Tuple[int, int, int] = (130, 255, 255),
		invert_mask: bool = True,
		min_contour_area: float = 25.0,
		show_preview: bool = True,
		preview_window: str = "Tube locator",
		roi_rotation_deg: float = 45.0,
		presence_scan_fraction: float = 0.3,
		presence_threshold: float = 0.25,
		color_ranges: Optional[tuple[ColorRange, ...]] = None,
	) -> None:
		if not 0.0 < presence_scan_fraction <= 1.0:
			raise ValueError("presence_scan_fraction must be in the range (0, 1]")
		if not 0.0 <= presence_threshold <= 1.0:
			raise ValueError("presence_threshold must be in the range [0, 1]")

		self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
		self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
		self.invert_mask = invert_mask
		self.min_contour_area = min_contour_area
		self.show_preview = show_preview
		self.preview_window = preview_window
		self.roi_rotation_deg = roi_rotation_deg
		self.presence_scan_fraction = presence_scan_fraction
		self.presence_threshold = presence_threshold
		self.color_ranges = color_ranges if color_ranges is not None else DEFAULT_COLOR_RANGES

	def _classify_color(
		self,
		roi_image: np.ndarray,
		center: Point,
		radius: int = 8,
		samples_mask: Optional[np.ndarray] = None,
		num_samples: int = 24,
		min_agreement: float = 0.6,
	) -> Optional[str]:
		"""Classify the cap color from samples on a ring at half radius.

		Tubes have black holes in the center, so a solid patch sample lands on the
		hole. This samples ``num_samples`` points evenly spaced on a ring at half
		``radius``, keeps only pixels ``samples_mask`` marks as cap, drops samples too
		dark to be cap (shadow or the hole edge), and averages each sample over its
		3x3 neighbourhood to damp single-pixel spikes. A range flagged
		``relative_value`` also accepts pixels that are bright relative to the ROI
		background, which is how a white cap reads under dim light. Returns the
		winning range name, or ``None`` when fewer than ``min_agreement`` of the
		samples agree.
		"""
		if not self.color_ranges:
			return None

		height, width = roi_image.shape[:2]
		if height < 3 or width < 3:
			return None

		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		value_plane = hsv[:, :, 2]
		cap_mask = samples_mask.astype(bool) if samples_mask is not None else np.ones(
			(height, width), dtype=bool
		)
		background = ~cap_mask
		background_median = int(np.median(value_plane[background])) if int(background.sum()) else 0

		cx, cy = center
		ring_radius = min(max(1, int(round(radius * 0.5))), max(1, min(height, width) // 2))

		samples: list[tuple[int, int, int]] = []
		for index in range(num_samples):
			angle = 2.0 * np.pi * index / num_samples
			sx = int(round(cx + ring_radius * np.cos(angle)))
			sy = int(round(cy + ring_radius * np.sin(angle)))
			if not (0 <= sx < width and 0 <= sy < height):
				continue
			if not cap_mask[sy, sx]:
				continue
			y0, y1 = max(0, sy - 1), min(height, sy + 2)
			x0, x1 = max(0, sx + 1), min(width, sx + 2)
			sample = tuple(int(v) for v in hsv[y0:y1, x0:x1].mean(axis=(0, 1)))
			if sample[2] < MIN_SAMPLE_VALUE:
				continue
			samples.append(sample)

		if len(samples) < MIN_COLOR_SAMPLES:
			return None

		best_name: Optional[str] = None
		best_key = None
		for color_range in self.color_ranges:
			value_floor = color_range.lower[2]
			if color_range.relative_value and background_median:
				value_floor = min(value_floor, int(background_median * NEUTRAL_RELATIVE_VALUE_GAIN))
			count = 0
			hue_distance = 0
			for hue, saturation, value in samples:
				if self._color_range_matches(color_range, hue, saturation, value, value_floor):
					count += 1
				hue_distance += self._hue_distance(color_range, hue)
			if count == 0:
				continue
			key = (
				count,
				color_range.priority,
				-int(hue_distance / len(samples)),
				value_floor,
				color_range.name,
			)
			if best_key is None or key > best_key:
				best_key = key
				best_name = color_range.name

		if best_name is None or best_key[0] / len(samples) < min_agreement:
			return None
		return best_name

	@staticmethod
	def _color_range_matches(
		color_range: ColorRange,
		hue: int,
		saturation: int,
		value: int,
		value_floor: int,
	) -> bool:
		"""Return whether one HSV sample falls inside a range, honouring hue wrap."""
		lower_hue, lower_saturation, lower_value = color_range.lower
		upper_hue, upper_saturation, upper_value = color_range.upper
		if lower_hue <= upper_hue:
			hue_ok = lower_hue <= hue <= upper_hue
		else:
			hue_ok = hue >= lower_hue or hue <= upper_hue
		return (
			hue_ok
			and lower_saturation <= saturation <= upper_saturation
			and value_floor <= value <= upper_value
		)

	@staticmethod
	def _hue_distance(color_range: ColorRange, hue: int) -> int:
		"""Distance from a hue to the nearest edge of the range's hue segment."""
		lower_hue, _, _ = color_range.lower
		upper_hue, _, _ = color_range.upper
		if lower_hue <= upper_hue:
			if lower_hue <= hue <= upper_hue:
				return 0
			return min(abs(hue - lower_hue), abs(hue - upper_hue))
		if hue >= lower_hue or hue <= upper_hue:
			return 0
		return min(hue - upper_hue, lower_hue - hue)

	def locate(self, frame: np.ndarray, roi: ROI) -> Tuple[Optional[Point], bool, Optional[str]]:
		"""Return the detected center, whether a tube is present, and its cap color.

		Tube presence is gated by the fraction of foreground pixels in a central
		patch of the inverted ROI mask.
		"""
		center, tube_present, color, _ = self.locate_with_mask(frame, roi)
		return center, tube_present, color

	def locate_with_mask(
		self,
		frame: np.ndarray,
		roi: ROI,
	) -> Tuple[Optional[Point], bool, Optional[str], Optional[np.ndarray]]:
		"""Return the detection result, cap color, and raw HSV threshold mask for the ROI."""
		if frame is None or frame.size == 0:
			return None, False, None, None

		height, width = frame.shape[:2]
		center_x, center_y, roi_width, roi_height = roi
		if roi_width <= 1 or roi_height <= 1:
			return None, False, None, None

		roi_image, warp = self._warp_roi(frame, roi)
		rotated_corners = self._rotated_roi_corners(roi)
		valid_pixels = cv2.warpPerspective(
			np.full((height, width), 255, dtype=np.uint8),
			warp,
			(roi_width, roi_height),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		mask = raw_mask.copy()
		if self.invert_mask:
			mask = cv2.bitwise_not(mask)
		mask = cv2.bitwise_and(mask, valid_pixels)
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		mask = cv2.bitwise_and(mask, valid_pixels)

		scan_width = max(1, int(round(roi_width * self.presence_scan_fraction)))
		scan_height = max(1, int(round(roi_height * self.presence_scan_fraction)))
		scan_x0 = max(0, (roi_width - scan_width) // 2)
		scan_y0 = max(0, (roi_height - scan_height) // 2)
		scan_x1 = min(roi_width, scan_x0 + scan_width)
		scan_y1 = min(roi_height, scan_y0 + scan_height)
		center_mask = mask[scan_y0:scan_y1, scan_x0:scan_x1]
		center_valid = valid_pixels[scan_y0:scan_y1, scan_x0:scan_x1]
		valid_pixel_count = cv2.countNonZero(center_valid)
		foreground_ratio = (
			cv2.countNonZero(cv2.bitwise_and(center_mask, center_valid)) / valid_pixel_count
			if valid_pixel_count
			else 0.0
		)
		tube_present = foreground_ratio >= self.presence_threshold

		center: Optional[Point] = None
		color: Optional[str] = None
		if tube_present:
			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			valid = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
			if valid:
				contour = max(valid, key=cv2.contourArea)
				moments = cv2.moments(contour)
				if moments["m00"]:
					roi_center = np.array(
						[[[moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]]],
						dtype=np.float32,
					)
					original_center = cv2.perspectiveTransform(roi_center, np.linalg.inv(warp))[0, 0]
					center = (
						int(round(float(original_center[0]))),
						int(round(float(original_center[1]))),
					)
					roi_center_point = (
						int(round(float(roi_center[0, 0, 0]))),
						int(round(float(roi_center[0, 0, 1]))),
					)
					enclosing_center, enclosing_radius = cv2.minEnclosingCircle(contour)
					color = self._classify_color(
						roi_image,
						(
							int(round(float(enclosing_center[0]))),
							int(round(float(enclosing_center[1]))),
						),
						radius=int(enclosing_radius),
						samples_mask=mask,
					)

		if self.show_preview:
			preview = self.annotate(frame, roi, center, tube_present, color=color)
			cv2.imshow(
				self.preview_window,
				np.hstack((roi_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))),
			)
			cv2.imshow(f"{self.preview_window} - frame", preview)
			cv2.waitKey(1)

		return center, tube_present, color, raw_mask

	def locate_with_circles(
		self,
		frame: np.ndarray,
		roi: ROI,
	) -> Tuple[Optional[Point], bool, Optional[str], Optional[np.ndarray], list[Circle], Optional[np.ndarray]]:
		"""Detect circles and return a montage of the intermediate OpenCV Mats."""
		if frame is None or frame.size == 0:
			return None, False, None, None, [], None

		center_x, center_y, roi_width, roi_height = roi
		if roi_width <= 1 or roi_height <= 1:
			return None, False, None, None, [], None

		roi_image, warp = self._warp_roi(frame, roi)
		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
		min_radius = max(2, min(roi_width, roi_height) // 20)
		detected = cv2.HoughCircles(
			blurred,
			cv2.HOUGH_GRADIENT,
			dp=1,
			minDist=20,
			param1=130,
			param2=30,
			minRadius=20,
			maxRadius=0,
		)
		circles: list[Circle] = []
		roi_circles: list[Circle] = []
		if detected is not None:
			for x, y, radius in np.rint(detected[0]).astype(int):
				roi_circles.append((int(x), int(y), int(radius)))
				roi_circle = np.array([[[float(x), float(y)]]], dtype=np.float32)
				frame_circle = cv2.perspectiveTransform(roi_circle, np.linalg.inv(warp))[0, 0]
				circles.append((
					int(round(float(frame_circle[0]))),
					int(round(float(frame_circle[1]))),
					int(radius),
				))

		center: Optional[Point] = None
		color: Optional[str] = None
		if circles:
			largest = max(range(len(circles)), key=lambda i: circles[i][2])
			center = (circles[largest][0], circles[largest][1])
			roi_circle = roi_circles[largest]
			color = self._classify_color(roi_image, (roi_circle[0], roi_circle[1]), radius=roi_circle[2])
		debug_montage = self._circle_debug_montage(roi_image, gray, blurred, roi_circles)
		return center, bool(circles), color, raw_mask, circles, debug_montage

	def roi_offset_from_frame_point(self, point: Point, roi: ROI) -> tuple[float, float]:
		"""Return a frame point's offset from the ROI center in rotated ROI pixels."""
		center_x, center_y, roi_width, roi_height = roi
		rotated_corners = self._rotated_roi_corners(roi)
		destination_corners = np.array(
			[
				[0, 0],
				[roi_width - 1, 0],
				[roi_width - 1, roi_height - 1],
				[0, roi_height - 1],
			],
			dtype=np.float32,
		)
		warp = cv2.getPerspectiveTransform(rotated_corners, destination_corners)
		roi_point = cv2.perspectiveTransform(
			np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32),
			warp,
		)[0, 0]
		return (
			float(roi_point[0]) - (roi_width - 1) / 2.0,
			float(roi_point[1]) - (roi_height - 1) / 2.0,
		)

	def annotate(
		self,
		frame: np.ndarray,
		roi: ROI,
		center: Optional[Point],
		tube_present: bool,
		circles: Optional[list[Circle]] = None,
		color: Optional[str] = None,
	) -> np.ndarray:
		"""Draw the rotated ROI, its target center, and any detected center."""
		center_x, center_y, _, _ = roi
		preview = frame.copy()
		cv2.polylines(
			preview,
			[np.rint(self._rotated_roi_corners(roi)).astype(np.int32)],
			isClosed=True,
			color=(255, 120, 30),
			thickness=2,
		)
		for circle_x, circle_y, radius in circles or []:
			cv2.circle(preview, (circle_x, circle_y), radius, (0, 255, 0), 2)
			cv2.circle(preview, (circle_x, circle_y), 2, (0, 255, 0), -1)
		cv2.drawMarker(preview, (center_x, center_y), (0, 220, 255), cv2.MARKER_CROSS, 16, 2)

		if center is not None:
			cv2.drawMarker(preview, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
		status = "Tube present" if tube_present else "No tube present"
		if center is not None:
			real_x, real_y = pixel_to_real(center[0], center[1])
			status += f"  center ({real_x:.2f}, {real_y:.2f})"
		if color is not None:
			status += f"  color: {color}"
		cv2.putText(
			preview,
			status,
			(10, 25),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 0) if tube_present else (0, 0, 255),
			2,
		)
		return preview

	def _warp_roi(self, frame: np.ndarray, roi: ROI) -> tuple[np.ndarray, np.ndarray]:
		_, _, roi_width, roi_height = roi
		rotated_corners = self._rotated_roi_corners(roi)
		destination_corners = np.array(
			[
				[0, 0],
				[roi_width - 1, 0],
				[roi_width - 1, roi_height - 1],
				[0, roi_height - 1],
			],
			dtype=np.float32,
		)
		warp = cv2.getPerspectiveTransform(rotated_corners, destination_corners)
		roi_image = cv2.warpPerspective(
			frame,
			warp,
			(roi_width, roi_height),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0),
		)
		return roi_image, warp

	@staticmethod
	def _circle_debug_montage(
		roi_image: np.ndarray,
		gray: np.ndarray,
		blurred: np.ndarray,
		circles: list[Circle],
	) -> np.ndarray:
		height, width = gray.shape
		result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
		for x, y, radius in circles:
			cv2.circle(result, (x, y), radius, (0, 255, 0), 2)
			cv2.circle(result, (x, y), 2, (0, 255, 0), -1)

		panels = (
			("Rotated ROI", roi_image),
			("Grayscale", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
			("Gaussian blur (Hough input)", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)),
			("Hough circles", result),
		)
		tile_width = max(1, width * 2)
		tile_height = max(1, height * 2)
		tiles: list[np.ndarray] = []
		for label, image in panels:
			tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_NEAREST)
			cv2.rectangle(tile, (0, 0), (tile_width - 1, 20), (0, 0, 0), cv2.FILLED)
			cv2.putText(tile, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
			tiles.append(tile)
		return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))

	def _rotated_roi_corners(self, roi: ROI) -> np.ndarray:
		center_x, center_y, roi_width, roi_height = roi
		half_width = (roi_width - 1) / 2.0
		half_height = (roi_height - 1) / 2.0
		corners = np.array(
			[
				[center_x - half_width, center_y - half_height],
				[center_x + half_width, center_y - half_height],
				[center_x + half_width, center_y + half_height],
				[center_x - half_width, center_y + half_height],
			],
			dtype=np.float32,
		)
		rotation = cv2.getRotationMatrix2D(
			(center_x, center_y),
			self.roi_rotation_deg,
			1.0,
		)
		return cv2.transform(corners[None, :, :], rotation)[0]

	def close(self) -> None:
		"""Close preview windows."""
		if self.show_preview:
			cv2.destroyAllWindows()


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Test tube detection using an image or webcam."
	)
	parser.add_argument("image", nargs="?", help="Path to the input image")
	parser.add_argument(
		"--webcam",
		action="store_true",
		help="Use a webcam instead of an image file",
	)
	parser.add_argument(
		"--camera-index",
		type=int,
		default=0,
		help="Webcam device index (default: 0)",
	)
	parser.add_argument(
		"--backend",
		choices=("dshow", "msmf", "default"),
		default="dshow",
		help="Camera backend (default: dshow; try msmf if controls are unavailable)",
	)
	parser.add_argument(
		"--exposure",
		type=float,
		default=40,
		help="Manual webcam exposure value (default: -4.0; backend-dependent)",
	)
	args = parser.parse_args()

	if not args.webcam and args.image is None:
		parser.error("provide an image path or use --webcam")

	locator = TubeLocator(show_preview=True)

	try:
		if args.webcam:
			backends = {
				"dshow": cv2.CAP_DSHOW,
				"msmf": cv2.CAP_MSMF,
				"default": cv2.CAP_ANY,
			}
			camera = cv2.VideoCapture(args.camera_index, backends[args.backend])
			camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			if not camera.isOpened():
				raise RuntimeError(
					f"Could not open webcam with index {args.camera_index}"
				)
			auto_exposure_set = camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
			exposure_set = camera.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
			actual_auto_exposure = camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
			actual_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
			print(
				f"Camera backend: {camera.getBackendName()}; "
				f"manual mode set={auto_exposure_set} "
				f"(reported {actual_auto_exposure:g}); "
				f"exposure set={exposure_set} "
				f"(requested {args.exposure:g}, reported {actual_exposure:g})"
			)
			if (
				not auto_exposure_set
				or not exposure_set
				or abs(actual_exposure - args.exposure) > 1e-3
			):
				print(
					"Warning: the backend did not report the requested exposure. "
					"Try --backend msmf or a different --exposure value."
				)

			try:
				while True:
					ok, frame = camera.read()
					if not ok:
						break

					center, tube_present, color = locator.locate(frame, (160, 120, 60, 60))
					print(
						f"Tube present: {tube_present}; center: {center}; color: {color}",
						end="\r",
						flush=True,
					)

					key = cv2.waitKey(1) & 0xFF
					if key in (27, ord("q")):
						break
			finally:
				camera.release()
		else:
			frame = cv2.imread(args.image)
			if frame is None:
				raise FileNotFoundError(f"Could not read image: {args.image}")

			frame = resize_to_480p(frame)
			center, tube_present, color = locator.locate(frame, (160, 120, 75, 75))
			print(f"Tube present: {tube_present}; center: {center}; color: {color}")
			while True:
				key = cv2.waitKey(50) & 0xFF
				if key in (27, ord("q")):
					break

				try:
					mask_window_open = cv2.getWindowProperty(
						locator.preview_window,
						cv2.WND_PROP_VISIBLE,
					) >= 0
					frame_window_open = cv2.getWindowProperty(
						f"{locator.preview_window} - frame",
						cv2.WND_PROP_VISIBLE,
					) >= 0
				except cv2.error:
					break

				if not mask_window_open or not frame_window_open:
					break
	finally:
		locator.close()


if __name__ == "__main__":
	main()
