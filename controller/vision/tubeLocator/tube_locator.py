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
HOUGH_GRADIENT = "gradient"
HOUGH_GRADIENT_ALT = "gradient_alt"
# Weight of the ROI-center proximity bonus relative to circumference support.
# Support is scored in [0, 1], so 0.5 lets a centrally located circle beat a
# stronger edge elsewhere while still preferring a well-supported ring when two
# candidates are similarly close to the center.
CIRCLE_PROXIMITY_BONUS = 0.9


@dataclass(frozen=True)
class ColorRange:
    """HSV range for classifying a tube cap color.

    Hue uses degrees from 0 to 360, while saturation and value use percentages
    from 0 to 100. Hue wraps when ``lower`` exceeds ``upper``; this represents red
    crossing the hue boundary. The values are scaled to OpenCV's HSV ranges only
    when matching a sample.
    """

    name: str
    lower: Tuple[float, float, float]
    upper: Tuple[float, float, float]
    priority: int = 1
    relative_value: bool = False


MIN_COLOR_SAMPLES = 8
COLOR_NAMES = ("gray", "white", "red", "purple", "green", "light teal")

DEFAULT_COLOR_RANGES: tuple[ColorRange, ...] = (
    # White caps are pale ivory, and the annulus average wanders between ~30°
    # and ~60° hue as the cap's centre shifts by a pixel or two. Keep a wide
    # warm-hue band; the white/gray tie-break below stops cream-grays becoming
    # white.
    ColorRange("white", (0, 5, 40), (70, 40, 100), priority=1, relative_value=True),
    # Red caps sit under a strong shadow/highlight gradient, so the annulus
    # average is much dimmer than a clean swatch. Keep the value floor low and
    # put the saturation floor above gray's range so dark gray caps cannot be
    # pulled into red.
    ColorRange("red", (0, 45, 40), (20, 100, 100), priority=2),
    ColorRange("purple", (300, 25, 50), (360, 100, 100), priority=2),
    # Dark teal caps measure 26-51% value and can climb above the old 55%
    # ceiling in brighter light, which dropped them out of green.
    ColorRange("green", (160, 67, 15), (200, 100, 75), priority=2),
    # Pale mint caps measure 146-164 deg hue with edge pixels near 175 deg; the
    # old 170 deg/35% ceiling made them flicker out of light teal.
    ColorRange("light teal", (105, 5, 50), (190, 45, 100), priority=3),
    ColorRange("gray", (0, 0, 0), (360, 40, 70), priority=1),
)

CALIBRATION_PATH = Path(__file__).resolve().with_name("grid_mapping.json")

with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
    calibration = json.load(file)

pixel_to_world = np.array(
    calibration["pixel_to_world_homography"],
    dtype=np.float64,
)


# Gantry Z at which grid_mapping.json was captured; pixel_to_real's
# camera_raise_mm is measured from here.
CALIBRATION_GANTRY_Z_MM = float(calibration.get("gantry_z_mm", 50.0))
# Whole-frame cap radius limits (pixels). Caps get bigger as the camera
# descends, so these cover both the survey and the close-up height.
FULL_FRAME_MIN_RADIUS_PX = 13
FULL_FRAME_MAX_RADIUS_PX = 90


@dataclass(frozen=True)
class CapCircle:
    """One cap circle found in a whole frame, in frame pixels."""

    x: float
    y: float
    radius: float
    color: Optional[str]


world_to_pixel = np.linalg.inv(pixel_to_world)
world_to_pixel /= world_to_pixel[2, 2]
INTRINSICS_PATH = Path(__file__).resolve().with_name("camera_calibration.npz")


@dataclass(frozen=True)
class RackCameraPose:
    """Camera pose recovered from the rack homography plus lens intrinsics.

    ``height_column`` is what one millimetre of height above the rack adds to
    the third column of ``world_to_pixel``; ``camera_height_mm`` is the camera's
    distance above the rack top at the gantry Z used for grid_mapping.json.
    """

    height_column: np.ndarray
    camera_height_mm: float
    camera_xy_mm: tuple[float, float]
    tilt_deg: float
    axis_scale_ratio: float


def _decompose_rack_pose(homography: np.ndarray, camera_matrix: np.ndarray) -> RackCameraPose:
    """Split a world(mm)->pixel plane homography into K[r1 r2 t] and return the pose."""
    columns = np.linalg.inv(camera_matrix) @ homography
    norm_1 = float(np.linalg.norm(columns[:, 0]))
    norm_2 = float(np.linalg.norm(columns[:, 1]))
    scale = 2.0 / (norm_1 + norm_2)
    if columns[2, 2] * scale < 0:
        scale = -scale  # The rack must be in front of the camera.
    r1, r2, translation = (columns * scale).T
    # Snap r1/r2 to the nearest true rotation before taking the normal.
    u, _, vt = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
    rotation = u @ vt
    camera_center = -rotation.T @ translation
    # +Z points from the rack toward the camera.
    up = rotation[:, 2] if camera_center[2] > 0 else -rotation[:, 2]
    return RackCameraPose(
        height_column=(camera_matrix @ up) / scale,
        camera_height_mm=float(abs(camera_center[2])),
        camera_xy_mm=(float(camera_center[0]), float(camera_center[1])),
        tilt_deg=float(np.degrees(np.arccos(min(1.0, abs(rotation[2, 2]))))),
        axis_scale_ratio=norm_1 / norm_2,
    )


_rack_pose: Optional[RackCameraPose] = None


def rack_camera_pose() -> RackCameraPose:
    """Load (once) the camera pose implied by grid_mapping.json and the intrinsics."""
    global _rack_pose
    if _rack_pose is None:
        with np.load(INTRINSICS_PATH) as intrinsics:
            camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
        _rack_pose = _decompose_rack_pose(world_to_pixel, camera_matrix)
    return _rack_pose


def _world_to_pixel_at(height_mm: float) -> np.ndarray:
    """Homography for the horizontal plane ``height_mm`` above the rack top."""
    if height_mm == 0.0:
        return world_to_pixel
    if not np.isfinite(height_mm):
        raise ValueError("height_mm must be finite")
    pose = rack_camera_pose()
    if height_mm >= pose.camera_height_mm:
        raise ValueError(
            f"height_mm={height_mm:.1f} is at or above the camera "
            f"({pose.camera_height_mm:.1f} mm above the rack)"
        )
    matrix = world_to_pixel.copy()
    matrix[:, 2] += height_mm * pose.height_column
    return matrix


def pixel_to_real(
    pixel_x: float,
    pixel_y: float,
    height_mm: float = 0.0,
    camera_raise_mm: float = 0.0,
) -> tuple[float, float]:
    """Map an image pixel to rack-frame XY in mm.

    ``height_mm`` is how far the imaged point (e.g. a cap top) sits above the
    rack top. ``camera_raise_mm`` is how far the gantry Z is above the Z used
    when grid_mapping.json was calibrated. With both at 0 this is the plain
    rack-plane homography.
    """
    matrix = np.linalg.inv(_world_to_pixel_at(height_mm - camera_raise_mm))
    pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float64)
    real = cv2.perspectiveTransform(pixel, matrix)[0, 0]
    return float(real[0]), float(real[1])


def real_to_pixel(
    real_x: float,
    real_y: float,
    height_mm: float = 0.0,
    camera_raise_mm: float = 0.0,
) -> tuple[float, float]:
    """Inverse of :func:`pixel_to_real`: where a rack-frame point appears in the image."""
    matrix = _world_to_pixel_at(height_mm - camera_raise_mm)
    point = np.array([[[real_x, real_y]]], dtype=np.float64)
    pixel = cv2.perspectiveTransform(point, matrix)[0, 0]
    return float(pixel[0]), float(pixel[1])


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
		circle_min_distance: float = 20.0,
		circle_edge_threshold: float = 34.0,
		circle_canny_threshold: float = 130.0,
		circle_method: str = HOUGH_GRADIENT_ALT,
	) -> None:
		if not 0.0 < presence_scan_fraction <= 1.0:
			raise ValueError("presence_scan_fraction must be in the range (0, 1]")
		if not 0.0 <= presence_threshold <= 1.0:
			raise ValueError("presence_threshold must be in the range [0, 1]")
		if circle_min_distance <= 0:
			raise ValueError("circle_min_distance must be greater than zero")
		if circle_edge_threshold <= 0:
			raise ValueError("circle_edge_threshold must be greater than zero")
		if circle_canny_threshold <= 0:
			raise ValueError("circle_canny_threshold must be greater than zero")
		if circle_method not in {HOUGH_GRADIENT, HOUGH_GRADIENT_ALT}:
			raise ValueError("circle_method must be 'gradient' or 'gradient_alt'")

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
		self.circle_min_distance = circle_min_distance
		self.circle_edge_threshold = circle_edge_threshold
		self.circle_canny_threshold = circle_canny_threshold
		self.circle_method = circle_method
		self.last_color_stats: Optional[dict[str, float | int]] = None
		self.last_color_radius: Optional[int] = None

	def _classify_color(
		self,
		roi_image: np.ndarray,
		center: Point,
		radius: int = 8,
		samples_mask: Optional[np.ndarray] = None,
		num_samples: int = 24,
		min_agreement: float = 0.6,
	) -> Optional[str]:
		"""Classify the average color in an annulus centered on the tube.

	 The center of a tube cap is commonly a dark hole, so the annulus is fixed at
	 half the detected radius instead of sampling the center or arbitrary points.
	 The optional mask excludes the plate and contour edges before averaging.
	 """
		height, width = roi_image.shape[:2]
		if roi_image.ndim != 3 or roi_image.shape[2] != 3 or min(height, width) < 3:
			return None
		if radius <= 0:
			return None

		cx, cy = center
		if not (0 <= cx < width and 0 <= cy < height):
			return None
		ring_radius = max(1, int(round(radius * 0.5)))
		ring_thickness = max(1, int(round(radius * 0.15)))
		outer = np.zeros((height, width), dtype=np.uint8)
		inner = np.zeros_like(outer)
		cv2.circle(outer, (cx, cy), ring_radius + ring_thickness, 255, -1)
		cv2.circle(inner, (cx, cy), max(0, ring_radius - ring_thickness), 255, -1)
		ring = cv2.subtract(outer, inner).astype(bool)
		if samples_mask is not None:
			if samples_mask.shape[:2] != (height, width):
				return None
			ring &= samples_mask.astype(bool)
		if int(ring.sum()) < MIN_COLOR_SAMPLES:
			return None

		average_bgr = roi_image[ring].mean(axis=0).astype(np.uint8)
		ring_hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)[ring]
		self.last_color_stats = {
			"sample_count": int(len(ring_hsv)),
			"hue_min": float(np.min(ring_hsv[:, 0]) * 2.0),
			"hue_max": float(np.max(ring_hsv[:, 0]) * 2.0),
			"saturation_min": float(np.min(ring_hsv[:, 1]) / 255.0 * 100.0),
			"saturation_max": float(np.max(ring_hsv[:, 1]) / 255.0 * 100.0),
			"value_min": float(np.min(ring_hsv[:, 2]) / 255.0 * 100.0),
			"value_max": float(np.max(ring_hsv[:, 2]) / 255.0 * 100.0),
		}
		average_hsv = cv2.cvtColor(average_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
		hue, saturation, value = (int(component) for component in average_hsv)

		background_value = 0
		if samples_mask is not None:
			background = ~samples_mask.astype(bool)
			if int(background.sum()):
				background_hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
				background_value = int(background_hsv[:, :, 2][background].mean())

		sample = (hue, saturation, value)
		matching_ranges: list[tuple[int, str]] = []
		for color_range in self.color_ranges:
			lower = self._scale_hsv(color_range.lower)
			upper = self._scale_hsv(color_range.upper)
			value_floor = lower[2]
			if color_range.relative_value and background_value:
				value_floor = min(value_floor, int(background_value * 1.35))
			if self._color_range_matches(color_range, *sample, value_floor=value_floor):
				matching_ranges.append((color_range.priority, color_range.name))

		if not matching_ranges:
			return None
		# White and gray overlap in the low-saturation band. Only let white win
		# when it is clearly brighter than the rack; otherwise a gray cap is
		# reported as white.
		names = {name for _, name in matching_ranges}
		if "white" in names and "gray" in names and background_value:
			if value <= int(background_value * 1.25):
				matching_ranges = [
					(priority, name)
					for priority, name in matching_ranges
					if name != "white"
				]
		return max(matching_ranges)[1]

	@staticmethod
	def _scale_hsv(
		hsv_percentage: Tuple[float, float, float],
	) -> tuple[int, int, int]:
		"""Scale degree/percentage HSV values to OpenCV's (180, 255, 255) ranges."""
		hue, saturation, value = hsv_percentage
		if not 0 <= hue <= 360 or not 0 <= saturation <= 100 or not 0 <= value <= 100:
			raise ValueError("HSV hue must be 0-360; saturation and value must be 0-100")
		return (
			round(hue * 180 / 360),
			round(saturation * 255 / 100),
			round(value * 255 / 100),
		)

	@staticmethod
	def _color_range_matches(
		color_range: ColorRange,
		hue: int,
		saturation: int,
		value: int,
		value_floor: int,
	) -> bool:
		"""Return whether one HSV sample falls inside a range, honouring hue wrap."""
		lower_hue, lower_saturation, _ = TubeLocator._scale_hsv(color_range.lower)
		upper_hue, upper_saturation, upper_value = TubeLocator._scale_hsv(color_range.upper)
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
		lower_hue, _, _ = TubeLocator._scale_hsv(color_range.lower)
		upper_hue, _, _ = TubeLocator._scale_hsv(color_range.upper)
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
		self.last_color_stats = None
		self.last_color_radius = None
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
					self.last_color_radius = int(round(enclosing_radius))
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
		self.last_color_stats = None
		self.last_color_radius = None
		if frame is None or frame.size == 0:
			return None, False, None, None, [], None

		center_x, center_y, roi_width, roi_height = roi
		if roi_width <= 1 or roi_height <= 1:
			return None, False, None, None, [], None

		roi_image, warp = self._warp_roi(frame, roi)
		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		mask = raw_mask.copy()
		if self.invert_mask:
			mask = cv2.bitwise_not(mask)
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
		denoised = cv2.medianBlur(gray, 5)
		blurred = cv2.GaussianBlur(denoised, (9, 9), 1)
		hough_input = cv2.bitwise_and(blurred, blurred, mask=mask)
		min_radius = max(2, min(roi_width, roi_height) // 12)
		max_radius = max(min_radius, min(roi_width, roi_height) // 2)
		method = (
			cv2.HOUGH_GRADIENT_ALT
			if self.circle_method == HOUGH_GRADIENT_ALT
			else cv2.HOUGH_GRADIENT
		)
		param2 = 0.90 if self.circle_method == HOUGH_GRADIENT_ALT else self.circle_edge_threshold
		detected = cv2.HoughCircles(
			hough_input,
			method,
			dp=1.5 if self.circle_method == HOUGH_GRADIENT_ALT else 1,
			minDist=self.circle_min_distance,
			param1=self.circle_canny_threshold,
			param2=param2,
			minRadius=min_radius,
			maxRadius=max_radius,
		)
		circles: list[Circle] = []
		roi_circles: list[Circle] = []
		if detected is not None:
			detected_circles = [
				(tuple(int(value) for value in circle))
				for circle in np.rint(detected[0]).astype(int)
			]
			for x, y, radius in self._suppress_inner_circles(detected_circles):
				roi_circles.append((int(x), int(y), int(radius)))
				roi_circle = np.array([[[float(x), float(y)]]], dtype=np.float32)
				frame_circle = cv2.perspectiveTransform(roi_circle, np.linalg.inv(warp))[0, 0]
				circles.append((
					int(round(float(frame_circle[0]))),
					int(round(float(frame_circle[1]))),
					int(radius),
				))
		if not circles:
			fallback = self._circle_from_mask(mask, min_radius, max_radius)
			if fallback is not None:
				x, y, radius = fallback
				roi_circles.append(fallback)
				roi_circle = np.array([[[float(x), float(y)]]], dtype=np.float32)
				frame_circle = cv2.perspectiveTransform(roi_circle, np.linalg.inv(warp))[0, 0]
				circles.append((
					int(round(float(frame_circle[0]))),
					int(round(float(frame_circle[1]))),
					radius,
				))

		center: Optional[Point] = None
		color: Optional[str] = None
		if circles:
			best_index = max(
				range(len(circles)),
				key=lambda i: self._circle_priority(
					blurred,
					roi_circles[i],
					roi_width,
					roi_height,
				),
			)
			center = (circles[best_index][0], circles[best_index][1])
			roi_circle = roi_circles[best_index]
			self.last_color_radius = roi_circle[2]
			color = self._classify_color(roi_image, (roi_circle[0], roi_circle[1]), radius=roi_circle[2])
		debug_montage = self._circle_debug_montage(roi_image, gray, hough_input, roi_circles)
		return center, bool(circles), color, raw_mask, circles, debug_montage

	def detect_all_circles(
		self,
		frame: np.ndarray,
		min_radius: int = FULL_FRAME_MIN_RADIUS_PX,
		max_radius: int = FULL_FRAME_MAX_RADIUS_PX,
	) -> tuple[list[CapCircle], Optional[np.ndarray], Optional[np.ndarray]]:
		"""Detect every cap circle in the whole frame (no ROI) and classify each one's color.

		Returns the circles, the raw HSV threshold mask, and a debug montage.
		``last_color_stats``/``last_color_radius`` describe the circle nearest the
		frame center so color-range recording keeps working.
		"""
		self.last_color_stats = None
		self.last_color_radius = None
		if frame is None or frame.size == 0:
			return [], None, None
		height, width = frame.shape[:2]
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		raw_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		mask = cv2.bitwise_not(raw_mask) if self.invert_mask else raw_mask.copy()
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (9, 9), 1)
		hough_input = cv2.bitwise_and(blurred, blurred, mask=mask)
		alt = self.circle_method == HOUGH_GRADIENT_ALT
		detected = cv2.HoughCircles(
			hough_input,
			cv2.HOUGH_GRADIENT_ALT if alt else cv2.HOUGH_GRADIENT,
			dp=1.5 if alt else 1,
			minDist=self.circle_min_distance,
			param1=self.circle_canny_threshold,
			param2=0.90 if alt else self.circle_edge_threshold,
			minRadius=min_radius,
			maxRadius=max_radius,
		)
		raw_circles: list[Circle] = []
		if detected is not None:
			raw_circles = self._suppress_inner_circles([
				tuple(int(value) for value in circle)
				for circle in np.rint(detected[0]).astype(int)
			])
		edges = cv2.Canny(hough_input, self.circle_canny_threshold / 2, self.circle_canny_threshold)
		circles = []
		for x, y, radius in raw_circles:
			fx, fy, fr = self._refine_circle(edges, x, y, radius)
			circles.append(CapCircle(
				x=fx,
				y=fy,
				radius=fr,
				color=self._classify_color(frame, (x, y), radius=radius, samples_mask=mask),
			))
		circles = self._drop_inner_holes(circles)
		if raw_circles:
			nearest = min(raw_circles, key=lambda c: (c[0] - width / 2) ** 2 + (c[1] - height / 2) ** 2)
			self._classify_color(frame, nearest[:2], radius=nearest[2], samples_mask=mask)
			self.last_color_radius = nearest[2]
		scale = 0.25
		small = lambda image: cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
		debug_montage = self._circle_debug_montage(
			small(frame),
			small(gray),
			small(hough_input),
			[(int(c.x * scale), int(c.y * scale), max(1, int(c.radius * scale))) for c in circles],
		)
		return circles, raw_mask, debug_montage

	@staticmethod
	def _refine_circle(edges: np.ndarray, x: int, y: int, radius: int) -> tuple[float, float, float]:
		"""Least-squares circle fit to rim edge pixels near a Hough detection.

		Hough centers are quantised by the accumulator; fitting the edge pixels in
		an annulus around the detected rim recovers sub-pixel centers. Points from
		touching neighbours are rejected as outliers; if the fit drifts, the Hough
		circle is kept.
		"""
		height, width = edges.shape[:2]
		pad = int(radius * 1.3) + 2
		x0, x1 = max(0, x - pad), min(width, x + pad + 1)
		y0, y1 = max(0, y - pad), min(height, y + pad + 1)
		ys, xs = np.nonzero(edges[y0:y1, x0:x1])
		xs = xs.astype(np.float64) + x0
		ys = ys.astype(np.float64) + y0
		distance = np.hypot(xs - x, ys - y)
		keep = (distance > radius * 0.75) & (distance < radius * 1.25)
		xs, ys = xs[keep], ys[keep]
		fallback = (float(x), float(y), float(radius))
		for _ in range(3):
			if len(xs) < 12:
				return fallback
			matrix = np.column_stack((xs, ys, np.ones_like(xs)))
			rhs = xs * xs + ys * ys
			(a, b, c), *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
			cx, cy = a / 2.0, b / 2.0
			fit_radius = float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))
			residual = np.abs(np.hypot(xs - cx, ys - cy) - fit_radius)
			inliers = residual < 2.0
			if inliers.all():
				break
			xs, ys = xs[inliers], ys[inliers]
		if (
			not 0.8 * radius <= fit_radius <= 1.2 * radius
			or np.hypot(cx - x, cy - y) > 0.3 * radius
		):
			return fallback
		return float(cx), float(cy), fit_radius

	def annotate_circles(
		self,
		frame: np.ndarray,
		circles: list[CapCircle],
		highlight: Optional[Point] = None,
	) -> np.ndarray:
		"""Draw every detected cap with its color label; mark ``highlight`` in red."""
		preview = frame.copy()
		for circle in circles:
			center = (int(round(circle.x)), int(round(circle.y)))
			cv2.circle(preview, center, int(round(circle.radius)), (0, 255, 0), 2)
			cv2.circle(preview, center, 2, (0, 255, 0), -1)
			cv2.putText(
				preview,
				circle.color or "?",
				(center[0] + 6, center[1] - 6),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(0, 0, 0),
				3,
				cv2.LINE_AA,
			)
			cv2.putText(
				preview,
				circle.color or "?",
				(center[0] + 6, center[1] - 6),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.45,
				(255, 255, 255),
				1,
				cv2.LINE_AA,
			)
		if highlight is not None:
			cv2.drawMarker(preview, highlight, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
		cv2.putText(
			preview,
			f"{len(circles)} cap(s) detected",
			(10, 25),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 255, 0) if circles else (0, 0, 255),
			2,
		)
		return preview

	@staticmethod
	def _circle_from_mask(
		mask: np.ndarray,
		min_radius: int,
		max_radius: int,
	) -> Optional[Circle]:
		"""Recover a cap circle from its segmented contour when Hough rejects it."""
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return None
		contour = max(contours, key=cv2.contourArea)
		if cv2.contourArea(contour) < np.pi * min_radius**2 * 0.25:
			return None
		(center_x, center_y), radius = cv2.minEnclosingCircle(contour)
		radius = int(round(radius))
		if not min_radius <= radius <= max_radius:
			return None
		return int(round(center_x)), int(round(center_y)), radius

	@staticmethod
	def _suppress_inner_circles(circles: list[Circle]) -> list[Circle]:
		"""Keep the larger circle when detections are nearly concentric."""
		kept: list[Circle] = []
		for candidate in sorted(circles, key=lambda circle: circle[2], reverse=True):
			cx, cy, radius = candidate
			is_inner_detection = any(
				(cx - kept_x) ** 2 + (cy - kept_y) ** 2
				<= (kept_radius * 0.6) ** 2
				for kept_x, kept_y, kept_radius in kept
			)
			if not is_inner_detection:
				kept.append(candidate)
		return kept

	@staticmethod
	def _drop_inner_holes(
		circles: list[CapCircle],
		min_reference_radius: float = 22.0,
		fraction: float = 0.55,
	) -> list[CapCircle]:
		"""Drop small detections that sit well below the frame's typical cap size.

		When a cap's outer rim is missed, Hough can report the dark inner hole as
		its own circle. Holes are roughly 40% of the cap radius, so use the
		frame's 80th-percentile radius as the reference. Below
		``min_reference_radius`` every cap is small (high survey), so radius is
		not diagnostic and nothing is dropped. White caps are exempt because
		their short tubes genuinely project smaller than the tall colours.
		"""
		if len(circles) < 2:
			return circles
		reference = float(np.percentile([circle.radius for circle in circles], 80))
		if reference < min_reference_radius:
			return circles
		limit = fraction * reference
		return [
			circle
			for circle in circles
			if circle.radius >= limit or (circle.color or "").lower() == "white"
		]

	@classmethod
	def _circle_priority(
		cls,
		image: np.ndarray,
		circle: Circle,
		roi_width: int,
		roi_height: int,
	) -> float:
		"""Rank a candidate by edge support plus a bonus for ROI-center proximity."""
		support = cls._circle_score(image, circle)
		cx, cy, _ = circle
		half_width = (roi_width - 1) / 2.0
		half_height = (roi_height - 1) / 2.0
		max_distance = max(1.0, float(np.hypot(half_width, half_height)))
		distance = float(np.hypot(cx - half_width, cy - half_height))
		proximity = 1.0 - min(distance / max_distance, 1.0)
		return support + CIRCLE_PROXIMITY_BONUS * proximity

	@staticmethod
	def _circle_score(image: np.ndarray, circle: Circle) -> float:
		"""Score how strongly an image supports a circle's circumference."""
		height, width = image.shape[:2]
		cx, cy, radius = circle
		angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
		support = 0
		samples = 0
		for angle in angles:
			x = int(round(cx + radius * np.cos(angle)))
			y = int(round(cy + radius * np.sin(angle)))
			if not (1 <= x < width - 1 and 1 <= y < height - 1):
				continue
			gradient_x = int(image[y, x + 1]) - int(image[y, x - 1])
			gradient_y = int(image[y + 1, x]) - int(image[y - 1, x])
			if gradient_x * gradient_x + gradient_y * gradient_y >= 35 * 35:
				support += 1
			samples += 1
		return support / samples if samples else 0.0

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
		color_radius: Optional[int] = None,
	) -> np.ndarray:
		"""Draw the rotated ROI, detected center, and color-sampling ring."""
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
			if color_radius is None and circles:
				color_radius = min(
					circles,
					key=lambda circle: (circle[0] - center[0]) ** 2 + (circle[1] - center[1]) ** 2,
				)[2]
			if color_radius is not None and color_radius > 0:
				ring_radius = max(1, int(round(color_radius * 0.5)))
				for start_angle in range(0, 360, 20):
					cv2.ellipse(
						preview,
						center,
						(ring_radius, ring_radius),
						0,
						start_angle,
						start_angle + 10,
						(0, 255, 255),
						1,
					)
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
