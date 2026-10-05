"""HSV-based tube localisation inside a moving region of interest."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]


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
		lower_hsv: Tuple[int, int, int] = (110, 16, 20),
		upper_hsv: Tuple[int, int, int] = (120, 255, 255),
		invert_mask: bool = True,
		min_contour_area: float = 25.0,
		show_preview: bool = True,
		preview_window: str = "Tube locator",
		roi_rotation_deg: float = 45.0,
		presence_scan_fraction: float = 0.3,
		presence_threshold: float = 0.25,
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

	def locate(self, frame: np.ndarray, roi: ROI) -> Tuple[Optional[Point], bool]:
		"""Return the detected center and whether a tube is present.

		Tube presence is gated by the fraction of foreground pixels in a central
		patch of the inverted ROI mask.
		"""
		if frame is None or frame.size == 0:
			return None, False

		height, width = frame.shape[:2]
		center_x, center_y, roi_width, roi_height = roi
		if roi_width <= 1 or roi_height <= 1:
			return None, False

		half_width = (roi_width - 1) / 2.0
		half_height = (roi_height - 1) / 2.0
		rotation = cv2.getRotationMatrix2D(
			(center_x, center_y),
			self.roi_rotation_deg,
			1.0,
		)
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
		valid_pixels = cv2.warpPerspective(
			np.full((height, width), 255, dtype=np.uint8),
			warp,
			(roi_width, roi_height),
			flags=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
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

		if self.show_preview:
			preview = self.annotate(frame, roi, center, tube_present)
			cv2.imshow(
				self.preview_window,
				np.hstack((roi_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))),
			)
			cv2.imshow(f"{self.preview_window} - frame", preview)
			cv2.waitKey(1)

		return center, tube_present

	def annotate(
		self,
		frame: np.ndarray,
		roi: ROI,
		center: Optional[Point],
		tube_present: bool,
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
		cv2.drawMarker(preview, (center_x, center_y), (0, 220, 255), cv2.MARKER_CROSS, 16, 2)
		if center is not None:
			cv2.drawMarker(preview, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
		status = "Tube present" if tube_present else "No tube present"
		if center is not None:
			status += f"  center ({center[0]}, {center[1]})"
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
		default=-4.0,
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
			camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
			camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
			if not camera.isOpened():
				raise RuntimeError(
					f"Could not open webcam with index {args.camera_index}"
				)
			auto_exposure_set = camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
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

					center, tube_present = locator.locate(frame, (160, 120, 100, 100))
					print(
						f"Tube present: {tube_present}; center: {center}",
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

			frame = resize_to_240p(frame)
			center, tube_present = locator.locate(frame, (160, 120, 85, 85))
			print(f"Tube present: {tube_present}; center: {center}")
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
