"""HSV-based tube localisation inside a moving region of interest."""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
ROI = Tuple[int, int, int, int]


def crop_to_480p(image: np.ndarray) -> np.ndarray:
	"""Center-crop and resize an image to 640x480."""
	target_width, target_height = 640, 480
	height, width = image.shape[:2]

	target_ratio = target_width / target_height
	image_ratio = width / height

	if image_ratio > target_ratio:
		crop_width = int(height * target_ratio)
		offset = (width - crop_width) // 2
		image = image[:, offset:offset + crop_width]
	else:
		crop_height = int(width / target_ratio)
		offset = (height - crop_height) // 2
		image = image[offset:offset + crop_height, :]

	return cv2.resize(
		image,
		(target_width, target_height),
		interpolation=cv2.INTER_AREA,
	)


class TubeLocator:
	"""Locate the centre of the largest thresholded contour in a dynamic ROI.

	``roi`` is supplied for every frame as ``(x, y, width, height)``.  The
	returned point is in full-frame coordinates, rather than ROI coordinates.
	"""

	def __init__(
		self,
		lower_hsv: Tuple[int, int, int] = (110, 16, 20),
		upper_hsv: Tuple[int, int, int] = (120, 255, 255),
		invert_mask: bool = True,
		min_contour_area: float = 25.0,
		show_preview: bool = True,
		preview_window: str = "Tube locator",
	) -> None:
		self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
		self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
		self.invert_mask = invert_mask
		self.min_contour_area = min_contour_area
		self.show_preview = show_preview
		self.preview_window = preview_window

	def locate(self, frame: np.ndarray, roi: ROI) -> Optional[Point]:
		"""Process one frame and return ``(x, y)`` or ``None`` if not found."""
		if frame is None or frame.size == 0:
			return None

		height, width = frame.shape[:2]
		x, y, roi_width, roi_height = roi
		x0, y0 = max(0, x), max(0, y)
		x1 = min(width, x + max(0, roi_width))
		y1 = min(height, y + max(0, roi_height))
		if x0 >= x1 or y0 >= y1:
			return None

		roi_image = frame[y0:y1, x0:x1]
		hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
		if self.invert_mask:
			mask = cv2.bitwise_not(mask)
		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		valid = [c for c in contours if cv2.contourArea(c) >= self.min_contour_area]
		center: Optional[Point] = None
		if valid:
			contour = max(valid, key=cv2.contourArea)
			moments = cv2.moments(contour)
			if moments["m00"]:
				center = (
					x0 + int(moments["m10"] / moments["m00"]),
					y0 + int(moments["m01"] / moments["m00"]),
				)

		if self.show_preview:
			preview = frame.copy()
			cv2.rectangle(preview, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
			if center is not None:
				cv2.drawMarker(preview, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
			cv2.imshow(self.preview_window, np.hstack((roi_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))))
			cv2.imshow(f"{self.preview_window} - frame", preview)
			cv2.waitKey(1)

		return center

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
	args = parser.parse_args()

	if not args.webcam and args.image is None:
		parser.error("provide an image path or use --webcam")

	locator = TubeLocator(show_preview=True)

	try:
		if args.webcam:
			camera = cv2.VideoCapture(args.camera_index)
			if not camera.isOpened():
				raise RuntimeError(
					f"Could not open webcam with index {args.camera_index}"
				)

			try:
				while True:
					ok, frame = camera.read()
					if not ok:
						break

					frame = crop_to_480p(frame)
					center = locator.locate(frame, (140, 125, 140, 140))
					print(f"Detected center: {center}", end="\r", flush=True)

					key = cv2.waitKey(1) & 0xFF
					if key in (27, ord("q")):
						break
			finally:
				camera.release()
		else:
			frame = cv2.imread(args.image)
			if frame is None:
				raise FileNotFoundError(f"Could not read image: {args.image}")

			frame = crop_to_480p(frame)
			center = locator.locate(frame, (140, 125, 140, 140))
			print(f"Detected center: {center}")
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
