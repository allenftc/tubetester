from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from controller.config.settings import CameraSettings
from controller.vision.tubeLocator.cap_matching import MergedCircle, merge_frame_detections
from controller.vision.tubeLocator.tube_locator import HOUGH_GRADIENT, CapCircle, TubeLocator


PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480
PHOTO_DIRECTORY = Path(__file__).resolve().parents[2]


class CameraPreviewService:
    """Own the camera and publish JPEG frames with the live locator overlay."""

    def __init__(self, settings: CameraSettings, *, fps: int = 15, jpeg_quality: int = 82) -> None:
        self.settings = settings
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._lifecycle_lock = threading.Lock()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._locator_generation = 0
        self._reopen_requested = False
        self._photo_lock = threading.Lock()
        self._photo_requested = False
        self._photo_directory = PHOTO_DIRECTORY
        self._photo_result: dict[str, Any] | None = None
        self._photo_error: str | None = None
        self._sequence = 0
        self._jpeg: bytes | None = None
        self._mask_jpeg: bytes | None = None
        self._debug_jpeg: bytes | None = None
        self._detection_mode = "hough"
        self._circle_min_distance = 20.0
        self._circle_edge_threshold = 34.0
        self._circle_canny_threshold = 130.0
        self._circle_method = "gradient_alt"
        self._record_color_range = False
        self._recorded_color_range: dict[str, float | int] | None = None
        self._status: dict[str, Any] = {
            "available": False,
            "state": "not_started",
            "message": "Camera preview has not started.",
            "device": self._device_name(),
            "width": PREVIEW_WIDTH,
            "height": PREVIEW_HEIGHT,
            "roi": self._configured_roi(),
            "detected_center": None,
            "detected_offset_roi": None,
            "tube_present": False,
            "detected_color": None,
            "detected_circles": [],
            "detection_mode": self._detection_mode,
            "circle_min_distance": self._circle_min_distance,
            "circle_edge_threshold": self._circle_edge_threshold,
            "circle_canny_threshold": self._circle_canny_threshold,
            "circle_method": self._circle_method,
            "sequence": 0,
            "color_range_recording": False,
            "color_range": None,
        }

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                if not self._stop_event.is_set():
                    return
                self._thread.join(timeout=3)
                if self._thread.is_alive():
                    raise RuntimeError("The previous camera preview is still stopping.")
            self._stop_event.clear()
            with self._condition:
                self._sequence = 0
                self._jpeg = None
                self._mask_jpeg = None
                self._debug_jpeg = None
                self._photo_requested = False
                self._photo_result = None
                self._photo_error = None
            self._set_status(state="starting", message=f"Opening {self._device_name()}.")
            self._thread = threading.Thread(target=self._capture_loop, name="camera-preview", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop_event.set()
            with self._condition:
                capture = self._capture
                self._condition.notify_all()
            if capture is not None:
                capture.release()
            if self._thread and self._thread is not threading.current_thread():
                self._thread.join(timeout=3)
            if not self._thread or not self._thread.is_alive():
                with self._condition:
                    self._thread = None
                    self._jpeg = None
                    self._mask_jpeg = None
                    self._debug_jpeg = None
                self._set_status(available=False, state="stopped", message="Camera preview is off.")
            else:
                self._set_status(available=False, state="stopping", message="Stopping camera preview.")

    def apply_settings(self, settings: CameraSettings) -> None:
        """Swap camera settings and refresh the capture thread in place."""
        with self._condition:
            previous = self.settings
            self.settings = settings
            self._locator_generation += 1
            if (
                previous.device_index != settings.device_index
                or previous.resolution != settings.resolution
            ):
                self._reopen_requested = True
            self._status["device"] = self._device_name()
            self._status["roi"] = self._configured_roi()
            self._condition.notify_all()

    def set_color_range_recording(self, recording: bool) -> None:
        with self._condition:
            self._record_color_range = recording
            if recording:
                self._recorded_color_range = None
            self._status["color_range_recording"] = recording
            self._status["color_range"] = self._recorded_color_range
            self._condition.notify_all()

    def reset_color_range_recording(self) -> None:
        with self._condition:
            self._recorded_color_range = None
            self._status["color_range"] = None
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return {
                **self._status,
                "roi": dict(self._status["roi"]),
                "detected_center": list(self._status["detected_center"])
                if self._status["detected_center"] is not None
                else None,
                "detected_offset_roi": list(self._status["detected_offset_roi"])
                if self._status["detected_offset_roi"] is not None
                else None,
                "detected_circles": [dict(c) for c in self._status.get("detected_circles") or []],
            }

    def take_photo(self, *, timeout_seconds: float = 8.0, directory: Path | None = None) -> dict[str, Any]:
        """Save the next raw (unannotated) camera frame as a 640x480 PNG."""
        target_directory = Path(directory) if directory is not None else PHOTO_DIRECTORY
        with self._photo_lock:
            with self._condition:
                if (
                    self._thread is None
                    or not self._thread.is_alive()
                    or self._stop_event.is_set()
                    or self._status["state"] in {"not_started", "stopped", "stopping"}
                ):
                    raise RuntimeError("Turn on the camera preview before taking a picture.")
                self._photo_directory = target_directory
                self._photo_result = None
                self._photo_error = None
                self._photo_requested = True
                self._condition.notify_all()
                self._condition.wait_for(
                    lambda: self._photo_result is not None
                    or self._photo_error is not None
                    or self._stop_event.is_set()
                    or self._status["state"] in {"unavailable", "error", "stopped"},
                    timeout=timeout_seconds,
                )
                result = self._photo_result
                error = self._photo_error
                self._photo_requested = False
                self._photo_result = None
                self._photo_error = None
            if error is not None:
                raise RuntimeError(error)
            if result is None:
                raise RuntimeError("The camera did not provide a frame in time.")
            return result

    def set_detection_mode(self, mode: str) -> None:
        if mode not in {"hsv", "hough"}:
            raise ValueError("detection_mode must be 'hsv' or 'hough'")
        with self._condition:
            self._detection_mode = mode
            self._status["detection_mode"] = mode
            self._condition.notify_all()

    def set_circle_parameters(
        self,
        *,
        min_distance: float,
        edge_threshold: float,
        canny_threshold: float,
        method: str,
    ) -> None:
        if not 1.0 <= min_distance <= 100.0:
            raise ValueError("circle_min_distance must be between 1 and 100")
        if not 1.0 <= edge_threshold <= 100.0:
            raise ValueError("circle_edge_threshold must be between 1 and 100")
        if not 50.0 <= canny_threshold <= 250.0:
            raise ValueError("circle_canny_threshold must be between 50 and 250")
        if method not in {"gradient", "gradient_alt"}:
            raise ValueError("circle_method must be 'gradient' or 'gradient_alt'")
        with self._condition:
            self._circle_min_distance = min_distance
            self._circle_edge_threshold = edge_threshold
            self._circle_canny_threshold = canny_threshold
            self._circle_method = method
            self._status["circle_min_distance"] = min_distance
            self._status["circle_edge_threshold"] = edge_threshold
            self._status["circle_canny_threshold"] = canny_threshold
            self._status["circle_method"] = method
            self._condition.notify_all()

    def wait_for_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        return self._wait_for_image(after_sequence, timeout_seconds, mask=False)

    def wait_for_mask_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        return self._wait_for_image(after_sequence, timeout_seconds, mask=True)

    def wait_for_debug_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        return self._wait_for_image(after_sequence, timeout_seconds, debug=True)

    def _wait_for_image(
        self,
        after_sequence: int,
        timeout_seconds: float,
        *,
        mask: bool = False,
        debug: bool = False,
    ) -> tuple[int, bytes | None]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > after_sequence
                or self._status["state"] in {"unavailable", "retrying", "error", "stopped"}
                or self._stop_event.is_set(),
                timeout=timeout_seconds,
            )
            if self._sequence <= after_sequence:
                return self._sequence, None
            image = self._debug_jpeg if debug else self._mask_jpeg if mask else self._jpeg
            return self._sequence, image

    def wait_for_detection(self, after_sequence: int, timeout_seconds: float = 8.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while True:
                if self._sequence > after_sequence and self._status["detected_center"] is not None:
                    return {
                        **self._status,
                        "roi": dict(self._status["roi"]),
                        "detected_center": list(self._status["detected_center"]),
                    }
                if self._status["state"] in {"unavailable", "error", "stopped"}:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def wait_for_hough_detection(
        self,
        after_sequence: int,
        frame_count: int = 10,
        timeout_seconds: float = 8.0,
        require_color: bool = False,
        color_frame_count: int = 3,
    ) -> dict[str, Any] | None:
        """Average fresh detections and optionally require a stable color."""
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero")
        if color_frame_count <= 0:
            raise ValueError("color_frame_count must be greater than zero")
        deadline = time.monotonic() + timeout_seconds
        center_samples: list[tuple[float, float]] = []
        offset_samples: list[tuple[float, float]] = []
        color_samples: list[str] = []
        sequence = after_sequence
        with self._condition:
            while len(center_samples) < frame_count or (
                require_color and len(color_samples) < color_frame_count
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._status["state"] in {"unavailable", "error", "stopped"}:
                    return None
                self._condition.wait_for(
                    lambda: self._sequence > sequence
                    or self._status["state"] in {"unavailable", "error", "stopped"}
                    or self._stop_event.is_set(),
                    timeout=remaining,
                )
                if self._sequence <= sequence:
                    return None
                sequence = self._sequence
                center = self._status["detected_center"]
                offset = self._status["detected_offset_roi"]
                if center is not None:
                    center_samples.append((float(center[0]), float(center[1])))
                if offset is not None:
                    offset_samples.append((float(offset[0]), float(offset[1])))
                color = self._status.get("detected_color")
                if isinstance(color, str) and color:
                    color_samples.append(color)

            average_center = (
                sum(point[0] for point in center_samples) / len(center_samples),
                sum(point[1] for point in center_samples) / len(center_samples),
            )
            average_offset = (
                sum(point[0] for point in offset_samples) / len(offset_samples),
                sum(point[1] for point in offset_samples) / len(offset_samples),
            ) if offset_samples else None
            detected_color = max(set(color_samples), key=color_samples.count) if color_samples else None
            return {
                **self._status,
                "roi": dict(self._status["roi"]),
                "detected_center": list(average_center),
                "detected_offset_roi": list(average_offset) if average_offset is not None else None,
                "sample_count": len(center_samples),
                "color_sample_count": len(color_samples),
                "detected_color": detected_color,
                "sequence": sequence,
            }

    def wait_for_circle_detections(
        self,
        after_sequence: int,
        frame_count: int = 10,
        timeout_seconds: float = 8.0,
    ) -> dict[str, Any] | None:
        """Collect whole-frame circles from ``frame_count`` fresh frames and merge them.

        Returns ``None`` if the camera stops or times out before enough frames
        arrive. ``circles`` holds caps seen in at least half the frames.
        """
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero")
        deadline = time.monotonic() + timeout_seconds
        frames: list[list[CapCircle]] = []
        sequence = after_sequence
        with self._condition:
            while len(frames) < frame_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._status["state"] in {"unavailable", "error", "stopped"}:
                    return None
                self._condition.wait_for(
                    lambda: self._sequence > sequence
                    or self._status["state"] in {"unavailable", "error", "stopped"}
                    or self._stop_event.is_set(),
                    timeout=remaining,
                )
                if self._sequence <= sequence:
                    return None
                sequence = self._sequence
                frames.append([
                    CapCircle(x=c["x"], y=c["y"], radius=c["radius"], color=c["color"])
                    for c in self._status.get("detected_circles") or []
                ])
            width, height = self._status["width"], self._status["height"]
        merged: list[MergedCircle] = merge_frame_detections(frames)
        return {
            "circles": merged,
            "frame_count": len(frames),
            "width": width,
            "height": height,
            "sequence": sequence,
        }

    def _update_color_range(self, sample: dict[str, float | int] | None) -> None:
        if sample is None:
            return
        with self._condition:
            if not self._record_color_range:
                return
            if self._recorded_color_range is None:
                self._recorded_color_range = dict(sample)
            else:
                for key in (
                    "hue_min",
                    "saturation_min",
                    "value_min",
                ):
                    self._recorded_color_range[key] = min(
                        float(self._recorded_color_range[key]),
                        float(sample[key]),
                    )
                for key in (
                    "hue_max",
                    "saturation_max",
                    "value_max",
                ):
                    self._recorded_color_range[key] = max(
                        float(self._recorded_color_range[key]),
                        float(sample[key]),
                    )
                self._recorded_color_range["sample_count"] = (
                    int(self._recorded_color_range["sample_count"])
                    + int(sample["sample_count"])
                )
            self._status["color_range"] = dict(self._recorded_color_range)
            self._condition.notify_all()

    def _capture_loop(self) -> None:
        locator = TubeLocator(show_preview=False, roi_rotation_deg=self.settings.roi_rotation_deg)
        locator_generation = self._locator_generation
        center_history: deque[tuple[int, int]] = deque(maxlen=9)
        radius_history: deque[int] = deque(maxlen=9)
        previous_detection_mode = "hough"
        try:
            while not self._stop_event.is_set():
                capture = self._open_capture()
                if not capture.isOpened():
                    capture.release()
                    self._set_status(
                        available=False,
                        state="unavailable",
                        message=f"Cannot open {self._device_name()}. Check the camera connection and Linux video permissions.",
                    )
                    self._stop_event.wait(2)
                    continue

                with self._condition:
                    self._capture = capture
                    self._status.update(available=True, state="connecting", message="Camera opened; waiting for frames.")
                    self._condition.notify_all()
                center_history.clear()
                radius_history.clear()
                try:
                    while not self._stop_event.is_set():
                        with self._condition:
                            if self._reopen_requested:
                                self._reopen_requested = False
                                reopen = True
                            else:
                                reopen = False
                                generation = self._locator_generation
                        if reopen:
                            break
                        if generation != locator_generation:
                            locator = TubeLocator(
                                show_preview=False,
                                roi_rotation_deg=self.settings.roi_rotation_deg,
                            )
                            locator_generation = generation
                            center_history.clear()
                            radius_history.clear()
                        frame_started = time.monotonic()
                        grabbed, frame = capture.read()
                        if not grabbed or frame is None:
                            self._set_status(
                                available=False,
                                state="retrying",
                                message="Camera frame read failed; reconnecting.",
                            )
                            break

                        self._service_photo_request(frame)
                        frame, roi = self._prepare_preview_frame(frame)
                        with self._condition:
                            detection_mode = self._detection_mode
                            locator.circle_min_distance = self._circle_min_distance
                            locator.circle_edge_threshold = self._circle_edge_threshold
                            locator.circle_canny_threshold = self._circle_canny_threshold
                            locator.circle_method = self._circle_method
                        if detection_mode != previous_detection_mode:
                            center_history.clear()
                            radius_history.clear()
                            previous_detection_mode = detection_mode
                        debug_montage: np.ndarray | None = None
                        all_circles: list[CapCircle] = []
                        detected_offset_roi = None
                        if detection_mode == "hough":
                            # Whole frame, no ROI: every cap is reported and the
                            # workflow decides which one belongs to the slot.
                            all_circles, mask, debug_montage = locator.detect_all_circles(frame)
                            nearest = min(
                                all_circles,
                                key=lambda c: (c.x - frame.shape[1] / 2) ** 2 + (c.y - frame.shape[0] / 2) ** 2,
                                default=None,
                            )
                            center = None if nearest is None else (int(round(nearest.x)), int(round(nearest.y)))
                            color = None if nearest is None else nearest.color
                            tube_present = bool(all_circles)
                        else:
                            center, tube_present, color, mask = locator.locate_with_mask(frame, roi)
                        self._update_color_range(locator.last_color_stats)
                        if mask is None:
                            self._set_status(
                                available=False,
                                state="error",
                                message="The locator could not produce an HSV mask for the camera frame.",
                            )
                            break
                        if detection_mode == "hough":
                            preview = locator.annotate_circles(frame, all_circles, center)
                        else:
                            preview = locator.annotate(
                                frame,
                                roi,
                                center,
                                tube_present,
                                color=color,
                                color_radius=locator.last_color_radius,
                            )
                        encoded, image = cv2.imencode(
                            ".jpg",
                            preview,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        )
                        if not encoded:
                            self._set_status(available=False, state="error", message="OpenCV could not encode the camera frame.")
                            break
                        mask_encoded, mask_image = cv2.imencode(
                            ".jpg",
                            mask,
                            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        )
                        if not mask_encoded:
                            self._set_status(available=False, state="error", message="OpenCV could not encode the HSV mask.")
                            break
                        debug_image: bytes | None = None
                        if debug_montage is not None:
                            debug_encoded, debug_buffer = cv2.imencode(
                                ".jpg",
                                debug_montage,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                            )
                            if not debug_encoded:
                                self._set_status(available=False, state="error", message="OpenCV could not encode the Hough debug Mats.")
                                break
                            debug_image = debug_buffer.tobytes()

                        with self._condition:
                            self._sequence += 1
                            self._jpeg = image.tobytes()
                            self._mask_jpeg = mask_image.tobytes()
                            self._debug_jpeg = debug_image
                            self._status.update(
                                available=True,
                                state="streaming",
                                message="Live camera preview.",
                                width=PREVIEW_WIDTH,
                                height=PREVIEW_HEIGHT,
                                roi=self._roi_status(roi),
                                detected_center=center,
                                detected_color=color,
                                detected_offset_roi=detected_offset_roi,
                                detected_circles=[
                                    {"x": c.x, "y": c.y, "radius": c.radius, "color": c.color}
                                    for c in all_circles
                                ],
                                tube_present=tube_present,
                                sequence=self._sequence,
                            )
                            self._condition.notify_all()
                        self._stop_event.wait(max(0.0, 1 / self.fps - (time.monotonic() - frame_started)))
                finally:
                    capture.release()
                    with self._condition:
                        if self._capture is capture:
                            self._capture = None

                if not self._stop_event.is_set():
                    self._stop_event.wait(1)
        finally:
            locator.close()
            self._set_status(available=False, state="stopped", message="Camera preview stopped.")

    def _open_capture(self) -> cv2.VideoCapture:
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.settings.device_index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(self.settings.device_index, cv2.CAP_ANY)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_HEIGHT)
        return capture

    def _service_photo_request(self, frame: np.ndarray) -> None:
        with self._condition:
            if not self._photo_requested:
                return
            directory = self._photo_directory
        try:
            path = self._write_photo(frame, directory)
        except Exception as exc:
            with self._condition:
                self._photo_requested = False
                self._photo_error = f"OpenCV could not save the camera photo: {exc}"
                self._condition.notify_all()
            return
        with self._condition:
            self._photo_requested = False
            self._photo_result = {
                "filename": path.name,
                "path": str(path),
                "width": PREVIEW_WIDTH,
                "height": PREVIEW_HEIGHT,
            }
            self._condition.notify_all()

    @staticmethod
    def _write_photo(frame: np.ndarray, directory: Path) -> Path:
        height, width = frame.shape[:2]
        if (width, height) != (PREVIEW_WIDTH, PREVIEW_HEIGHT):
            frame = cv2.resize(
                frame,
                (PREVIEW_WIDTH, PREVIEW_HEIGHT),
                interpolation=cv2.INTER_AREA if width > PREVIEW_WIDTH else cv2.INTER_LINEAR,
            )
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = directory / f"photo_{stamp}_{int(time.time() * 1000) % 1000:03d}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"OpenCV could not write {path}.")
        return path

    def _prepare_preview_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        source_height, source_width = frame.shape[:2]
        scale = min(PREVIEW_WIDTH / source_width, PREVIEW_HEIGHT / source_height)
        width = max(1, round(source_width * scale))
        height = max(1, round(source_height * scale))
        resized = cv2.resize(
            frame,
            (width, height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        offset_x = (PREVIEW_WIDTH - width) // 2
        offset_y = (PREVIEW_HEIGHT - height) // 2
        preview = cv2.copyMakeBorder(
            resized,
            offset_y,
            PREVIEW_HEIGHT - height - offset_y,
            offset_x,
            PREVIEW_WIDTH - width - offset_x,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        scale_x = width / self.settings.resolution.width
        scale_y = height / self.settings.resolution.height
        return (
            preview,
            (
                round(self.settings.roi_center_x * scale_x) + offset_x,
                round(self.settings.roi_center_y * scale_y) + offset_y,
                max(2, round(self.settings.roi_width * scale_x)),
                max(2, round(self.settings.roi_height * scale_y)),
            ),
        )

    def _configured_roi(self) -> dict[str, int]:
        return self._roi_status((
            self.settings.roi_center_x,
            self.settings.roi_center_y,
            self.settings.roi_width,
            self.settings.roi_height,
        ))

    @staticmethod
    def _roi_status(roi: tuple[int, int, int, int]) -> dict[str, int]:
        center_x, center_y, width, height = roi
        return {"center_x": center_x, "center_y": center_y, "width": width, "height": height}

    def _device_name(self) -> str:
        if sys.platform.startswith("linux"):
            return f"/dev/video{self.settings.device_index}"
        return f"Camera {self.settings.device_index}"

    def _set_status(self, **updates: Any) -> None:
        with self._condition:
            self._status.update(updates)
            self._condition.notify_all()