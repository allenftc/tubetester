from __future__ import annotations

import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from controller.config.settings import CameraSettings
from controller.vision.tubeLocator.tube_locator import TubeLocator


PREVIEW_WIDTH = 320
PREVIEW_HEIGHT = 240


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
        self._sequence = 0
        self._jpeg: bytes | None = None
        self._mask_jpeg: bytes | None = None
        self._debug_jpeg: bytes | None = None
        self._detection_mode = "hsv"
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
            "detection_mode": self._detection_mode,
            "sequence": 0,
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
            }

    def set_detection_mode(self, mode: str) -> None:
        if mode not in {"hsv", "hough"}:
            raise ValueError("detection_mode must be 'hsv' or 'hough'")
        with self._condition:
            self._detection_mode = mode
            self._status["detection_mode"] = mode
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
    ) -> dict[str, Any] | None:
        """Average valid Hough offsets from successive frames in rotated ROI space."""
        if frame_count <= 0:
            raise ValueError("frame_count must be greater than zero")
        deadline = time.monotonic() + timeout_seconds
        samples: list[tuple[float, float]] = []
        sequence = after_sequence
        with self._condition:
            while len(samples) < frame_count:
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
                offset = self._status["detected_offset_roi"]
                if offset is not None:
                    samples.append((float(offset[0]), float(offset[1])))

            average = (
                sum(offset[0] for offset in samples) / len(samples),
                sum(offset[1] for offset in samples) / len(samples),
            )
            return {
                **self._status,
                "roi": dict(self._status["roi"]),
                "detected_center": list(self._status["detected_center"])
                if self._status["detected_center"] is not None
                else None,
                "detected_offset_roi": list(average),
                "sample_count": len(samples),
                "sequence": sequence,
            }

    def _capture_loop(self) -> None:
        locator = TubeLocator(show_preview=False, roi_rotation_deg=self.settings.roi_rotation_deg)
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
                try:
                    while not self._stop_event.is_set():
                        frame_started = time.monotonic()
                        grabbed, frame = capture.read()
                        if not grabbed or frame is None:
                            self._set_status(
                                available=False,
                                state="retrying",
                                message="Camera frame read failed; reconnecting.",
                            )
                            break

                        frame, roi = self._prepare_preview_frame(frame)
                        with self._condition:
                            detection_mode = self._detection_mode
                        circles: list[tuple[int, int, int]] = []
                        debug_montage: np.ndarray | None = None
                        if detection_mode == "hough":
                            center, tube_present, mask, circles, debug_montage = locator.locate_with_circles(frame, roi)
                        else:
                            center, tube_present, mask = locator.locate_with_mask(frame, roi)
                        detected_offset_roi = (
                            locator.roi_offset_from_frame_point(center, roi)
                            if detection_mode == "hough" and center is not None
                            else None
                        )
                        if mask is None:
                            self._set_status(
                                available=False,
                                state="error",
                                message="The locator could not produce an HSV mask for the camera frame.",
                            )
                            break
                        preview = locator.annotate(frame, roi, center, tube_present, circles)
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
                                detected_offset_roi=detected_offset_roi,
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