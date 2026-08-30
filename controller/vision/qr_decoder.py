from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover - import guard for optional dependency
    Image = None

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except ImportError:  # pragma: no cover - import guard for optional dependency
    pyzbar_decode = None


@dataclass(frozen=True)
class QrDecodeResult:
    payload: str | None
    confidence: float
    frame_id: int | None = None


class QrDecoder:
    """Fast QR decoder tuned for grayscale camera streams and small QR markers."""

    def __init__(self, library: str = "opencv", max_dimension: int = 960) -> None:
        self.library = (library or "opencv").lower()
        self.max_dimension = max_dimension

    def decode(self, frame_bytes: bytes, frame_id: int | None = None) -> QrDecodeResult:
        if not frame_bytes:
            return QrDecodeResult(None, 0.0, frame_id)

        frame = self._decode_to_gray(frame_bytes)
        if frame is None:
            return QrDecodeResult(None, 0.0, frame_id)

        for candidate in self._candidate_images(frame):
            result = self._decode_candidate(candidate, frame_id)
            if result.payload is not None:
                return result

        return QrDecodeResult(None, 0.0, frame_id)

    def decode_from_frame(self, frame: np.ndarray | None, frame_id: int | None = None) -> QrDecodeResult:
        if frame is None:
            return QrDecodeResult(None, 0.0, frame_id)

        gray = self._to_gray(frame)
        if gray is None:
            return QrDecodeResult(None, 0.0, frame_id)

        for candidate in self._candidate_images(gray):
            result = self._decode_candidate(candidate, frame_id)
            if result.payload is not None:
                return result

        return QrDecodeResult(None, 0.0, frame_id)

    def scan_camera(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        max_frames: int = 8,
        timeout_seconds: float = 1.5,
    ) -> QrDecodeResult:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            return QrDecodeResult(None, 0.0, None)

        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(fps))

        deadline = time.monotonic() + timeout_seconds
        for frame_id in range(max_frames):
            if time.monotonic() >= deadline:
                break

            grabbed, frame = capture.read()
            if not grabbed or frame is None:
                continue

            result = self.decode_from_frame(frame, frame_id=frame_id)
            if result.payload is not None:
                capture.release()
                return result

        capture.release()
        return QrDecodeResult(None, 0.0, None)

    def preview_stream(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> None:
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            print("Unable to open camera.")
            return

        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(fps))

        preview_width = 480
        preview_height = 320
        window_name = "QR Preview"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        frame_id = 0

        try:
            while True:
                grabbed, frame = capture.read()
                if not grabbed or frame is None:
                    time.sleep(0.001)
                    continue

                if frame_id % 4 != 0:
                    frame_id += 1
                    continue

                gray = self._to_gray(frame)
                if gray is None:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), ord("Q"), 27}:
                        break
                    frame_id += 1
                    continue

                detector = cv2.QRCodeDetector()
                data, points, _ = detector.detectAndDecode(gray)
                if points is not None and len(points) > 0:
                    points = points.astype(int)
                    for idx in range(4):
                        p1 = tuple(points[0][idx])
                        p2 = tuple(points[0][(idx + 1) % 4])
                        cv2.line(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, "QR detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cropped = self._crop_to_points(gray, points[0])
                    payload = self._decode_variants(cropped, frame_id)
                    if payload is not None:
                        cv2.putText(
                            frame,
                            f"QR: {payload}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                        print(f"Decoded QR payload: {payload}")

                preview = cv2.resize(frame, (preview_width, preview_height), interpolation=cv2.INTER_LINEAR)
                cv2.imshow(window_name, preview)
                frame_id += 1

                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), ord("Q"), 27}:
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    def _decode_candidate(self, gray: np.ndarray, frame_id: int | None) -> QrDecodeResult:
        if self.library == "pyzbar" and pyzbar_decode is not None and Image is not None:
            return self._decode_with_pyzbar(gray, frame_id)
        return self._decode_with_opencv(gray, frame_id)

    def _decode_with_opencv(self, gray: np.ndarray, frame_id: int | None) -> QrDecodeResult:
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(gray)
        if data:
            return QrDecodeResult(data, 0.95, frame_id)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        data, points, _ = detector.detectAndDecode(binary)
        if data:
            return QrDecodeResult(data, 0.9, frame_id)

        for scale in (1.0, 1.25, 1.5):
            resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            data, points, _ = detector.detectAndDecode(resized)
            if data:
                return QrDecodeResult(data, 0.85, frame_id)

        return QrDecodeResult(None, 0.0, frame_id)

    def _decode_variants(self, gray: np.ndarray, frame_id: int | None) -> str | None:
        variants: list[np.ndarray] = [gray]
        if gray.ndim == 2:
            variants.append(cv2.equalizeHist(gray))
            variants.append(cv2.GaussianBlur(gray, (3, 3), 0))

        for variant in variants:
            for scale in (0.9, 1.0, 1.1, 1.25):
                resized = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                result = self._decode_candidate(resized, frame_id)
                if result.payload is not None:
                    return result.payload
        return None

    def _decode_with_pyzbar(self, gray: np.ndarray, frame_id: int | None) -> QrDecodeResult:
        if pyzbar_decode is None or Image is None:
            return QrDecodeResult(None, 0.0, frame_id)

        pil_image = Image.fromarray(gray)
        for symbol in pyzbar_decode(pil_image):
            payload = symbol.data.decode("utf-8", errors="replace")
            if payload:
                return QrDecodeResult(payload, 0.85, frame_id)
        return QrDecodeResult(None, 0.0, frame_id)

    def _candidate_images(self, gray: np.ndarray) -> list[np.ndarray]:
        candidates: list[np.ndarray] = []
        resized = self._resize_for_speed(gray)
        if resized is not None:
            candidates.append(resized)
        candidates.append(gray)
        return candidates

    def _resize_for_speed(self, gray: np.ndarray) -> np.ndarray | None:
        height, width = gray.shape[:2]
        if max(height, width) <= self.max_dimension:
            return None

        scale = self.max_dimension / max(height, width)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _crop_to_points(self, gray: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.array(points, dtype=np.float32)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped = gray[y : y + h, x : x + w]
        if cropped.size == 0:
            return gray
        return cropped

    def _decode_to_gray(self, frame_bytes: bytes) -> np.ndarray | None:
        array = np.frombuffer(frame_bytes, dtype=np.uint8)
        if array.size == 0:
            return None
        image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        return self._to_gray(image)

    def _to_gray(self, image: np.ndarray) -> np.ndarray | None:
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return None