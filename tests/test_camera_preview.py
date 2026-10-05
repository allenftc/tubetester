from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from controller.config.settings import load_settings
from controller.vision.camera_preview import CameraPreviewService


class FakeCapture:
    def __init__(self, *_args: object) -> None:
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def set(self, *_args: object) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        pass


class CameraPreviewTests(unittest.TestCase):
    def test_preview_streams_annotated_jpeg_and_scales_roi_to_frame(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration").camera
        preview = CameraPreviewService(settings, fps=10)

        with patch("controller.vision.camera_preview.cv2.VideoCapture", FakeCapture):
            preview.start()
            sequence, jpeg = preview.wait_for_frame(0, timeout_seconds=3)
            status = preview.status()
            preview.stop()

        self.assertGreater(sequence, 0)
        self.assertIsNotNone(jpeg)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(status["state"], "streaming")
        self.assertEqual(status["roi"], {
            "center_x": round(settings.roi_center_x * 320 / settings.resolution.width),
            "center_y": round(settings.roi_center_y * 240 / settings.resolution.height),
            "width": round(settings.roi_width * 320 / settings.resolution.width),
            "height": round(settings.roi_height * 240 / settings.resolution.height),
        })
        self.assertIsInstance(status["tube_present"], bool)


if __name__ == "__main__":
    unittest.main()