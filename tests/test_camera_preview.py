from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from controller.config.settings import load_settings
from controller.vision.camera_preview import (
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    CameraPreviewService,
)
from controller.vision.tubeLocator.tube_locator import ColorRange, TubeLocator
from controller.vision.tubeLocator.tube_locator import triangulate_stereo_circles


class FakeCapture:
    instances: list[FakeCapture] = []

    def __init__(self, *_args: object) -> None:
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.requested_size: dict[int, int] = {}
        self.instances.append(self)

    def isOpened(self) -> bool:
        return True

    def set(self, property_id: int, value: int) -> bool:
        self.requested_size[property_id] = value
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, self.frame.copy()

    def release(self) -> None:
        pass


class CameraPreviewTests(unittest.TestCase):
    def test_locator_exposes_raw_threshold_before_inversion_and_cleanup(self) -> None:
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[8:12, 8:12] = (255, 0, 0)
        locator = TubeLocator(
            lower_hsv=(110, 200, 100),
            upper_hsv=(120, 255, 255),
            invert_mask=True,
            min_contour_area=1,
            show_preview=False,
            roi_rotation_deg=0,
        )

        _, _, raw_mask = locator.locate_with_mask(frame, (10, 10, 10, 10))

        self.assertIsNotNone(raw_mask)
        assert raw_mask is not None
        self.assertEqual(raw_mask[4, 4], 255)
        self.assertEqual(raw_mask[0, 0], 0)

    def test_hough_circles_are_detected_inside_rotated_roi(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        locator = TubeLocator(show_preview=False, roi_rotation_deg=0)
        detected = np.array([[[10.0, 10.0, 4.0]]], dtype=np.float32)

        with patch("controller.vision.tubeLocator.tube_locator.cv2.HoughCircles", return_value=detected):
            center, tube_present, raw_mask, circles, debug = locator.locate_with_circles(frame, (20, 20, 20, 20))

        self.assertEqual(center, (20, 20))
        self.assertTrue(tube_present)
        self.assertEqual(circles, [(20, 20, 4)])
        self.assertIsNotNone(raw_mask)
        self.assertEqual(raw_mask.shape, (20, 20))
        self.assertIsNotNone(debug)
        self.assertEqual(debug.shape, (80, 80, 3))

    def test_hough_detector_finds_synthetic_circle(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(frame, (50, 50), 20, (255, 255, 255), 2)
        locator = TubeLocator(show_preview=False, roi_rotation_deg=0)

        center, tube_present, _, circles, debug = locator.locate_with_circles(frame, (50, 50, 80, 80))

        self.assertTrue(tube_present)
        self.assertIsNotNone(center)
        self.assertLessEqual(np.linalg.norm(np.array(center) - np.array((50, 50))), 3)
        self.assertTrue(circles)
        self.assertEqual(debug.shape, (320, 320, 3))

    def test_stereo_triangulation_uses_calibration_and_scaled_image_dimensions(self) -> None:
        calibration = np.load(
            Path(__file__).resolve().parents[1] / "controller/vision/tubeLocator/camera_calibration.npz"
        )
        camera_matrix = calibration["camera_matrix"]
        distortion = calibration["distortion_coefficients"]
        calibration.close()
        object_points = np.array([[[5.0, 0.0, 100.0], [-5.0, 0.0, 100.0]]], dtype=np.float32)
        left_pixel, right_pixel = (
            cv2.projectPoints(point.reshape(1, 1, 3), np.zeros(3), np.zeros(3), camera_matrix, distortion)[0][0, 0]
            for point in object_points[0]
        )

        xyz = triangulate_stereo_circles(
            (float(left_pixel[0]), float(left_pixel[1])),
            (float(right_pixel[0]), float(right_pixel[1])),
            (640, 480),
        )
        xyz_half_size = triangulate_stereo_circles(
            (float(left_pixel[0] / 2), float(left_pixel[1] / 2)),
            (float(right_pixel[0] / 2), float(right_pixel[1] / 2)),
            (320, 240),
        )

        np.testing.assert_allclose(xyz, (0.0, 0.0, 100.0), atol=0.05)
        np.testing.assert_allclose(xyz_half_size, xyz, atol=0.05)

    def test_stereo_triangulation_rejects_invalid_disparity(self) -> None:
        with self.assertRaisesRegex(ValueError, "disparity must be positive"):
            triangulate_stereo_circles((160, 120), (160, 120), (320, 240))

    def test_preview_preserves_widescreen_aspect_ratio_and_roi_position(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration").camera
        preview = CameraPreviewService(settings)
        source = np.full((720, 1280, 3), 255, dtype=np.uint8)

        image, roi = preview._prepare_preview_frame(source)

        self.assertEqual(image.shape, (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3))
        self.assertTrue(np.all(image[:30] == 0))
        self.assertTrue(np.all(image[30:210] == 255))
        self.assertTrue(np.all(image[210:] == 0))
        self.assertEqual(roi, (
            round(settings.roi_center_x * 320 / settings.resolution.width),
            round(settings.roi_center_y * 180 / settings.resolution.height) + 30,
            round(settings.roi_width * 320 / settings.resolution.width),
            round(settings.roi_height * 180 / settings.resolution.height),
        ))

    def test_preview_streams_annotated_jpeg_and_scales_roi_to_frame(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration").camera
        preview = CameraPreviewService(settings, fps=10)

        with patch("controller.vision.camera_preview.cv2.VideoCapture", FakeCapture):
            FakeCapture.instances.clear()
            preview.start()
            sequence, jpeg = preview.wait_for_frame(0, timeout_seconds=3)
            mask_sequence, mask_jpeg = preview.wait_for_mask_frame(0, timeout_seconds=3)
            status = preview.status()
            preview.stop()

        self.assertGreater(sequence, 0)
        self.assertIsNotNone(jpeg)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertEqual(status["state"], "streaming")
        self.assertEqual(status["width"], PREVIEW_WIDTH)
        self.assertEqual(status["height"], PREVIEW_HEIGHT)
        self.assertEqual(status["detection_mode"], "hsv")
        self.assertEqual(
            FakeCapture.instances[0].requested_size,
            {
                cv2.CAP_PROP_FRAME_WIDTH: PREVIEW_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT: PREVIEW_HEIGHT,
            },
        )
        decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2], (PREVIEW_HEIGHT, PREVIEW_WIDTH))
        self.assertGreaterEqual(mask_sequence, sequence)
        self.assertIsNotNone(mask_jpeg)
        decoded_mask = cv2.imdecode(np.frombuffer(mask_jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(decoded_mask.shape, (status["roi"]["height"], status["roi"]["width"]))
        self.assertEqual(status["roi"], {
            "center_x": round(settings.roi_center_x * PREVIEW_WIDTH / settings.resolution.width),
            "center_y": round(settings.roi_center_y * PREVIEW_HEIGHT / settings.resolution.height),
            "width": round(settings.roi_width * PREVIEW_WIDTH / settings.resolution.width),
            "height": round(settings.roi_height * PREVIEW_HEIGHT / settings.resolution.height),
        })
        self.assertIsInstance(status["tube_present"], bool)

    def test_detection_mode_can_be_switched_and_rejects_unknown_modes(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration").camera
        preview = CameraPreviewService(settings)

        preview.set_detection_mode("hough")

        self.assertEqual(preview.status()["detection_mode"], "hough")
        with self.assertRaises(ValueError):
            preview.set_detection_mode("unknown")

    def test_preview_uses_hough_detector_when_selected(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration").camera
        preview = CameraPreviewService(settings, fps=10)
        preview.set_detection_mode("hough")
        raw_mask = np.zeros((80, 80), dtype=np.uint8)
        debug_mats = np.zeros((160, 160, 3), dtype=np.uint8)

        with (
            patch("controller.vision.camera_preview.cv2.VideoCapture", FakeCapture),
            patch(
                "controller.vision.camera_preview.TubeLocator.locate_with_circles",
                return_value=(None, False, raw_mask, [], debug_mats),
            ) as locate_with_circles,
        ):
            preview.start()
            sequence, jpeg = preview.wait_for_frame(0, timeout_seconds=3)
            debug_sequence, debug_jpeg = preview.wait_for_debug_frame(0, timeout_seconds=3)
            status = preview.status()
            preview.stop()

        self.assertGreater(sequence, 0)
        self.assertGreaterEqual(debug_sequence, sequence)
        self.assertIsNotNone(jpeg)
        self.assertIsNotNone(debug_jpeg)
        debug_frame = cv2.imdecode(np.frombuffer(debug_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(debug_frame.shape[:2], (160, 160))
        self.assertTrue(locate_with_circles.called)
        self.assertEqual(status["detection_mode"], "hough")


class ColorClassificationTests(unittest.TestCase):
    """Ring-sampling cap-colour classification.

    The scenes are a blue plate (the locator's thresholded background) with a
    coloured cap disc, so the classifier sees a real ROI rather than a flat
    swatch: it needs background pixels for the relative-brightness rule and a
    mask to keep its samples on the cap.
    """

    PLATE = (110, 150, 100)
    SIZE = 60
    CAP_RADIUS = 20

    def _hsv_to_bgr(self, hsv):
        return cv2.cvtColor(np.array([[hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]

    def _scene(self, cap_hsv, scale=1.0):
        cap = (cap_hsv[0], cap_hsv[1], min(255, int(cap_hsv[2] * scale)))
        frame = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        frame[:, :] = self._hsv_to_bgr(self.PLATE)
        yy, xx = np.mgrid[0 : self.SIZE, 0 : self.SIZE]
        disc = (xx - self.SIZE // 2) ** 2 + (yy - self.SIZE // 2) ** 2 <= self.CAP_RADIUS**2
        frame[disc] = self._hsv_to_bgr(cap)
        return frame

    def _classify(self, frame):
        locator = TubeLocator(show_preview=False, roi_rotation_deg=0, min_contour_area=1)
        _, _, color, _ = locator.locate_with_mask(frame, (self.SIZE // 2, self.SIZE // 2, self.SIZE, self.SIZE))
        return color

    def test_each_cap_colour_is_classified_from_a_realistic_roi(self) -> None:
        cases = {
            "white": (0, 20, 240),
            "green": (60, 150, 180),
            "pink": (165, 120, 200),
            "red": (5, 150, 200),
            "gray": (0, 30, 120),
            "cyan": (90, 150, 200),
        }
        for name, hsv in cases.items():
            with self.subTest(color=name):
                self.assertEqual(self._classify(self._scene(hsv)), name)

    def test_red_cap_on_the_far_side_of_the_hue_wrap_reads_as_red(self) -> None:
        # OpenCV hue wraps at 180, so red spans 170-180 and 0-10.
        self.assertEqual(self._classify(self._scene((175, 150, 200))), "red")

    def test_white_cap_under_dim_light_still_reads_as_white(self) -> None:
        # Dim light pulls every value down; white is still brighter than its
        # background, which is what the relative-brightness rule keys on.
        self.assertEqual(self._classify(self._scene((0, 20, 240), scale=0.6)), "white")

    def test_pale_cap_falls_to_the_chromatic_range_when_both_match(self) -> None:
        # S=55 sits inside both green and white, so priority decides.
        self.assertEqual(self._classify(self._scene((60, 55, 210))), "green")

    def test_returns_none_when_the_ring_does_not_agree(self) -> None:
        frame = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        frame[:, :] = self._hsv_to_bgr(self.PLATE)
        yy, xx = np.mgrid[0 : self.SIZE, 0 : self.SIZE]
        disc = (xx - self.SIZE // 2) ** 2 + (yy - self.SIZE // 2) ** 2 <= self.CAP_RADIUS**2
        top = disc & (yy < self.SIZE // 2)
        frame[top] = self._hsv_to_bgr((60, 150, 180))
        frame[disc & ~top] = self._hsv_to_bgr((0, 20, 240))
        self.assertIsNone(self._classify(frame))

    def test_returns_none_when_the_ring_is_too_small_to_sample(self) -> None:
        frame = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        frame[:, :] = self._hsv_to_bgr(self.PLATE)
        frame[29:31, 29:31] = self._hsv_to_bgr((60, 150, 180))
        self.assertIsNone(self._classify(frame))

    def test_hue_distance_wraps_across_the_180_boundary(self) -> None:
        red = ColorRange("red", (170, 100, 40), (10, 255, 255))
        self.assertEqual(TubeLocator._hue_distance(red, 175), 0)
        self.assertEqual(TubeLocator._hue_distance(red, 3), 0)
        self.assertEqual(TubeLocator._hue_distance(red, 90), 80)

if __name__ == "__main__":
    unittest.main()