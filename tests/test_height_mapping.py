from __future__ import annotations

import unittest

import cv2
import numpy as np

from controller.vision.tubeLocator import tube_locator
from controller.vision.tubeLocator.tube_locator import (
    _decompose_rack_pose,
    pixel_to_real,
    rack_camera_pose,
    real_to_pixel,
)


def synthetic_homography(
    camera_matrix: np.ndarray,
    camera_xyz: tuple[float, float, float],
    tilt_deg: float,
) -> np.ndarray:
    """World(mm, Z up)->pixel homography of the Z=0 plane for a camera looking down."""
    tilt = np.radians(tilt_deg)
    # Camera looking down -Z (x right, y down in image), then tilted about X.
    look_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tilt_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(tilt), -np.sin(tilt)], [0.0, np.sin(tilt), np.cos(tilt)]]
    )
    rotation = tilt_x @ look_down
    translation = -rotation @ np.asarray(camera_xyz, dtype=np.float64)
    homography = camera_matrix @ np.column_stack((rotation[:, 0], rotation[:, 1], translation))
    return homography / homography[2, 2], rotation, translation


class RackPoseDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    def test_recovers_camera_height_and_tilt(self) -> None:
        homography, _, _ = synthetic_homography(self.camera_matrix, (15.0, -20.0, 120.0), 3.0)
        pose = _decompose_rack_pose(homography, self.camera_matrix)
        self.assertAlmostEqual(pose.camera_height_mm, 120.0, places=6)
        self.assertAlmostEqual(pose.tilt_deg, 3.0, places=6)
        self.assertAlmostEqual(pose.camera_xy_mm[0], 15.0, places=6)
        self.assertAlmostEqual(pose.camera_xy_mm[1], -20.0, places=6)

    def test_height_column_matches_true_projection(self) -> None:
        homography, rotation, translation = synthetic_homography(
            self.camera_matrix, (15.0, -20.0, 120.0), 3.0
        )
        pose = _decompose_rack_pose(homography, self.camera_matrix)
        lifted = homography.copy()
        lifted[:, 2] += 40.0 * pose.height_column
        for world in ((0.0, 0.0), (30.0, 10.0), (-25.0, 45.0)):
            camera_point = rotation @ np.array([world[0], world[1], 40.0]) + translation
            expected = (self.camera_matrix @ camera_point)[:2] / camera_point[2]
            mapped = cv2.perspectiveTransform(np.array([[world]], dtype=np.float64), lifted)[0, 0]
            np.testing.assert_allclose(mapped, expected, atol=1e-6)


class PixelToRealHeightTests(unittest.TestCase):
    def test_zero_height_is_the_plain_homography(self) -> None:
        pixel = np.array([[[400.0, 300.0]]], dtype=np.float64)
        expected = cv2.perspectiveTransform(pixel, tube_locator.pixel_to_world)[0, 0]
        np.testing.assert_allclose(pixel_to_real(400.0, 300.0), expected)

    def test_round_trip_at_height(self) -> None:
        for height in (0.0, 20.0, 45.0):
            pixel = real_to_pixel(30.0, -15.0, height_mm=height)
            np.testing.assert_allclose(pixel_to_real(*pixel, height_mm=height), (30.0, -15.0), atol=1e-6)

    def test_raised_points_appear_further_from_the_camera_nadir(self) -> None:
        pose = rack_camera_pose()
        nadir = np.array(pose.camera_xy_mm)
        rack_point = nadir + np.array([40.0, 0.0])
        nadir_pixel = np.array(real_to_pixel(*nadir))
        low = np.linalg.norm(np.array(real_to_pixel(*rack_point)) - nadir_pixel)
        high = np.linalg.norm(np.array(real_to_pixel(*rack_point, height_mm=40.0)) - nadir_pixel)
        expected_ratio = pose.camera_height_mm / (pose.camera_height_mm - 40.0)
        self.assertAlmostEqual(high / low, expected_ratio, delta=0.05 * expected_ratio)

    def test_camera_raise_cancels_object_height(self) -> None:
        self.assertEqual(
            pixel_to_real(250.0, 200.0, height_mm=30.0, camera_raise_mm=30.0),
            pixel_to_real(250.0, 200.0),
        )

    def test_rejects_heights_at_or_above_the_camera(self) -> None:
        with self.assertRaises(ValueError):
            pixel_to_real(320.0, 240.0, height_mm=rack_camera_pose().camera_height_mm)


if __name__ == "__main__":
    unittest.main()
