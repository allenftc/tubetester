from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from controller.config.settings import load_settings
from controller.web.server import create_control_app
from controller.vision.tubeLocator.cap_matching import (
    LocatedCap,
    MergedCircle,
    assign_caps_to_slots,
    locate_caps,
    merge_frame_detections,
    select_cap,
)
from controller.vision.tubeLocator.tube_locator import (
    CALIBRATION_GANTRY_Z_MM,
    CapCircle,
    TubeLocator,
    real_to_pixel,
)
from controller.web.runtime import RuntimeUnavailable, WorkflowRuntime
from tests.test_web import FakeCameraPreview, FakeMoonraker

SETTINGS_DIR = Path(__file__).resolve().parents[1] / "calibration"


def cap_pixel(cap_xy, gantry_xy, gantry_z, cap_height):
    """Where a cap at gantry-frame ``cap_xy`` appears for a camera at ``gantry_xy``/``gantry_z``."""
    return real_to_pixel(
        cap_xy[0] - gantry_xy[0],
        cap_xy[1] - gantry_xy[1],
        height_mm=cap_height,
        camera_raise_mm=gantry_z - CALIBRATION_GANTRY_Z_MM,
    )


class MergeFrameDetectionsTests(unittest.TestCase):
    def test_averages_stable_caps_and_drops_flicker(self) -> None:
        frames = [
            [CapCircle(100, 100, 30, "red"), CapCircle(300, 200, 25, "white")],
            [CapCircle(102, 98, 32, "red"), CapCircle(301, 201, 25, None)],
            [CapCircle(101, 101, 31, "purple"), CapCircle(500, 50, 20, "gray")],
            [CapCircle(99, 99, 29, "red")],
        ]
        merged = sorted(merge_frame_detections(frames), key=lambda c: c.x)
        self.assertEqual(len(merged), 2)  # the gray one-frame flicker is dropped
        red, white = merged
        self.assertAlmostEqual(red.x, 100.5)
        self.assertAlmostEqual(red.y, 99.5)
        self.assertEqual(red.color, "red")  # majority vote
        self.assertEqual(red.frames_seen, 4)
        self.assertEqual(white.color, "white")  # None votes are ignored

    def test_no_frames_gives_no_caps(self) -> None:
        self.assertEqual(merge_frame_detections([]), [])


class LocateAndSelectTests(unittest.TestCase):
    heights = {"red": 39.8, "white": 10.0}

    def test_clumped_tilted_caps_are_matched_by_rack_position(self) -> None:
        gantry, z = (100.0, 50.0), 75.0
        slot = (125.0, 30.0)
        tilted_target = (slot[0] + 4.0, slot[1] - 3.0)
        leaning_neighbor = (slot[0] + 20.2 - 6.0, slot[1])
        white_neighbor = (slot[0], slot[1] + 20.2)
        circles = [
            MergedCircle(*cap_pixel(leaning_neighbor, gantry, z, 39.8), 35, "red", 10),
            MergedCircle(*cap_pixel(tilted_target, gantry, z, 39.8), 35, "red", 10),
            MergedCircle(*cap_pixel(white_neighbor, gantry, z, 10.0), 25, "white", 10),
            MergedCircle(*cap_pixel(slot, gantry, z, 0.0), 20, None, 10),  # unclassified
        ]
        caps = locate_caps(circles, gantry, z, self.heights)
        np.testing.assert_allclose((caps[1].x_mm, caps[1].y_mm), tilted_target, atol=1e-6)
        np.testing.assert_allclose((caps[2].x_mm, caps[2].y_mm), white_neighbor, atol=1e-6)
        self.assertIsNone(caps[3].x_mm)  # no color -> no height -> not placed
        match = select_cap(caps, slot, tolerance_mm=12.0)
        self.assertIs(match, caps[1])

    def test_ignoring_cap_height_would_misplace_a_tall_cap(self) -> None:
        gantry, z = (100.0, 50.0), 75.0
        cap = (140.0, 10.0)
        circle = MergedCircle(*cap_pixel(cap, gantry, z, 39.8), 35, "red", 10)
        flat = locate_caps([circle], gantry, z, {"red": 0.0})[0]
        self.assertGreater(flat.distance_to(*cap), 5.0)

    def test_nothing_within_tolerance_returns_none(self) -> None:
        gantry, z = (100.0, 50.0), 75.0
        circles = [MergedCircle(*cap_pixel((150.0, 40.0), gantry, z, 10.0), 25, "white", 10)]
        caps = locate_caps(circles, gantry, z, self.heights)
        self.assertIsNone(select_cap(caps, (125.0, 30.0), tolerance_mm=12.0))


class SlotAssignmentTests(unittest.TestCase):
    @staticmethod
    def cap(x: float, y: float) -> LocatedCap:
        return LocatedCap((0.0, 0.0), 30.0, "red", x, y, 39.8)

    def test_strongly_leaning_cap_keeps_its_own_slot(self) -> None:
        slots = {(0, 0): (0.0, 0.0), (0, 1): (20.2, 0.0), (0, 2): (40.4, 0.0)}
        leaning = self.cap(13.0, 0.0)  # 13 mm toward slot (0, 1), only 7.2 mm from its center
        neighbor = self.cap(21.0, 1.0)  # slot (0, 1)'s own cap
        assigned = assign_caps_to_slots([leaning, neighbor], slots, max_distance_mm=15.15)
        self.assertIs(assigned[(0, 0)], leaning)
        self.assertIs(assigned[(0, 1)], neighbor)
        self.assertNotIn((0, 2), assigned)

    def test_caps_beyond_the_limit_are_not_assigned(self) -> None:
        assigned = assign_caps_to_slots([self.cap(16.0, 0.0)], {(0, 0): (0.0, 0.0)}, max_distance_mm=15.15)
        self.assertEqual(assigned, {})


class WholeFrameDetectionTests(unittest.TestCase):
    def test_drop_inner_holes_keeps_white_caps(self) -> None:
        circles = [
            CapCircle(0.0, 0.0, 35.0, "red"),
            CapCircle(30.0, 0.0, 33.0, "red"),
            CapCircle(60.0, 0.0, 14.0, "red"),    # inner hole
            CapCircle(90.0, 0.0, 15.0, "gray"),   # inner hole
            CapCircle(120.0, 0.0, 14.0, "white"),  # short white cap
            CapCircle(150.0, 0.0, 20.0, "green"),  # short but above the hole ratio
        ]
        kept = TubeLocator._drop_inner_holes(circles)
        self.assertEqual([(c.x, c.color) for c in kept], [
            (0.0, "red"),
            (30.0, "red"),
            (120.0, "white"),
            (150.0, "green"),
        ])

    def test_drop_inner_holes_does_nothing_when_every_cap_is_small(self) -> None:
        circles = [
            CapCircle(0.0, 0.0, 16.0, "red"),
            CapCircle(30.0, 0.0, 15.0, "red"),
            CapCircle(60.0, 0.0, 14.0, "purple"),
            CapCircle(90.0, 0.0, 9.0, "red"),
        ]
        self.assertEqual(TubeLocator._drop_inner_holes(circles), circles)

    def test_detects_every_cap_without_an_roi(self) -> None:
        frame = np.full((480, 640, 3), (200, 80, 20), dtype=np.uint8)  # blue rack
        caps = {(120, 380): (40, 40, 220), (330, 240): (40, 40, 220), (540, 90): (40, 40, 220)}
        for (x, y), bgr in caps.items():
            cv2.circle(frame, (x, y), 32, bgr, -1)
            cv2.circle(frame, (x, y), 10, (30, 30, 30), -1)
        locator = TubeLocator(show_preview=False)
        circles, mask, montage = locator.detect_all_circles(frame)
        self.assertIsNotNone(mask)
        self.assertIsNotNone(montage)
        found = sorted((c.x, c.y) for c in circles)
        self.assertEqual(len(found), 3, circles)
        for (x, y), (fx, fy) in zip(sorted(caps), found):
            self.assertLessEqual(np.hypot(x - fx, y - fy), 1.0)
        for circle in circles:
            self.assertGreater(circle.radius, 25)  # outer rim, not the hole
            self.assertEqual(circle.color, "red")


class StaticCirclePreview(FakeCameraPreview):
    """Streaming camera fake that returns a fixed set of merged circles."""

    def __init__(self, circles: list[MergedCircle]) -> None:
        super().__init__()
        self.circles = circles
        self.sequence = 3

    def status(self) -> dict[str, object]:
        status = super().status()
        status["sequence"] = self.sequence
        return status

    def wait_for_circle_detections(self, after_sequence: int, frame_count: int = 10, timeout_seconds: float = 8.0):
        return {
            "circles": self.circles,
            "frame_count": frame_count,
            "width": 640,
            "height": 480,
            "sequence": self.sequence,
        }


class FakeCirclePreview(FakeCameraPreview):
    """Camera fake that renders caps from their true gantry-frame positions."""

    def __init__(self, runtime: WorkflowRuntime, caps: list[tuple[tuple[float, float], float, str | None]]) -> None:
        super().__init__()
        self.runtime = runtime
        self.caps = caps
        self.sequence = 1
        self.views: list[tuple[tuple[float, float], float]] = []

    def status(self) -> dict[str, object]:
        status = super().status()
        status.update(state="streaming", available=True, sequence=self.sequence, width=640, height=480)
        return status

    def wait_for_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        return self.sequence, b"frame"

    def wait_for_circle_detections(self, after_sequence: int, frame_count: int = 10, timeout_seconds: float = 8.0):
        session = self.runtime._debug_pickup
        gantry = (session["camera_xy"]["x"], session["camera_xy"]["y"])
        z = session["current_z"]
        self.views.append((gantry, z))
        self.sequence += frame_count
        circles = [
            MergedCircle(*cap_pixel(xy, gantry, z, height), 30, color, frame_count)
            for xy, height, color in self.caps
        ]
        return {"circles": circles, "frame_count": frame_count, "width": 640, "height": 480, "sequence": self.sequence}


class CameraPickupWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings = load_settings(SETTINGS_DIR)
        self.moonraker = FakeMoonraker(ready=True)
        self.runtime = WorkflowRuntime(self.settings, moonraker=self.moonraker)
        await self.runtime.refresh_machine()
        self.moonraker.commands.clear()
        self.rack = self.settings.rack
        self.slot = self.rack.tube_position(1, 1)
        self.red = self.rack.cap_height_for_color("red")

    async def test_survey_matches_tilted_cap_then_centers_close_up_and_moves_over_it(self) -> None:
        cap_xy = (self.slot.x + 4.0, self.slot.y - 3.0)
        caps = [
            (cap_xy, self.red, "red"),
            ((self.slot.x + 20.2 - 6.0, self.slot.y), self.red, "red"),  # neighbor leaning in
            ((self.slot.x, self.slot.y + 20.2), self.rack.cap_height_for_color("white"), "white"),
        ]
        camera = FakeCirclePreview(self.runtime, caps)
        session = await self.runtime.begin_camera_pickup(2, 2, camera)  # type: ignore[arg-type]
        for _ in range(4):
            session = await self.runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]

        offset = self.rack.camera_offset_mm
        survey_camera = (self.slot.x - offset.x, self.slot.y - offset.y)
        closeup_camera = (cap_xy[0] - offset.x, cap_xy[1] - offset.y)
        closeup_z = self.rack.calibrated_scan_height_for_color("red")
        motion = self.runtime.motion
        self.assertEqual(self.moonraker.commands, [
            motion.synchronized_move_command(x=survey_camera[0], y=survey_camera[1], z=self.rack.scan_height_mm, feedrate=10000),
            "\n".join((
                motion.synchronized_move_command(z=closeup_z, feedrate=3000),  # red close-up is above survey Z: climb first
                motion.synchronized_move_command(x=closeup_camera[0], y=closeup_camera[1], feedrate=5000),
            )),
            motion.synchronized_move_command(x=cap_xy[0], y=cap_xy[1], feedrate=5000),
        ])
        self.assertEqual(camera.views[0], (survey_camera, self.rack.scan_height_mm))
        self.assertEqual(camera.views[1][1], closeup_z)
        self.assertEqual(session["detected_color"], "red")
        self.assertAlmostEqual(session["target_xy"]["x"], cap_xy[0], places=6)
        self.assertAlmostEqual(session["target_xy"]["y"], cap_xy[1], places=6)
        self.assertEqual(len(session["survey_caps"]), 3)

    async def test_cap_leaning_past_half_pitch_is_still_matched_when_neighbor_is_occupied(self) -> None:
        leaning = (self.slot.x + 13.0, self.slot.y)
        neighbor_slot = self.rack.tube_position(1, 2)
        camera = FakeCirclePreview(self.runtime, [
            (leaning, self.red, "red"),
            ((neighbor_slot.x + 1.0, neighbor_slot.y), self.red, "red"),
        ])
        session = await self.runtime.begin_camera_pickup(2, 2, camera)  # type: ignore[arg-type]
        for _ in range(2):
            session = await self.runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
        self.assertAlmostEqual(session["survey_xy"]["x"], leaning[0], places=6)

    async def test_descends_after_travel_for_short_caps(self) -> None:
        white = self.rack.cap_height_for_color("white")
        camera = FakeCirclePreview(self.runtime, [((self.slot.x, self.slot.y), white, "white")])
        session = await self.runtime.begin_camera_pickup(2, 2, camera)  # type: ignore[arg-type]
        for _ in range(3):
            session = await self.runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
        closeup_z = self.rack.calibrated_scan_height_for_color("white")
        self.assertLess(closeup_z, self.rack.scan_height_mm)
        self.assertTrue(self.moonraker.commands[1].index(" X") < self.moonraker.commands[1].index(" Z"))

    async def test_empty_slot_fails_without_moving(self) -> None:
        far = (self.slot.x + 40.4, self.slot.y)
        camera = FakeCirclePreview(self.runtime, [(far, self.red, "red")])
        session = await self.runtime.begin_camera_pickup(2, 2, camera)  # type: ignore[arg-type]
        await self.runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
        with self.assertRaisesRegex(RuntimeUnavailable, "No cap could be assigned to row 2, column 2"):
            await self.runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
        self.assertEqual(len(self.moonraker.commands), 1)
        self.assertEqual(self.runtime._debug_pickup["state"], "failed")


class SurveyTubesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings = load_settings(SETTINGS_DIR)
        self.moonraker = FakeMoonraker(ready=True)
        self.runtime = WorkflowRuntime(self.settings, moonraker=self.moonraker)
        await self.runtime.refresh_machine()
        self.moonraker.commands.clear()
        self.rack = self.settings.rack

    def _cap(self, world_xy, color, camera_xy, z):
        height = self.rack.cap_height_for_color(color)
        pixel = real_to_pixel(
            world_xy[0] - camera_xy[0],
            world_xy[1] - camera_xy[1],
            height_mm=height,
            camera_raise_mm=z - CALIBRATION_GANTRY_Z_MM,
        )
        return MergedCircle(*pixel, 30.0, color, 10)

    async def test_survey_populates_visible_slots_after_climbing(self) -> None:
        row, column = 2, 2
        slot = self.rack.tube_position(row - 1, column - 1)
        offset = self.rack.camera_offset_mm
        camera_xy = (slot.x - offset.x, slot.y - offset.y)
        z = self.rack.survey_height_mm
        red_slot = self.rack.tube_position(0, 0)
        white_slot = self.rack.tube_position(0, 1)
        camera = StaticCirclePreview([
            self._cap((red_slot.x + 2.0, red_slot.y - 1.0), "red", camera_xy, z),
            self._cap((white_slot.x - 1.0, white_slot.y + 2.0), "white", camera_xy, z),
        ])

        result = await self.runtime.survey_tubes(row, column, camera)  # type: ignore[arg-type]

        self.assertEqual(result["caps_seen"], 2)
        self.assertEqual(len(result["tubes_populated"]), 2)
        snapshot = await self.runtime.snapshot()
        colors = {(tube["row"], tube["column"]): tube["color"] for tube in snapshot["rack"]["tubes"]}
        self.assertEqual(colors[(1, 1)], "red")
        self.assertEqual(colors[(1, 2)], "white")
        red = snapshot["rack"]["tubes"][0]
        self.assertAlmostEqual(red["survey"]["offset_x_mm"], 2.0, places=6)
        self.assertAlmostEqual(red["survey"]["offset_y_mm"], -1.0, places=6)
        self.assertAlmostEqual(
            red["survey"]["cap_height_mm"],
            self.rack.cap_height_for_color("red"),
            places=6,
        )
        expected_tilt = float(np.degrees(np.arctan2(np.hypot(2.0, 1.0), self.rack.cap_height_for_color("red"))))
        self.assertAlmostEqual(red["survey"]["tilt_deg"], expected_tilt, places=5)
        self.assertIsNone(snapshot["rack"]["tubes"][2]["survey"])
        self.assertEqual(snapshot["rack"]["survey_height_mm"], z)
        motion = self.runtime.motion
        # Climb to the survey height before travelling over the rack.
        self.assertEqual(self.moonraker.commands, [
            "\n".join((
                motion.synchronized_move_command(z=z, feedrate=10000),
                motion.synchronized_move_command(x=camera_xy[0], y=camera_xy[1], feedrate=10000),
            )),
        ])
        self.assertTrue(camera.finished)

    async def test_survey_endpoint_returns_the_populated_tubes(self) -> None:
        row, column = 2, 2
        slot = self.rack.tube_position(row - 1, column - 1)
        offset = self.rack.camera_offset_mm
        camera_xy = (slot.x - offset.x, slot.y - offset.y)
        z = self.rack.survey_height_mm
        red_slot = self.rack.tube_position(0, 0)
        camera = StaticCirclePreview([self._cap((red_slot.x, red_slot.y), "red", camera_xy, z)])
        client = TestClient(TestServer(create_control_app(
            self.settings,
            runtime=self.runtime,
            camera_preview=camera,
            initialize_hardware=False,
        )))
        await client.start_server()
        try:
            response = await client.post(
                "/api/actions/survey-tubes",
                json={"row": row, "column": column},
            )
            payload = await response.json()
        finally:
            await client.close()
            await self.runtime.close()

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["action"], "camera.survey")
        self.assertEqual(payload["caps_seen"], 1)
        self.assertEqual(payload["tubes_populated"][0]["row"], 1)
        self.assertEqual(payload["tubes_populated"][0]["column"], 1)
        self.assertEqual(payload["tubes_populated"][0]["color"], "red")


if __name__ == "__main__":
    unittest.main()
