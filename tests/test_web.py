from __future__ import annotations

import asyncio
from dataclasses import replace
import time
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from controller.config.settings import load_settings
from controller.network.moonraker import MoonrakerResponse
from controller.web.runtime import RuntimeConflict, RuntimeUnavailable, WorkflowRuntime
from controller.web.server import create_control_app


class FakeMoonraker:
    def __init__(self, *, ready: bool = False, delay: float = 0.0) -> None:
        self.ready = ready
        self.delay = delay
        self.commands: list[str] = []

    def get_printer_info(self) -> MoonrakerResponse:
        if not self.ready:
            return MoonrakerResponse(False, 0, error_message="offline")
        return MoonrakerResponse(True, 200, payload={"result": {"state": "ready", "state_message": "Printer is ready"}})

    def send_gcode(self, script: str) -> MoonrakerResponse:
        self.commands.append(script)
        if self.delay:
            time.sleep(self.delay)
        return MoonrakerResponse(self.ready, 200 if self.ready else 0, error_message=None if self.ready else "offline")


class FakeQr:
    async def decode(self, row: int, column: int, yaw_angle_deg: float) -> None:
        return None


class FakeCameraPreview:
    def __init__(self) -> None:
        self.started = False
        self.finished = False

    def status(self) -> dict[str, object]:
        return {
            "available": self.started and not self.finished,
            "state": "streaming" if self.started and not self.finished else "stopped",
            "message": "Live camera preview." if self.started and not self.finished else "Camera preview is off.",
            "device": "/dev/video0",
            "width": 320,
            "height": 240,
            "roi": {"center_x": 160, "center_y": 120, "width": 100, "height": 100},
            "detected_center": None,
            "tube_present": False,
            "sequence": 1,
        }

    def wait_for_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        if after_sequence == 0:
            return 1, b"\xff\xd8fake-jpeg\xff\xd9"
        self.finished = True
        return 1, None

    def start(self) -> None:
        self.started = True
        self.finished = False

    def stop(self) -> None:
        self.started = False
        self.finished = True


class FakeDetectionCameraPreview(FakeCameraPreview):
    def __init__(self, detected_center: list[int] | None) -> None:
        super().__init__()
        self.detected_center = detected_center
        self.detection_after_sequence: int | None = None

    def wait_for_frame(self, after_sequence: int, timeout_seconds: float = 5.0) -> tuple[int, bytes | None]:
        return 1, b"frame"

    def wait_for_detection(self, after_sequence: int, timeout_seconds: float = 8.0) -> dict[str, object] | None:
        self.detection_after_sequence = after_sequence
        if self.detected_center is None:
            return None
        return {
            "detected_center": self.detected_center,
            "roi": {"center_x": 160, "center_y": 120, "width": 80, "height": 80},
        }


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        self.runtime = WorkflowRuntime(self.settings, moonraker=FakeMoonraker())
        self.client = TestClient(TestServer(create_control_app(self.settings, runtime=self.runtime, initialize_hardware=False)))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()

    async def test_index_static_and_normalized_offline_status(self) -> None:
        index = await self.client.get("/")
        html = await index.text()
        self.assertEqual(index.status, 200)
        for landmark in ("Home Machine", "Preview Run", "Start Scan", "Rack 6 × 12", "Begin Pickup Steps", "Run Next Step", "Cancel Steps", "Console", "Send code"):
            self.assertIn(landmark, html)
        self.assertIn('id="camera-preview-toggle" type="checkbox"', html)
        self.assertIn('id="camera-stream" alt=', html)
        self.assertNotIn('src="/api/camera/stream"', html)
        self.assertIn('id="locate-tube-plan-steps"', html)
        self.assertIn('id="locate-tube-steps"', html)
        self.assertNotIn("<pre", html)
        self.assertNotIn("Workflow Preview</h2>", html)

        css = await self.client.get("/static/app.css")
        self.assertEqual(css.status, 200)
        status = await (await self.client.get("/api/status")).json()
        self.assertEqual(status["schema_version"], 1)
        self.assertFalse(status["machine"]["connected"])
        self.assertFalse(status["capabilities"]["start"])
        self.assertFalse(status["capabilities"]["degraded_mode"])
        self.assertEqual(len(status["rack"]["tubes"]), 72)
        self.assertEqual(status["rack"]["pickup_height_mm"], self.settings.rack.pickup_height_mm)
        self.assertNotIn("plan", status["workflow"])
        self.assertNotIn("moonraker", status)

    async def test_preview_is_bounded_and_gcode_validation(self) -> None:
        preview = await self.client.post("/api/actions/preview", json={})
        body = await preview.json()
        self.assertEqual(preview.status, 200)
        self.assertEqual(body["plan"]["tube_count"], 72)
        self.assertEqual(body["plan"]["step_count"], 1153)
        self.assertNotIn("steps", body["plan"])
        invalid = await self.client.post("/api/gcode", json={"script": " "})
        self.assertEqual(invalid.status, 400)

    async def test_camera_status_and_unavailable_stream(self) -> None:
        status = await (await self.client.get("/api/camera/status")).json()
        self.assertFalse(status["available"])
        self.assertEqual(status["state"], "not_started")
        self.assertEqual(status["roi"]["center_x"], self.settings.camera.roi_center_x)
        stream = await self.client.get("/api/camera/stream")
        self.assertEqual(stream.status, 409)

    async def test_camera_preview_toggle_starts_and_stops_capture(self) -> None:
        preview = FakeCameraPreview()
        client = TestClient(TestServer(create_control_app(
            self.settings,
            runtime=self.runtime,
            camera_preview=preview,
            initialize_hardware=False,
        )))
        await client.start_server()
        try:
            enabled = await client.post("/api/camera/preview", json={"enabled": True})
            self.assertEqual(enabled.status, 200)
            self.assertTrue((await enabled.json())["enabled"])
            self.assertTrue(preview.started)

            disabled = await client.post("/api/camera/preview", json={"enabled": False})
            self.assertEqual(disabled.status, 200)
            self.assertFalse((await disabled.json())["enabled"])
            self.assertFalse(preview.started)
        finally:
            await client.close()

    async def test_camera_stream_emits_multipart_jpeg_frames(self) -> None:
        preview = FakeCameraPreview()
        client = TestClient(TestServer(create_control_app(
            self.settings,
            runtime=self.runtime,
            camera_preview=preview,
            initialize_hardware=False,
        )))
        await client.start_server()
        try:
            enabled = await client.post("/api/camera/preview", json={"enabled": True})
            self.assertEqual(enabled.status, 200)
            response = await client.get("/api/camera/stream")
            body = await response.read()
            self.assertEqual(response.status, 200)
            self.assertTrue(response.headers["Content-Type"].startswith("multipart/x-mixed-replace"))
            self.assertIn(b"--frame", body)
            self.assertIn(b"\xff\xd8fake-jpeg\xff\xd9", body)
        finally:
            await client.close()

    async def test_locate_tube_endpoint_uses_selected_grid_cell(self) -> None:
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(self.settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview([160, 120])
        client = TestClient(TestServer(create_control_app(
            self.settings,
            runtime=runtime,
            camera_preview=camera,  # type: ignore[arg-type]
            initialize_hardware=False,
        )))
        await client.start_server()
        try:
            response = await client.post("/api/actions/locate-tube", json={"row": 2, "column": 3})
            payload = await response.json()
            self.assertEqual(response.status, 200)
            self.assertEqual(payload["action"], "camera.pickup.stepper")
            self.assertEqual(payload["state"], "running")
            tube = self.settings.rack.tube_position(1, 2)
            self.assertEqual(payload["tube_xy"], {"x": tube.x, "y": tube.y})
            self.assertEqual(moonraker.commands, [])
            active_capabilities = (await runtime.snapshot())["capabilities"]
            self.assertFalse(active_capabilities["send_gcode"])
            self.assertFalse(active_capabilities["tooling"])
            resumed_snapshot = await (await client.get("/api/status")).json()
            self.assertEqual(resumed_snapshot["camera_pickup"]["session_id"], payload["session_id"])
            self.assertEqual(resumed_snapshot["camera_pickup"]["next_step"], 0)
            blocked_camera_stop = await client.post("/api/camera/preview", json={"enabled": False})
            self.assertEqual(blocked_camera_stop.status, 409)
            with self.assertRaises(RuntimeConflict):
                await runtime.start_vacuum()

            for _ in range(7):
                response = await client.post("/api/actions/locate-tube/step", json={"session_id": payload["session_id"]})
                payload = await response.json()
                self.assertEqual(response.status, 200)
            self.assertEqual(payload["state"], "completed")
            self.assertEqual(payload["target_xy"], {"x": tube.x, "y": tube.y})
            self.assertTrue((await runtime.snapshot())["capabilities"]["send_gcode"])
            self.assertFalse(camera.started)
        finally:
            await client.close()

    async def test_locate_tube_cancel_endpoint_lifts_before_unlocking(self) -> None:
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(self.settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview([160, 120])
        client = TestClient(TestServer(create_control_app(
            self.settings,
            runtime=runtime,
            camera_preview=camera,  # type: ignore[arg-type]
            initialize_hardware=False,
        )))
        await client.start_server()
        try:
            response = await client.post("/api/actions/locate-tube", json={"row": 1, "column": 1})
            session = await response.json()
            for _ in range(6):
                response = await client.post("/api/actions/locate-tube/step", json={"session_id": session["session_id"]})
                session = await response.json()

            response = await client.post("/api/actions/locate-tube/cancel", json={"session_id": session["session_id"]})
            cancelled = await response.json()
            self.assertEqual(response.status, 200)
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertTrue(cancelled["vacuum_enabled"])
            self.assertEqual(moonraker.commands[-1], f"G90\nG1 Z{self.settings.rack.safe_z_mm:.3f} F5000\nM400")
            self.assertTrue((await runtime.snapshot())["capabilities"]["send_gcode"])
        finally:
            await client.close()

    async def test_tooling_endpoints_send_klipper_commands(self) -> None:
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(self.settings, moonraker=moonraker)
        await runtime.initialize()
        client = TestClient(TestServer(create_control_app(self.settings, runtime=runtime, initialize_hardware=False)))
        await client.start_server()
        try:
            for endpoint, body, action in (
                ("/api/tooling/release", {}, "tooling.release"),
                ("/api/tooling/vacuum", {}, "tooling.vacuum"),
                ("/api/tooling/vacuum/off", {}, "tooling.vacuum_off"),
                ("/api/tooling/rotary/zero", {}, "tooling.rotary.zero"),
                ("/api/tooling/rotary/move", {"degrees": 45}, "tooling.rotary.move"),
            ):
                response = await client.post(endpoint, json=body)
                payload = await response.json()
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["action"], action)

            invalid = await client.post("/api/tooling/rotary/move", json={"degrees": 45.5})
            self.assertEqual(invalid.status, 400)
        finally:
            await client.close()

        self.assertEqual(moonraker.commands, [
            "SET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0",
            "SET_PIN PIN=solenoid VALUE=1\nG4 P500\nSET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0",
            "SET_PIN PIN=vacuum_pump VALUE=1",
            "SET_PIN PIN=vacuum_pump VALUE=0",
            "MANUAL_STEPPER STEPPER=rotary SET_POSITION=0.00",
            "MANUAL_STEPPER STEPPER=rotary MOVE=45.00",
        ])

    async def test_combined_macro_endpoints(self) -> None:
        ready_moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(self.settings, moonraker=ready_moonraker)
        await runtime.refresh_machine()
        client = TestClient(TestServer(create_control_app(self.settings, runtime=runtime, initialize_hardware=False)))
        await client.start_server()
        try:
            for endpoint, action in (
                ("/api/macros/calibrate", "macro.calibrate"),
                ("/api/macros/pickup", "macro.pickup"),
                ("/api/macros/deposit", "macro.deposit"),
            ):
                response = await client.post(endpoint, json={})
                payload = await response.json()
                self.assertEqual(response.status, 200)
                self.assertEqual(payload["action"], action)
        finally:
            await client.close()

    async def test_websocket_delivers_snapshot(self) -> None:
        socket = await self.client.ws_connect("/ws")
        hello = await socket.receive_json()
        snapshot = await socket.receive_json()
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(snapshot["type"], "status.snapshot")
        await socket.close()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_locate_tube_rejects_pickup_height_at_safe_z_before_motion(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        settings = replace(settings, rack=replace(settings.rack, pickup_height_mm=settings.rack.safe_z_mm))
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview([160, 120])

        with self.assertRaisesRegex(RuntimeUnavailable, "pickup_height_mm must be finite and below safe_z_mm"):
            await runtime.begin_camera_pickup(1, 1, camera)  # type: ignore[arg-type]

        self.assertEqual(moonraker.commands, [])
        self.assertFalse(camera.started)

    async def test_locate_tube_steps_one_command_per_click_and_keeps_vacuum_on(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview([170, 115])
        tube = settings.rack.tube_position(0, 0)

        session = await runtime.begin_camera_pickup(1, 1, camera)  # type: ignore[arg-type]
        self.assertEqual(session["state"], "running")
        self.assertEqual(session["next_step"], 0)
        self.assertEqual(moonraker.commands, [])

        for step_index in range(7):
            previous_command_count = len(moonraker.commands)
            session = await runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
            expected_added_commands = 0 if step_index == 2 else 1
            self.assertEqual(len(moonraker.commands) - previous_command_count, expected_added_commands)
            self.assertEqual(session["next_step"], step_index + 1)

        self.assertEqual(session["action"], "camera.pickup.stepper")
        self.assertEqual(session["state"], "completed")
        self.assertEqual(session["target_xy"], {
            "x": tube.x + 10 * settings.rack.pixel_to_mm_multiplier,
            "y": tube.y - 5 * settings.rack.pixel_to_mm_multiplier,
        })
        self.assertEqual(moonraker.commands, [
            f"G90\nG1 Z{settings.rack.safe_z_mm:.3f} F10000\nM400",
            f"G90\nG1 X{tube.x - settings.rack.camera_offset_mm.x:.3f} Y{tube.y - settings.rack.camera_offset_mm.y:.3f} F10000\nM400",
            f"G90\nG1 X{session['target_xy']['x']:.3f} Y{session['target_xy']['y']:.3f} F5000\nM400",
            "SET_PIN PIN=vacuum_pump VALUE=1",
            f"G90\nG1 Z{settings.rack.pickup_height_mm:.3f} F3000\nM400",
            f"G90\nG1 Z{settings.rack.safe_z_mm:.3f} F5000\nM400",
        ])
        self.assertEqual(len(session["steps"]), 7)
        self.assertTrue(all(step["state"] == "completed" for step in session["steps"]))
        self.assertTrue(session["vacuum_enabled"])
        self.assertFalse(camera.started)

    async def test_locate_tube_does_not_correct_gripper_without_detection(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview(None)

        session = await runtime.begin_camera_pickup(1, 1, camera)  # type: ignore[arg-type]
        await runtime.run_camera_pickup_step(session["session_id"], camera)  # safe Z
        await runtime.run_camera_pickup_step(session["session_id"], camera)  # camera XY
        with self.assertRaisesRegex(RuntimeUnavailable, "No tube center detected"):
            await runtime.run_camera_pickup_step(session["session_id"], camera)

        self.assertEqual(len(moonraker.commands), 2)
        self.assertEqual(runtime._debug_pickup["state"], "failed")
        self.assertFalse((await runtime.snapshot())["capabilities"]["send_gcode"])
        await runtime.cancel_camera_pickup(session["session_id"], camera)
        self.assertTrue((await runtime.snapshot())["capabilities"]["send_gcode"])
        self.assertFalse(camera.started)

    async def test_cancel_after_descent_lifts_to_safe_z_and_keeps_vacuum_on(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()
        moonraker.commands.clear()
        camera = FakeDetectionCameraPreview([160, 120])
        session = await runtime.begin_camera_pickup(1, 1, camera)  # type: ignore[arg-type]

        for _ in range(6):
            session = await runtime.run_camera_pickup_step(session["session_id"], camera)  # type: ignore[arg-type]
        self.assertTrue(session["vacuum_enabled"])

        cancelled = await runtime.cancel_camera_pickup(session["session_id"], camera)  # type: ignore[arg-type]

        self.assertEqual(cancelled["state"], "cancelled")
        self.assertTrue(cancelled["vacuum_enabled"])
        self.assertEqual(moonraker.commands[-1], f"G90\nG1 Z{settings.rack.safe_z_mm:.3f} F5000\nM400")
        self.assertFalse(camera.started)

    async def test_combined_macros_execute_klipper_tooling_steps_in_order(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()

        calibration = await runtime.calibrate_rotary_macro()
        self.assertEqual(calibration["action"], "macro.calibrate")
        self.assertIn("MANUAL_STEPPER STEPPER=rotary SET_POSITION=0.00", moonraker.commands)

        pickup = await runtime.pickup_macro()
        self.assertEqual(pickup["action"], "macro.pickup")
        self.assertIn("SET_PIN PIN=solenoid VALUE=1\nG4 P500\nSET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0", moonraker.commands)
        self.assertIn("SET_PIN PIN=vacuum_pump VALUE=1", moonraker.commands)

        deposit = await runtime.deposit_macro()
        self.assertEqual(deposit["action"], "macro.deposit")
        self.assertEqual(moonraker.commands[-3:], [
            "G90\nG1 Z20.000 F10000\nM400",
            "SET_PIN PIN=solenoid VALUE=1\nG4 P500\nSET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0",
            "G90\nG1 Z50.000 F10000\nM400",
        ])

    async def test_background_start_and_duplicate_guard(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True, delay=0.04)
        runtime = WorkflowRuntime(settings, moonraker=moonraker, qr_backend=FakeQr())
        await runtime.refresh_machine()
        started = time.monotonic()
        response = await runtime.start(selection=[(1, 1)])
        elapsed = time.monotonic() - started
        self.assertEqual(response["action"], "workflow.start")
        self.assertLess(elapsed, 0.03)
        with self.assertRaises(RuntimeConflict):
            await runtime.start(selection=[(1, 1)])
        await runtime.stop()
        await asyncio.sleep(0.12)
        snapshot = await runtime.snapshot()
        self.assertIn(snapshot["workflow"]["state"], {"stopping", "stopped"})
        await runtime.close()

    async def test_degraded_mode_keeps_tooling_active_without_qr(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True)
        runtime = WorkflowRuntime(settings, moonraker=moonraker)
        await runtime.refresh_machine()
        snapshot = await runtime.snapshot()
        self.assertTrue(snapshot["capabilities"]["start"])
        self.assertTrue(snapshot["capabilities"]["degraded_mode"])
        await runtime.start(selection=[(1, 1)], degraded_mode=True)
        if runtime._task:
            await runtime._task
        snapshot = await runtime.snapshot()
        self.assertEqual(snapshot["workflow"]["state"], "completed")
        self.assertEqual(snapshot["rack"]["tubes"][0]["status"], "released_without_decode")
        self.assertEqual(len(moonraker.commands), 6)
        self.assertEqual(moonraker.commands[0], "SET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0")
        self.assertEqual(moonraker.commands[1], "G28")
        self.assertTrue(moonraker.commands[2].startswith("G90\nG1 Z100.000"))
        self.assertIn("SET_PIN PIN=vacuum_pump VALUE=1", moonraker.commands[3])
        self.assertIn("SET_PIN PIN=solenoid VALUE=1\nG4 P500\nSET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0", moonraker.commands[4])
        self.assertEqual(moonraker.commands[5], "SET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0")
        await runtime.close()


if __name__ == "__main__":
    unittest.main()
