from __future__ import annotations

import asyncio
import time
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from controller.config.settings import load_settings
from controller.network.moonraker import MoonrakerResponse
from controller.web.runtime import RuntimeConflict, WorkflowRuntime
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
        for landmark in ("Home Machine", "Preview Run", "Start Scan", "Rack 6 × 12", "Console", "Send code"):
            self.assertIn(landmark, html)
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

    async def test_websocket_delivers_snapshot(self) -> None:
        socket = await self.client.ws_connect("/ws")
        hello = await socket.receive_json()
        snapshot = await socket.receive_json()
        self.assertEqual(hello["type"], "hello")
        self.assertEqual(snapshot["type"], "status.snapshot")
        await socket.close()


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_start_and_duplicate_guard(self) -> None:
        settings = load_settings(Path(__file__).resolve().parents[1] / "calibration")
        moonraker = FakeMoonraker(ready=True, delay=0.04)
        runtime = WorkflowRuntime(settings, moonraker=moonraker, qr_backend=FakeQr(), macros_available=True)
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

    async def test_degraded_mode_skips_unavailable_macro_and_camera_steps(self) -> None:
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
        self.assertEqual(len(moonraker.commands), 2)
        self.assertEqual(moonraker.commands[0], "G28")
        self.assertTrue(moonraker.commands[1].startswith("G1 Z100.000"))
        self.assertFalse(any("TUBE_" in command for command in moonraker.commands))
        await runtime.close()


if __name__ == "__main__":
    unittest.main()
