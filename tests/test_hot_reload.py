from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from controller.config.settings import load_settings
from controller.hot_reload import SourceWatcher, touched_vision
from controller.network.moonraker import MoonrakerResponse
from controller.web.runtime import RuntimeConflict, WorkflowRuntime

CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


class FakeMoonraker:
    def get_printer_info(self) -> MoonrakerResponse:
        return MoonrakerResponse(True, 200, payload={"result": {"state": "ready", "state_message": "ready"}})

    def send_gcode(self, script: str) -> MoonrakerResponse:
        return MoonrakerResponse(True, 200)


class SourceWatcherTests(unittest.TestCase):
    def test_poll_separates_calibration_from_source_edits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "controller" / "vision").mkdir(parents=True)
            (root / "calibration").mkdir()
            source = root / "controller" / "vision" / "tube_locator.py"
            calibration = root / "calibration" / "rack.json"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            calibration.write_text("{}\n", encoding="utf-8")

            watcher = SourceWatcher([root / "calibration"], root)
            self.assertTrue(watcher.poll().empty)

            _bump(source)
            changes = watcher.poll()
            self.assertTrue(changes.empty is False)
            self.assertEqual(changes.code_paths, (source,))
            self.assertEqual(changes.config_paths, ())

            _bump(calibration)
            changes = watcher.poll()
            self.assertEqual(changes.config_paths, (calibration,))
            self.assertEqual(changes.code_paths, ())

    def test_touched_vision_flags_camera_code_only(self) -> None:
        root = Path("/repo")
        self.assertTrue(touched_vision([root / "controller" / "vision" / "camera_preview.py"], root))
        self.assertFalse(touched_vision([root / "controller" / "web" / "actions.py"], root))
        self.assertFalse(touched_vision([Path("/elsewhere/calibration/rack.json")], root))


class SettingsReloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_apply_settings_resizes_the_rack_and_updates_snapshot(self) -> None:
        settings = load_settings(CALIBRATION_DIR)
        runtime = WorkflowRuntime(settings, moonraker=FakeMoonraker())
        resized = replace(
            settings,
            rack=replace(settings.rack, rows=settings.rack.rows + 1, columns=2),
        )

        await runtime.apply_settings(resized)
        snapshot = await runtime.snapshot()

        self.assertEqual(snapshot["rack"]["rows"], resized.rack.rows)
        self.assertEqual(snapshot["rack"]["columns"], 2)
        self.assertEqual(len(snapshot["rack"]["tubes"]), resized.rack.rows * 2)
        self.assertEqual(snapshot["workflow"]["progress"]["total_tubes"], resized.rack.rows * 2)
        await runtime.close()

    async def test_apply_settings_is_refused_while_the_workflow_runs(self) -> None:
        settings = load_settings(CALIBRATION_DIR)
        runtime = WorkflowRuntime(settings, moonraker=FakeMoonraker())
        runtime._controller._workflow["state"] = "running"  # simulate an active scan

        with self.assertRaises(RuntimeConflict):
            await runtime.apply_settings(replace(settings, rack=replace(settings.rack, rows=2)))
        await runtime.close()


def _bump(path: Path) -> None:
    """Move the mtime forward so the change is visible even on coarse clocks."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))


if __name__ == "__main__":
    unittest.main()
