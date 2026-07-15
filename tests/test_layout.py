from __future__ import annotations

from pathlib import Path
import unittest

from controller.config.settings import load_settings
from controller.workflow.state_machine import TubeScanWorkflow


class LayoutScaffoldTests(unittest.TestCase):
    def test_load_settings_and_build_workflow(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        settings = load_settings(repo_root / "calibration")
        workflow = TubeScanWorkflow(settings)

        plan = workflow.build_plan()

        self.assertGreaterEqual(len(plan.steps), 5)
        self.assertEqual(plan.steps[0].name, "home")
        self.assertEqual(plan.steps[-1].name, "release")
        self.assertTrue(any(step.yaw_angle_deg is not None for step in plan.steps))
        self.assertEqual(settings.network.web.port, 8080)
        self.assertEqual(settings.network.moonraker.base_url, "http://127.0.0.1:7125")


if __name__ == "__main__":
    unittest.main()