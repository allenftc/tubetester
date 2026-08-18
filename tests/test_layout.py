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
        self.assertEqual(plan.steps[1].name, "approach_r1_c1")
        self.assertEqual(plan.steps[2].name, "pickup_r1_c1")
        self.assertTrue(any(step.name.startswith("scan_r1_c1_yaw_") for step in plan.steps))
        self.assertTrue(any(step.name == "release_r1_c1" for step in plan.steps))
        self.assertTrue(any(step.yaw_angle_deg is not None for step in plan.steps))
        self.assertEqual(settings.network.web.port, 8080)
        self.assertTrue(settings.network.moonraker.base_url.startswith(("http://", "https://")))
        self.assertEqual(settings.rack.rows * settings.rack.columns, 72)
        self.assertEqual(len(plan.steps), 1153)


if __name__ == "__main__":
    unittest.main()