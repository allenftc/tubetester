from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from controller.config.settings import ControllerSettings


class WorkflowPhase(str, Enum):
    HOME = "home"
    APPROACH = "approach"
    PICKUP = "pickup"
    SCAN = "scan"
    RELEASE = "release"


@dataclass(frozen=True)
class ScanStep:
    name: str
    description: str
    yaw_angle_deg: float | None = None


@dataclass(frozen=True)
class ScanPlan:
    steps: tuple[ScanStep, ...]


class TubeScanWorkflow:
    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings

    def build_plan(self) -> ScanPlan:
        steps = [
            ScanStep("home", "Home all axes"),
            ScanStep("approach", "Move above the target tube"),
            ScanStep("pickup", "Lower, grip, and lift the tube"),
        ]

        for yaw_angle in self.settings.yaw.sweep_angles():
            steps.append(
                ScanStep(
                    name=f"scan_yaw_{yaw_angle:g}",
                    description=f"Rotate tube to {yaw_angle:.1f} degrees for QR scan",
                    yaw_angle_deg=yaw_angle,
                )
            )

        steps.append(ScanStep("release", "Release the tube and return to safe position"))
        return ScanPlan(steps=tuple(steps))

    def describe(self) -> list[str]:
        plan = self.build_plan()
        return [f"{step.name}: {step.description}" for step in plan.steps]