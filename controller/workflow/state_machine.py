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
    x_mm: float | None = None
    y_mm: float | None = None
    z_mm: float | None = None
    row: int | None = None
    column: int | None = None


@dataclass(frozen=True)
class ScanPlan:
    steps: tuple[ScanStep, ...]


class TubeScanWorkflow:
    def __init__(self, settings: ControllerSettings) -> None:
        self.settings = settings

    def build_plan(self) -> ScanPlan:
        steps: list[ScanStep] = [ScanStep("home", "Home all axes")]

        for row in range(1, self.settings.rack.rows + 1):
            for column in range(1, self.settings.rack.columns + 1):
                target = self.settings.rack.tube_position(row - 1, column - 1)

                steps.append(
                    ScanStep(
                        name=f"approach_r{row}_c{column}",
                        description=f"Move above tube at row {row}, column {column}",
                        x_mm=target.x,
                        y_mm=target.y,
                        z_mm=self.settings.rack.safe_z_mm,
                        row=row,
                        column=column,
                    )
                )
                steps.append(
                    ScanStep(
                        name=f"pickup_r{row}_c{column}",
                        description=f"Lower, grip, and lift tube at row {row}, column {column}",
                        x_mm=target.x,
                        y_mm=target.y,
                        z_mm=target.z,
                        row=row,
                        column=column,
                    )
                )

                for yaw_angle in self.settings.yaw.sweep_angles():
                    steps.append(
                        ScanStep(
                            name=f"scan_r{row}_c{column}_yaw_{yaw_angle:g}",
                            description=f"Rotate tube at row {row}, column {column} to {yaw_angle:.1f} degrees for QR scan",
                            yaw_angle_deg=yaw_angle,
                            x_mm=target.x,
                            y_mm=target.y,
                            z_mm=target.z,
                            row=row,
                            column=column,
                        )
                    )

                steps.append(
                    ScanStep(
                        name=f"release_r{row}_c{column}",
                        description=f"Release tube at row {row}, column {column} and return to safe position",
                        x_mm=target.x,
                        y_mm=target.y,
                        z_mm=self.settings.rack.safe_z_mm,
                        row=row,
                        column=column,
                    )
                )

        return ScanPlan(steps=tuple(steps))

    def describe(self) -> list[str]:
        plan = self.build_plan()
        return [f"{step.name}: {step.description}" for step in plan.steps]