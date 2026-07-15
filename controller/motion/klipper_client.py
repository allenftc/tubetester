from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KlipperMotionClient:
    """Builds the Klipper commands used by the workflow scaffold."""

    pickup_macro: str = "TUBE_PICKUP"
    release_macro: str = "TUBE_RELEASE"
    yaw_macro: str = "TUBE_SET_YAW"
    safe_z_macro: str = "TUBE_SAFE_Z"

    def home_command(self) -> str:
        return "G28"

    def move_command(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        feedrate: float | None = None,
    ) -> str:
        parts = ["G1"]
        if x is not None:
            parts.append(f"X{x:.3f}")
        if y is not None:
            parts.append(f"Y{y:.3f}")
        if z is not None:
            parts.append(f"Z{z:.3f}")
        if feedrate is not None:
            parts.append(f"F{feedrate:.0f}")
        return " ".join(parts)

    def pickup_command(self) -> str:
        return self.pickup_macro

    def release_command(self) -> str:
        return self.release_macro

    def set_yaw_command(self, angle_deg: float) -> str:
        return f"{self.yaw_macro} ANGLE={angle_deg:.2f}"

    def safe_z_command(self) -> str:
        return self.safe_z_macro