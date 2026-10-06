from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KlipperMotionClient:
    """Builds the Klipper commands used by the scanner hardware."""

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
        parts = ["G90\nG1"]
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
        return "SET_PIN PIN=vacuum_pump VALUE=1"

    def vacuum_off_command(self) -> str:
        return "SET_PIN PIN=vacuum_pump VALUE=0"

    def release_command(self) -> str:
        return "SET_PIN PIN=solenoid VALUE=1\nG4 P500\nSET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0"

    def set_yaw_command(self, angle_deg: float) -> str:
        return f"MANUAL_STEPPER STEPPER=rotary MOVE={angle_deg:.2f}"

    def initialize_tooling_command(self) -> str:
        return "SET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0"

    def shutdown_tooling_command(self) -> str:
        return "SET_PIN PIN=solenoid VALUE=0\nSET_PIN PIN=vacuum_pump VALUE=0"

    def set_rotary_position_command(self, degrees: float = 0) -> str:
        return f"MANUAL_STEPPER STEPPER=rotary SET_POSITION={degrees:.2f}"

    def synchronized_move_command(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        feedrate: float,
    ) -> str:
        """Move and wait until Klipper completes the physical motion."""
        return f"{self.move_command(x=x, y=y, z=z, feedrate=feedrate)}\nM400"

    def safe_z_command(self) -> str:
        return self.safe_z_macro