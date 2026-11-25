from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any

from controller.motion.klipper_client import (
    ROTARY_MAX_DEG,
    ROTARY_MIN_DEG,
    ROTARY_SWEEP_SPEED,
)
from controller.network.moonraker import MoonrakerResponse
from controller.vision.camera_preview import CameraPreviewService
from controller.vision.tubeLocator.cap_matching import LocatedCap, assign_caps_to_slots, locate_caps, select_cap
from controller.vision.tubeLocator.tube_locator import (
    STEREO_BASELINE_MM,
    triangulate_stereo_circles,
)
from controller.web.events import utc_timestamp
from controller.web.errors import RuntimeConflict, RuntimeUnavailable

class MachineActions:
    """Web-triggered machine and tooling controls."""

    async def home(self) -> dict[str, Any]:
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["home"]:
            raise RuntimeUnavailable("Klipper must be ready and the workflow inactive before homing.")
        return await self._send_action("machine.home", "G28", source="user")

    async def send_gcode(self, script: str) -> dict[str, Any]:
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["send_gcode"]:
            raise RuntimeUnavailable("Klipper must be ready and the workflow inactive to send G-code.")
        return await self._send_action("gcode.send", script, source="user")

    async def restart_klipper(self) -> dict[str, Any]:
        """Run FIRMWARE_RESTART through Moonraker to recover a stopped/lost MCU."""
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["restart_klipper"]:
            raise RuntimeUnavailable(
                "Moonraker must be reachable and the machine idle before Klipper can be restarted."
            )
        correlation_id = f"req_{uuid.uuid4().hex}"
        self.events.publish(
            "Requesting Klipper firmware restart.",
            source="user",
            correlation_id=correlation_id,
        )
        try:
            response = await asyncio.to_thread(self.moonraker.firmware_restart)
        except Exception as exc:
            response = MoonrakerResponse(False, 0, error_message=str(exc))
        if not response.ok:
            message = response.error_message or "Moonraker rejected the firmware restart."
            self.events.publish(message, source="moonraker", level="error", correlation_id=correlation_id)
            raise RuntimeUnavailable(message)
        async with self._lock:
            # Force the tooling initialization to run again once Klipper is ready.
            self._hardware_initialized = False
            # A firmware restart resets the manual stepper's tracked position.
            self._machine["rotary_angle_deg"] = 0.0
        self.events.publish("Firmware restart accepted by Moonraker.", source="moonraker", correlation_id=correlation_id)
        # Briefly wait for the state transition so the dashboard sees the result.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while True:
            await asyncio.sleep(0.5)
            await self.refresh_machine()
            if self._machine["klipper_state"] == "ready" or loop.time() >= deadline:
                break
        state = self._machine["klipper_state"]
        message = (
            "Klipper is ready again. Re-home the axes before moving."
            if state == "ready"
            else f"Firmware restart requested; Klipper is currently {state}."
        )
        return self._action("machine.firmware_restart", message)

    async def release_tube(self) -> dict[str, Any]:
        return await self._send_action("tooling.release", self.motion.release_command(), source="klipper")

    async def start_vacuum(self) -> dict[str, Any]:
        return await self._send_action("tooling.vacuum", self.motion.pickup_command(), source="klipper")

    async def stop_vacuum(self) -> dict[str, Any]:
        return await self._send_action("tooling.vacuum_off", self.motion.vacuum_off_command(), source="klipper")

    async def zero_rotary(self) -> dict[str, Any]:
        result = await self._send_action(
            "tooling.rotary.zero",
            self.motion.set_rotary_position_command(),
            source="klipper",
        )
        self._remember_rotary_angle(0.0)
        self._broadcast_status()
        return result

    async def rotate_to_position(self, degrees: int) -> dict[str, Any]:
        if isinstance(degrees, bool) or not isinstance(degrees, int):
            raise ValueError("degrees must be an integer")
        if not ROTARY_MIN_DEG <= degrees <= ROTARY_MAX_DEG:
            raise ValueError(
                f"degrees must be between {ROTARY_MIN_DEG:.0f} and {ROTARY_MAX_DEG:.0f}"
            )
        result = await self._send_action(
            "tooling.rotary.move",
            self.motion.set_yaw_command(degrees),
            source="klipper",
        )
        self._remember_rotary_angle(float(degrees))
        self._broadcast_status()
        return result

    async def sweep_rotary(self) -> dict[str, Any]:
        """Slowly sweep the rotary to the opposite end of its travel.

        The direction follows the current firmware position (falling back to the
        last commanded angle), so repeated presses alternate between -180 and
        +180 while the rotary always stays inside that range.
        """
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["tooling"]:
            raise RuntimeUnavailable(
                "Klipper must be ready and the workflow inactive before sweeping the rotary."
            )
        current = await self._query_rotary_angle()
        if current is None:
            current = float(self._machine.get("rotary_angle_deg", 0.0))
        current = max(ROTARY_MIN_DEG, min(ROTARY_MAX_DEG, current))
        target = ROTARY_MIN_DEG if current > 0 else ROTARY_MAX_DEG
        result = await self._send_action(
            "tooling.rotary.sweep",
            self.motion.sweep_rotary_command(target),
            source="klipper",
        )
        self._remember_rotary_angle(target)
        self._broadcast_status()
        result["message"] = (
            f"Rotary sweeping from {current:.1f}° to {target:.1f}° "
            f"at speed {ROTARY_SWEEP_SPEED:.0f}."
        )
        return result

    async def _query_rotary_angle(self) -> float | None:
        """Read the live rotary position when Moonraker supports it."""
        getter = getattr(self.moonraker, "get_manual_stepper_position", None)
        if getter is None:
            return None
        try:
            response = await asyncio.to_thread(getter)
        except Exception:
            return None
        if not response.ok:
            return None
        result = (response.payload or {}).get("result", response.payload or {})
        stepper = (result.get("status") or {}).get("manual_stepper rotary") or {}
        try:
            angle = float(stepper.get("position"))
        except (TypeError, ValueError):
            return None
        return angle if math.isfinite(angle) else None

    def _remember_rotary_angle(self, angle: float) -> None:
        self._machine["rotary_angle_deg"] = float(angle)

    async def calibrate_rotary_macro(self) -> dict[str, Any]:
        result = await self._run_manual_macro(
            "macro.calibrate",
            "Calibration macro completed.",
            (
                ("gcode", self.motion.synchronized_move_command(z=150, feedrate=10000)),
                ("gcode", self.motion.synchronized_move_command(x=200, feedrate=15000)),
                ("gcode", self.motion.synchronized_move_command(y=0, feedrate=15000)),
                ("gcode", self.motion.synchronized_move_command(z=0, feedrate=10000)),
                ("gcode", self.motion.synchronized_move_command(x=239, feedrate=1800)),
                ("gcode", self.motion.set_rotary_position_command()),
                ("gcode", self.motion.synchronized_move_command(z=100, feedrate=15000)),
            ),
        )
        self._remember_rotary_angle(0.0)
        self._broadcast_status()
        return result

    async def pickup_macro(self) -> dict[str, Any]:
        return await self._run_manual_macro(
            "macro.pickup",
            "Pickup macro completed.",
            (
                ("gcode", self.motion.release_command()),
                ("gcode", self.motion.synchronized_move_command(z=0, feedrate=10000)),
                ("gcode", self.motion.pickup_command()),
                ("gcode", self.motion.synchronized_move_command(z=150, feedrate=10000)),
            ),
        )

    async def deposit_macro(self) -> dict[str, Any]:
        return await self._run_manual_macro(
            "macro.deposit",
            "Deposit macro completed.",
            (
                ("gcode", self.motion.synchronized_move_command(z=50, feedrate=10000)),
                ("gcode", self.motion.release_command()),
                ("gcode", self.motion.synchronized_move_command(z=50, feedrate=10000)),
            ),
        )

    async def _send_action(self, action: str, script: str, source: str) -> dict[str, Any]:
        if self._stereo_test_active:
            raise RuntimeConflict("Machine controls are locked while the standalone stereo test is running.")
        if self.debug_pickup_active:
            raise RuntimeConflict("Machine controls are locked while a camera-guided pickup is being stepped.")
        correlation_id = f"req_{uuid.uuid4().hex}"
        self.events.publish(script, source=source, command=script, correlation_id=correlation_id)
        try:
            response = await asyncio.to_thread(self.moonraker.send_gcode, script)
        except Exception as exc:
            response = MoonrakerResponse(False, 0, error_message=str(exc))
        if not response.ok:
            message = response.error_message or "Moonraker rejected the command."
            self.events.publish(message, source="moonraker", level="error", correlation_id=correlation_id)
            raise RuntimeUnavailable(message)
        self.events.publish("Command accepted by Moonraker.", source="moonraker", correlation_id=correlation_id)
        return self._action(action, "Command accepted.")

    async def _run_manual_macro(
        self,
        action: str,
        message: str,
        steps: tuple[tuple[str, str], ...],
    ) -> dict[str, Any]:
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["send_gcode"]:
            raise RuntimeUnavailable("Klipper must be ready and the workflow inactive before running a tooling macro.")
        correlation_id = f"req_{uuid.uuid4().hex}"
        async with self._manual_macro_lock:
            if self._stereo_test_active:
                raise RuntimeConflict("Machine controls are locked while the standalone stereo test is running.")
            self.events.publish(f"Starting {action}.", source="user", correlation_id=correlation_id)
            for kind, value in steps:
                if kind == "gcode":
                    await self._send_macro_gcode(value, correlation_id)
                else:  # defensive validation for internal macro definitions
                    raise RuntimeError(f"Unknown manual macro step type: {kind}")
        self.events.publish(message, source="controller", correlation_id=correlation_id)
        return self._action(action, message)

    async def _send_macro_gcode(self, script: str, correlation_id: str) -> None:
        self.events.publish(script, source="user", command=script, correlation_id=correlation_id)
        try:
            response = await asyncio.to_thread(self.moonraker.send_gcode, script)
        except Exception as exc:
            response = MoonrakerResponse(False, 0, error_message=str(exc))
        if not response.ok:
            message = response.error_message or "Moonraker rejected the macro motion command."
            self.events.publish(message, source="moonraker", level="error", correlation_id=correlation_id)
            raise RuntimeUnavailable(message)
        self.events.publish("Macro motion completed.", source="moonraker", correlation_id=correlation_id)

class CameraPickupActions:
    """Camera-guided pickup controls, including the step-by-step safety flow."""

    async def test_camera_stereo(
        self,
        row: int,
        column: int,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        if isinstance(row, bool) or not isinstance(row, int) or not 1 <= row <= self.settings.rack.rows:
            raise ValueError("row is outside rack bounds")
        if isinstance(column, bool) or not isinstance(column, int) or not 1 <= column <= self.settings.rack.columns:
            raise ValueError("column is outside rack bounds")

        async with self._debug_pickup_lock:
            if self._stereo_test_active or self.debug_pickup_active:
                raise RuntimeConflict("Another camera-guided operation is already active.")
            async with self._lock:
                workflow_active = self._workflow["state"] in {"starting", "running", "paused", "stopping"}
                if not self._machine["connected"] or self._machine["klipper_state"] != "ready" or workflow_active:
                    raise RuntimeUnavailable("Klipper must be ready and the workflow inactive before running a stereo test.")
                self._stereo_test_active = True
            self._broadcast_status()
            preview_was_running = False
            original_detection_mode = "hsv"
            tube = self.settings.rack.tube_position(row - 1, column - 1)
            camera_offset = self.settings.rack.camera_offset_mm
            center_x = tube.x - camera_offset.x
            center_y = tube.y - camera_offset.y
            half_baseline = STEREO_BASELINE_MM / 2.0
            moved_from_center = False
            correlation_id = f"req_{uuid.uuid4().hex}"
            try:
                camera_status = camera_preview.status()
                preview_was_running = camera_status["state"] not in {"not_started", "stopped"}
                original_detection_mode = camera_status["detection_mode"]
                if not preview_was_running:
                    await asyncio.to_thread(camera_preview.start)
                camera_preview.set_detection_mode("hough")
                await self._send_macro_gcode(
                    self.motion.synchronized_move_command(z=self.settings.rack.safe_z_mm, feedrate=10000),
                    correlation_id,
                )
                moved_from_center = True
                await self._send_macro_gcode(
                    self.motion.synchronized_move_command(
                        x=center_x - half_baseline,
                        y=center_y,
                        feedrate=10000,
                    ),
                    correlation_id,
                )
                left_sequence = camera_preview.status()["sequence"]
                left = await asyncio.to_thread(camera_preview.wait_for_hough_detection, left_sequence, 10, 8.0)
                if left is None:
                    raise RuntimeUnavailable("No tube circle was detected at the left stereo test position.")

                await self._send_macro_gcode(
                    self.motion.synchronized_move_command(
                        x=center_x + half_baseline,
                        y=center_y,
                        feedrate=10000,
                    ),
                    correlation_id,
                )
                right_sequence = camera_preview.status()["sequence"]
                right = await asyncio.to_thread(camera_preview.wait_for_hough_detection, right_sequence, 10, 8.0)
                if right is None:
                    raise RuntimeUnavailable("No tube circle was detected at the right stereo test position.")

                position = triangulate_stereo_circles(
                    (float(left["detected_center"][0]), float(left["detected_center"][1])),
                    (float(right["detected_center"][0]), float(right["detected_center"][1])),
                    (int(right["width"]), int(right["height"])),
                )
                result = {
                    "ok": True,
                    "action": "camera.stereo_test",
                    "row": row,
                    "column": column,
                    "center_xy_mm": {"x": center_x, "y": center_y},
                    "left_center_px": left["detected_center"],
                    "right_center_px": right["detected_center"],
                    "position_relative_to_center_mm": {
                        "x": position[0],
                        "y": position[1],
                        "z": position[2],
                    },
                    "baseline_mm": STEREO_BASELINE_MM,
                    "message": (
                        f"Stereo estimate relative to scan midpoint: "
                        f"X={position[0]:.2f} mm, Y={position[1]:.2f} mm, Z={position[2]:.2f} mm."
                    ),
                }
                self.events.publish(result["message"], source="camera", correlation_id=correlation_id)
                return result
            finally:
                try:
                    if moved_from_center:
                        await self._send_macro_gcode(
                            self.motion.synchronized_move_command(x=center_x, y=center_y, feedrate=10000),
                            correlation_id,
                        )
                except Exception:
                    self.events.publish(
                        "Standalone stereo test could not return the camera to the scan midpoint.",
                        source="camera",
                        level="error",
                        correlation_id=correlation_id,
                    )
                    raise
                finally:
                    try:
                        camera_preview.set_detection_mode(original_detection_mode)
                        if not preview_was_running:
                            await asyncio.to_thread(camera_preview.stop)
                    finally:
                        self._stereo_test_active = False
                        self._broadcast_status()

    async def begin_camera_pickup(
        self,
        row: int,
        column: int,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        if isinstance(row, bool) or not isinstance(row, int) or not 1 <= row <= self.settings.rack.rows:
            raise ValueError("row is outside rack bounds")
        if isinstance(column, bool) or not isinstance(column, int) or not 1 <= column <= self.settings.rack.columns:
            raise ValueError("column is outside rack bounds")
        safe_z = self.settings.rack.safe_z_mm
        scan_height = self.settings.rack.scan_height_mm
        post_pickup_height = self.settings.rack.post_pickup_height_mm
        if not math.isfinite(scan_height):
            raise RuntimeUnavailable("rack.json scan_height_mm must be finite.")
        if not math.isfinite(post_pickup_height) or post_pickup_height < scan_height:
            raise RuntimeUnavailable("rack.json post_pickup_height_mm must be finite and at or above scan_height_mm.")
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["send_gcode"]:
            raise RuntimeUnavailable("Klipper must be ready and the workflow inactive before locating a tube.")

        async with self._debug_pickup_lock:
            if self.debug_pickup_active:
                raise RuntimeConflict("A camera-guided pickup is already in progress.")
            was_running = camera_preview.status()["state"] not in {"not_started", "stopped"}
            started_by_session = not was_running
            camera_preview.set_detection_mode("hough")
            if started_by_session:
                await asyncio.to_thread(camera_preview.start)
            _, frame = await asyncio.to_thread(camera_preview.wait_for_frame, 0, 8.0)
            if frame is None:
                if started_by_session:
                    await asyncio.to_thread(camera_preview.stop)
                raise RuntimeUnavailable("Camera did not provide a frame; check the camera connection and permissions.")

            tube = self.settings.rack.tube_position(row - 1, column - 1)
            offset = self.settings.rack.camera_offset_mm
            camera_x = tube.x - offset.x
            camera_y = tube.y - offset.y
            target = {"x": None, "y": None}
            steps = [
                {"label": f"Move camera to survey Z={scan_height:.2f} mm above the slot", "command": self.motion.synchronized_move_command(x=camera_x, y=camera_y, z=scan_height, feedrate=10000), "state": "pending"},
                {"label": "Detect every cap and match the one in this slot", "command": "WAITING_FOR_DETECTION", "state": "pending"},
                {"label": "Center camera over the matched cap at its close-up height", "command": "WAITING_FOR_CLOSEUP", "state": "pending"},
                {"label": "Re-detect the cap close up and move the gantry over it", "command": "WAITING_FOR_WORLD_OFFSET", "state": "pending"},
                {"label": "Turn vacuum pump on", "command": self.motion.pickup_command(), "state": "pending"},
                {"label": "Move down to the color-specific pickup height", "command": "WAITING_FOR_PICKUP_HEIGHT", "state": "pending"},
                {"label": f"Raise to Z={post_pickup_height:.2f} mm", "command": self.motion.synchronized_move_command(z=post_pickup_height, feedrate=5000), "state": "pending"},
                {"label": "Turn vacuum pump off", "command": self.motion.vacuum_off_command(), "state": "pending"},
            ]
            self._debug_pickup = {
                "id": f"pickup_{uuid.uuid4().hex}",
                "correlation_id": f"req_{uuid.uuid4().hex}",
                "row": row,
                "column": column,
                "state": "running",
                "next_step": 0,
                "steps": steps,
                "tube_xy": {"x": tube.x, "y": tube.y},
                "camera_xy": {"x": camera_x, "y": camera_y},
                "target_xy": target,
                "detection_sequence": None,
                "left_center_px": None,
                "detected_center_px": None,
                "stereo_position_mm": None,
                "correction_mm": None,
                "pickup_height_mm": None,
                "calibrated_scan_height_mm": None,
                "detected_color": None,
                "world_offset_mm": None,
                "survey_xy": None,
                "survey_caps": [],
                "matched_cap": None,
                "scan_height_mm": scan_height,
                "safe_z_mm": safe_z,
                "current_z": None,
                "vacuum_enabled": False,
                "camera_started_by_session": started_by_session,
            }
            self.events.publish(
                f"Camera-guided pickup {self._debug_pickup['id']} is ready. Review the plan, then run one step at a time.",
                source="camera",
                correlation_id=self._debug_pickup["correlation_id"],
            )
            self._broadcast_status()
            return self._debug_pickup_payload()

    async def survey_tubes(
        self,
        row: int,
        column: int,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        """Move high above ``row``/``column`` and populate every cap the camera can see.

        The camera is raised to ``survey_height_mm`` (well above the normal scan
        height), moved over the selected slot, and every detected cap is placed
        in rack space and matched one-to-one to a slot. Matched slots get their
        cap colour recorded, which is what fills them in in the rack selector.
        """
        if isinstance(row, bool) or not isinstance(row, int) or not 1 <= row <= self.settings.rack.rows:
            raise ValueError("row is outside rack bounds")
        if isinstance(column, bool) or not isinstance(column, int) or not 1 <= column <= self.settings.rack.columns:
            raise ValueError("column is outside rack bounds")

        async with self._survey_lock:
            snapshot = await self.snapshot()
            if not snapshot["capabilities"]["survey_tubes"]:
                raise RuntimeUnavailable(
                    "Klipper must be ready and the workflow idle before surveying the rack."
                )
            self._survey_active = True
            self._broadcast_status()
            correlation_id = f"req_{uuid.uuid4().hex}"
            camera_started = False
            original_mode = "hsv"
            try:
                tube = self.settings.rack.tube_position(row - 1, column - 1)
                offset = self.settings.rack.camera_offset_mm
                camera_x = tube.x - offset.x
                camera_y = tube.y - offset.y
                survey_z = self.settings.rack.survey_height_mm
                if not math.isfinite(survey_z):
                    raise RuntimeUnavailable("rack.json survey_height_mm must be finite.")

                camera_status = camera_preview.status()
                camera_started = camera_status["state"] in {"not_started", "stopped"}
                original_mode = camera_status.get("detection_mode", "hsv")
                camera_preview.set_detection_mode("hough")
                if camera_started:
                    await asyncio.to_thread(camera_preview.start)

                self.events.publish(
                    f"Surveying from R{row} C{column} at Z={survey_z:.2f} mm.",
                    source="camera",
                    correlation_id=correlation_id,
                )
                # Climb straight up first, then travel at survey height.
                command = "\n".join((
                    self.motion.synchronized_move_command(z=survey_z, feedrate=10000),
                    self.motion.synchronized_move_command(x=camera_x, y=camera_y, feedrate=10000),
                ))
                await self._send_macro_gcode(command, correlation_id)

                sequence = camera_preview.status()["sequence"]
                result = await asyncio.to_thread(
                    camera_preview.wait_for_circle_detections,
                    sequence,
                    10,
                    8.0,
                )
                if result is None:
                    raise RuntimeUnavailable("The camera did not provide a fresh survey frame.")
                caps = locate_caps(
                    result["circles"],
                    (camera_x, camera_y),
                    survey_z,
                    self.settings.rack.cap_height_for_color,
                )
                slots = {
                    (slot_row, slot_column): (position.x, position.y)
                    for slot_row in range(self.settings.rack.rows)
                    for slot_column in range(self.settings.rack.columns)
                    for position in (self.settings.rack.tube_position(slot_row, slot_column),)
                }
                assigned = assign_caps_to_slots(
                    caps,
                    slots,
                    self.settings.rack.slot_tolerance_mm,
                )
                populated: list[dict[str, Any]] = []
                for (slot_row, slot_column), cap in assigned.items():
                    if cap.color is None:
                        continue
                    await self._record_tube_cap(slot_row + 1, slot_column + 1, cap)
                    populated.append({
                        "row": slot_row + 1,
                        "column": slot_column + 1,
                        "color": cap.color,
                        "x_mm": cap.x_mm,
                        "y_mm": cap.y_mm,
                        "cap_height_mm": cap.cap_height_mm,
                    })
                message = (
                    f"Survey from R{row} C{column} identified {len(populated)} tube(s) "
                    f"from {len(caps)} cap(s) in view."
                )
                self.events.publish(message, source="camera", correlation_id=correlation_id)
                return {
                    "ok": True,
                    "action": "camera.survey",
                    "row": row,
                    "column": column,
                    "camera_xy": {"x": camera_x, "y": camera_y},
                    "survey_z_mm": survey_z,
                    "caps_seen": len(caps),
                    "tubes_populated": populated,
                    "message": message,
                }
            finally:
                try:
                    if camera_started:
                        await asyncio.to_thread(camera_preview.stop)
                    else:
                        camera_preview.set_detection_mode(original_mode)
                finally:
                    self._survey_active = False
                    self._broadcast_status()

    async def run_camera_pickup_step(self, session_id: str, camera_preview: CameraPreviewService) -> dict[str, Any]:
        async with self._debug_pickup_lock:
            session = self._require_debug_pickup(session_id)
            if session["state"] != "running":
                raise RuntimeConflict(f"Camera-guided pickup is {session['state']}.")
            index = session["next_step"]
            if index >= len(session["steps"]):
                raise RuntimeConflict("Camera-guided pickup has no remaining steps.")

            step = session["steps"][index]
            correlation_id = session["correlation_id"]
            try:
                if index == 1:
                    step["state"] = "running"
                    match, caps = await self._locate_cap_near(
                        camera_preview,
                        session,
                        (session["tube_xy"]["x"], session["tube_xy"]["y"]),
                        "survey",
                    )
                    color = match.color
                    distance = match.distance_to(session["tube_xy"]["x"], session["tube_xy"]["y"])
                    session["detected_center_px"] = list(match.pixel)
                    session["detected_color"] = color
                    session["matched_cap"] = match.as_dict()
                    session["survey_xy"] = {"x": match.x_mm, "y": match.y_mm}
                    await self._record_tube_cap(session["row"], session["column"], match)
                    session["calibrated_scan_height_mm"] = (
                        self.settings.rack.calibrated_scan_height_for_color(color)
                    )
                    session["pickup_height_mm"] = self.settings.rack.pickup_height_for_color(color)
                    step["label"] = (
                        f"Matched {color} cap at X={match.x_mm:.2f} Y={match.y_mm:.2f} mm, "
                        f"{distance:.1f} mm from the slot center ({len(caps)} cap(s) in view)"
                    )
                elif index == 2:
                    closeup_z = session["calibrated_scan_height_mm"]
                    if closeup_z is None or not math.isfinite(closeup_z):
                        raise RuntimeUnavailable("Color-specific close-up height must be finite.")
                    offset = self.settings.rack.camera_offset_mm
                    camera_x = session["survey_xy"]["x"] - offset.x
                    camera_y = session["survey_xy"]["y"] - offset.y
                    move_xy = self.motion.synchronized_move_command(x=camera_x, y=camera_y, feedrate=5000)
                    move_z = self.motion.synchronized_move_command(z=closeup_z, feedrate=3000)
                    # Climb before travelling, travel before descending.
                    current_z = session["current_z"]
                    rising = current_z is None or closeup_z > current_z
                    step["command"] = "\n".join((move_z, move_xy) if rising else (move_xy, move_z))
                    step["label"] = (
                        f"Center camera over the {session['detected_color']} cap at "
                        f"close-up Z={closeup_z:.2f} mm"
                    )
                    await self._send_macro_gcode(step["command"], correlation_id)
                    session["camera_xy"] = {"x": camera_x, "y": camera_y}
                    session["current_z"] = closeup_z
                    session["detection_sequence"] = camera_preview.status()["sequence"]
                elif index == 3:
                    step["state"] = "running"
                    match, _ = await self._locate_cap_near(
                        camera_preview,
                        session,
                        (session["survey_xy"]["x"], session["survey_xy"]["y"]),
                        "close-up",
                        prefer_color=session["detected_color"],
                    )
                    session["detected_center_px"] = list(match.pixel)
                    session["matched_cap"] = match.as_dict()
                    await self._record_tube_cap(session["row"], session["column"], match)
                    target_x, target_y = match.x_mm, match.y_mm
                    offset_x = target_x - session["tube_xy"]["x"]
                    offset_y = target_y - session["tube_xy"]["y"]
                    session["target_xy"] = {"x": target_x, "y": target_y}
                    session["world_offset_mm"] = {"x": offset_x, "y": offset_y}
                    step["command"] = self.motion.synchronized_move_command(x=target_x, y=target_y, feedrate=5000)
                    step["label"] = (
                        f"Move gantry to cap X={target_x:.2f} Y={target_y:.2f} mm "
                        f"(slot offset {offset_x:+.2f}, {offset_y:+.2f})"
                    )
                    await self._send_macro_gcode(step["command"], correlation_id)
                elif index == 5:
                    pickup_height = session["pickup_height_mm"]
                    if pickup_height is None or not math.isfinite(pickup_height):
                        raise RuntimeUnavailable("Color-specific pickup height must be finite.")
                    step["command"] = self.motion.synchronized_move_command(z=pickup_height, feedrate=3000)
                    step["label"] = f"Move to {session['detected_color']} pickup Z={pickup_height:.2f} mm"
                    await self._send_macro_gcode(step["command"], correlation_id)
                else:
                    step["state"] = "running"
                    command = step["command"]
                    self.events.publish(
                        f"Camera pickup step {index + 1}/{len(session['steps'])}: {step['label']} | {command.replace(chr(10), ' ; ') }",
                        source="camera",
                        command=command,
                        correlation_id=correlation_id,
                    )
                    await self._send_macro_gcode(command, correlation_id)
                    if index == 0:
                        session["current_z"] = session["scan_height_mm"]
                        session["detection_sequence"] = camera_preview.status()["sequence"]
                    elif index == 4:
                        session["vacuum_enabled"] = True
                    elif index == 5:
                        session["current_z"] = session["pickup_height_mm"]
                    elif index == 6:
                        session["current_z"] = self.settings.rack.post_pickup_height_mm
                    elif index == 7:
                        session["vacuum_enabled"] = False

                step["state"] = "completed"
                session["next_step"] += 1
                if session["next_step"] == len(session["steps"]):
                    session["state"] = "completed"
                    self.events.publish(
                        f"Camera-guided tube positioning complete; tube raised to Z={self.settings.rack.post_pickup_height_mm:.2f} and vacuum turned off.",
                        source="camera",
                        correlation_id=correlation_id,
                    )
                    if session["camera_started_by_session"]:
                        await asyncio.to_thread(camera_preview.stop)
                self._broadcast_status()
                return self._debug_pickup_payload()
            except Exception:
                step["state"] = "failed"
                session["state"] = "failed"
                if index in {0, 2, 5, 6}:
                    session["current_z"] = None
                self._broadcast_status()
                raise

    async def cancel_camera_pickup(self, session_id: str, camera_preview: CameraPreviewService) -> dict[str, Any]:
        async with self._debug_pickup_lock:
            session = self._require_debug_pickup(session_id)
            if session["state"] not in {"running", "failed"}:
                raise RuntimeConflict(f"Camera-guided pickup is {session['state']}.")
            if session["vacuum_enabled"] and (
                session["current_z"] is None or session["current_z"] < session["safe_z_mm"]
            ):
                command = self.motion.synchronized_move_command(z=session["safe_z_mm"], feedrate=5000)
                await self._send_macro_gcode(command, session["correlation_id"])
                session["current_z"] = session["safe_z_mm"]
                self.events.publish("Cancel raised the held tube to safe Z; vacuum remains on.", source="camera", command=command, correlation_id=session["correlation_id"])
            session["state"] = "cancelled"
            if session["camera_started_by_session"]:
                await asyncio.to_thread(camera_preview.stop)
            self._broadcast_status()
            return self._debug_pickup_payload()

    async def _locate_cap_near(
        self,
        camera_preview: CameraPreviewService,
        session: dict[str, Any],
        target_xy: tuple[float, float],
        phase: str,
        prefer_color: str | None = None,
    ) -> tuple[LocatedCap, list[LocatedCap]]:
        """Detect all caps in view, place them in gantry XY, and pick the one nearest ``target_xy``."""
        result = await asyncio.to_thread(
            camera_preview.wait_for_circle_detections,
            session["detection_sequence"],
            10,
            8.0,
        )
        if result is None:
            raise RuntimeUnavailable(f"Camera did not provide {phase} frames; no pickup was made.")
        caps = locate_caps(
            result["circles"],
            (session["camera_xy"]["x"], session["camera_xy"]["y"]),
            session["current_z"],
            self.settings.rack.cap_height_for_color,
        )
        if phase == "survey":
            session["survey_caps"] = [cap.as_dict() for cap in caps]
        tolerance = self.settings.rack.slot_tolerance_mm
        match = None
        if phase == "survey":
            # Every rack slot competes for the caps in view, so a leaning
            # neighbour cannot be mistaken for this slot's tube.
            rack = self.settings.rack
            slots = {
                (row, column): (position.x, position.y)
                for row in range(rack.rows)
                for column in range(rack.columns)
                for position in (rack.tube_position(row, column),)
            }
            match = assign_caps_to_slots(caps, slots, tolerance).get(
                (session["row"] - 1, session["column"] - 1)
            )
            if match is None:
                nearest = min((cap.distance_to(*target_xy) for cap in caps), default=math.inf)
                raise RuntimeUnavailable(
                    f"No cap could be assigned to row {session['row']}, column {session['column']} "
                    f"({len(caps)} circle(s) seen, {sum(cap.x_mm is not None for cap in caps)} with a known color; "
                    f"nearest placed cap {nearest:.1f} mm away, limit {tolerance:.1f} mm); no pickup was made."
                )
            return match, caps
        if prefer_color:
            match = select_cap([cap for cap in caps if cap.color == prefer_color], target_xy, tolerance)
        if match is None:
            match = select_cap(caps, target_xy, tolerance)
        if match is None:
            placed = sum(cap.x_mm is not None for cap in caps)
            raise RuntimeUnavailable(
                f"No cap within {tolerance:.1f} mm of X={target_xy[0]:.2f} Y={target_xy[1]:.2f} at {phase} height "
                f"({len(caps)} circle(s) seen, {placed} with a known color); no pickup was made."
            )
        return match, caps

    def _require_debug_pickup(self, session_id: str) -> dict[str, Any]:
        session = self._debug_pickup
        if not session or session["id"] != session_id:
            raise RuntimeConflict("Camera-guided pickup session was not found.")
        return session

    def _debug_pickup_payload(self) -> dict[str, Any]:
        session = self._debug_pickup
        assert session is not None
        return {
            "ok": True,
            "action": "camera.pickup.stepper",
            "session_id": session["id"],
            "state": session["state"],
            "row": session["row"],
            "column": session["column"],
            "next_step": session["next_step"],
            "steps": [dict(step) for step in session["steps"]],
            "camera_xy": dict(session["camera_xy"]),
            "tube_xy": dict(session["tube_xy"]),
            "target_xy": dict(session["target_xy"]),
            "left_center_px": list(session["left_center_px"]) if session["left_center_px"] else None,
            "detected_center_px": session["detected_center_px"],
            "detected_color": session["detected_color"],
            "world_offset_mm": dict(session["world_offset_mm"]) if session["world_offset_mm"] else None,
            "survey_xy": dict(session["survey_xy"]) if session["survey_xy"] else None,
            "survey_caps": [dict(cap) for cap in session["survey_caps"]],
            "matched_cap": dict(session["matched_cap"]) if session["matched_cap"] else None,
            "stereo_position_mm": dict(session["stereo_position_mm"]) if session["stereo_position_mm"] else None,
            "correction_mm": dict(session["correction_mm"]) if session["correction_mm"] else None,
            "pickup_height_mm": session["pickup_height_mm"],
            "calibrated_scan_height_mm": session["calibrated_scan_height_mm"],
            "scan_height_mm": session["scan_height_mm"],
            "scan_height_with_offset_mm": self.settings.rack.scan_height_with_offset_mm,
            "post_pickup_height_mm": self.settings.rack.post_pickup_height_mm,
            "safe_z_mm": session["safe_z_mm"],
            "vacuum_enabled": session["vacuum_enabled"],
        }

class WorkflowControlActions:
    """Web-triggered workflow validation and lifecycle controls."""

    def preview(self) -> dict[str, Any]:
        plan = self.workflow_builder.build_plan()
        issues = self._readiness_issues(self._machine)
        return {
            "ok": True,
            "action": "workflow.preview",
            "plan": {
                "rows": self.settings.rack.rows,
                "columns": self.settings.rack.columns,
                "tube_count": self.settings.rack.rows * self.settings.rack.columns,
                "yaw_angles_deg": list(self.settings.yaw.sweep_angles()),
                "step_count": len(plan.steps),
                "estimated_motion_only": False,
            },
            "validation": {"valid": not issues, "issues": issues},
        }

    async def start(
        self,
        selection: list[tuple[int, int]] | None = None,
        *,
        degraded_mode: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._task and not self._task.done():
                raise RuntimeConflict("A workflow is already running.")
            if self.debug_pickup_active:
                raise RuntimeConflict("A camera-guided pickup step-through is active.")
            if self._stereo_test_active:
                raise RuntimeConflict("The standalone stereo test is active.")
            issues = self._readiness_issues(self._machine)
            blocking = [issue for issue in issues if not issue.get("overridable", False)]
            if blocking or (issues and not degraded_mode):
                raise RuntimeUnavailable("Workflow prerequisites are not ready. Enable degraded mode or run Preview for details.")
            selected = selection or [(r, c) for r in range(1, self.settings.rack.rows + 1) for c in range(1, self.settings.rack.columns + 1)]
            selected_set = set(selected)
            if not selected_set:
                raise RuntimeConflict("At least one tube must be selected.")
            self._tubes = self._new_tubes(selected_set)
            now = utc_timestamp()
            workflow_id = f"wf_{uuid.uuid4().hex}"
            plan = [
                step for step in self.workflow_builder.build_plan().steps
                if step.row is None or (step.row, step.column) in selected_set
            ]
            self._workflow = self._empty_workflow()
            self._workflow.update(id=workflow_id, state="starting", started_at=now, updated_at=now)
            self._workflow["progress"]["total_tubes"] = len(selected_set)
            self._workflow["progress"]["total_steps"] = len(plan)
            self._pause_gate.set()
            self._stop_requested.clear()
            self._task = asyncio.create_task(self._run(workflow_id, plan, degraded_mode), name=workflow_id)
        self.events.publish(f"Scan started for {len(selected_set)} tubes.", source="workflow", correlation_id=workflow_id)
        if degraded_mode:
            self.events.publish(
                "Degraded mode enabled: physical pickup and release remain active; QR and rotary scan steps are skipped.",
                source="workflow",
                level="warning",
                correlation_id=workflow_id,
            )
        self._broadcast_status()
        return self._action("workflow.start", f"Scan queued for {len(selected_set)} tubes.", workflow_id)

    async def pause(self) -> dict[str, Any]:
        async with self._lock:
            if self._workflow["state"] == "paused":
                return self._action("workflow.pause", "Workflow is already paused.", self._workflow["id"])
            if self._workflow["state"] != "running":
                raise RuntimeConflict("Only a running workflow can be paused.")
            self._workflow["pause_requested"] = True
            self._pause_gate.clear()
        self._broadcast_status()
        return self._action("workflow.pause", "Pause requested after the active command.", self._workflow["id"])

    async def resume(self) -> dict[str, Any]:
        async with self._lock:
            if self._workflow["state"] != "paused":
                raise RuntimeConflict("Only a paused workflow can be resumed.")
            self._workflow["state"] = "running"
            self._workflow["pause_requested"] = False
            self._workflow["updated_at"] = utc_timestamp()
            workflow_id = self._workflow["id"]
            self._pause_gate.set()
        self.events.publish("Scan resumed.", source="workflow", correlation_id=workflow_id)
        self._broadcast_status()
        return self._action("workflow.resume", "Workflow resumed.", workflow_id)

    async def stop(self) -> dict[str, Any]:
        async with self._lock:
            if self._workflow["state"] == "stopping":
                return self._action("workflow.stop", "Stop is already in progress.", self._workflow["id"])
            if self._workflow["state"] not in {"starting", "running", "paused"}:
                raise RuntimeConflict("No active workflow can be stopped.")
            self._workflow["state"] = "stopping"
            self._workflow["stop_requested"] = True
            self._workflow["updated_at"] = utc_timestamp()
            workflow_id = self._workflow["id"]
            self._stop_requested.set()
            self._pause_gate.set()
        self._broadcast_status()
        return self._action("workflow.stop", "Stop requested after the active command.", workflow_id)

    def _action(self, action: str, message: str, workflow_id: str | None = None) -> dict[str, Any]:
        result = {"ok": True, "action": action, "accepted_at": utc_timestamp(), "message": message}
        if workflow_id:
            result["workflow_id"] = workflow_id
        return result
