from __future__ import annotations

import asyncio
import math
import uuid
from typing import Any

from controller.network.moonraker import MoonrakerResponse
from controller.vision.camera_preview import CameraPreviewService
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

    async def release_tube(self) -> dict[str, Any]:
        return await self._send_action("tooling.release", self.motion.release_command(), source="klipper")

    async def start_vacuum(self) -> dict[str, Any]:
        return await self._send_action("tooling.vacuum", self.motion.pickup_command(), source="klipper")

    async def stop_vacuum(self) -> dict[str, Any]:
        return await self._send_action("tooling.vacuum_off", self.motion.vacuum_off_command(), source="klipper")

    async def zero_rotary(self) -> dict[str, Any]:
        return await self._send_action("tooling.rotary.zero", self.motion.set_rotary_position_command(), source="klipper")

    async def rotate_to_position(self, degrees: int) -> dict[str, Any]:
        if isinstance(degrees, bool) or not isinstance(degrees, int):
            raise ValueError("degrees must be an integer")
        return await self._send_action(
            "tooling.rotary.move",
            self.motion.set_yaw_command(degrees),
            source="klipper",
        )

    async def calibrate_rotary_macro(self) -> dict[str, Any]:
        return await self._run_manual_macro(
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
        multiplier = self.settings.rack.pixel_to_mm_multiplier
        if not 0 < multiplier <= 1:
            raise RuntimeUnavailable("rack.json pixel_to_mm_multiplier must be greater than 0 and at most 1 mm per pixel.")
        pickup_height = self.settings.rack.pickup_height_mm
        safe_z = self.settings.rack.safe_z_mm
        if not math.isfinite(pickup_height) or not math.isfinite(safe_z) or pickup_height >= safe_z:
            raise RuntimeUnavailable("rack.json pickup_height_mm must be finite and below safe_z_mm.")
        scan_height = self.settings.rack.scan_height_mm
        if not math.isfinite(scan_height) or scan_height < safe_z:
            raise RuntimeUnavailable("rack.json scan_height_mm must be finite and at or above safe_z_mm.")
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
            _, frame = await asyncio.to_thread(camera_preview.wait_for_frame, -1, 8.0)
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
                {"label": f"Raise to safe Z={safe_z:.2f} mm", "command": self.motion.synchronized_move_command(z=safe_z, feedrate=10000), "state": "pending"},
                {"label": f"Move camera to X={camera_x:.2f} Y={camera_y:.2f} mm", "command": self.motion.synchronized_move_command(x=camera_x, y=camera_y, feedrate=10000), "state": "pending"},
                {"label": "Scan camera ROI for fresh tube-center detection", "command": "CAMERA_DETECT", "state": "pending"},
                {"label": "Move gripper to corrected XY at safe Z", "command": "WAITING_FOR_DETECTION", "state": "pending"},
                {"label": "Turn vacuum on", "command": self.motion.pickup_command(), "state": "pending"},
                {"label": f"Lower to pickup Z={pickup_height:.2f} mm", "command": self.motion.synchronized_move_command(z=pickup_height, feedrate=3000), "state": "pending"},
                {"label": f"Lift to scan Z={scan_height:.2f} mm with vacuum maintained", "command": self.motion.synchronized_move_command(z=scan_height, feedrate=5000), "state": "pending"},
                {
                    "label": "Deposit tube at the center of the selected slot",
                    "command": "\n".join((
                        self.motion.synchronized_move_command(x=tube.x, y=tube.y, feedrate=5000),
                        self.motion.synchronized_move_command(z=pickup_height, feedrate=3000),
                        self.motion.release_command(),
                        self.motion.vacuum_off_command(),
                        self.motion.synchronized_move_command(z=scan_height, feedrate=5000),
                    )),
                    "state": "pending",
                },
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
                "detected_center_px": None,
                "correction_mm": None,
                "pickup_height_mm": pickup_height,
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
                if index == 2:
                    step["state"] = "running"
                    detection = await asyncio.to_thread(
                        camera_preview.wait_for_hough_detection,
                        session["detection_sequence"],
                        10,
                        8.0,
                    )
                    if detection is None:
                        raise RuntimeUnavailable("No tube center detected in the camera ROI; no gripper correction or pickup was made.")
                    offset = detection["detected_offset_roi"]
                    correction_x = offset[0] * self.settings.rack.pixel_to_mm_multiplier
                    correction_y = -offset[1] * self.settings.rack.pixel_to_mm_multiplier
                    center = detection["detected_center"]
                    target_x = session["tube_xy"]["x"] + correction_x
                    target_y = session["tube_xy"]["y"] + correction_y
                    session["target_xy"] = {"x": target_x, "y": target_y}
                    session["detected_center_px"] = center
                    session["correction_mm"] = {"x": correction_x, "y": correction_y}
                    next_step = session["steps"][3]
                    next_step["command"] = self.motion.synchronized_move_command(x=target_x, y=target_y, feedrate=5000)
                    next_step["label"] = f"Move gripper to corrected X={target_x:.2f} Y={target_y:.2f} mm at safe Z"
                    self.events.publish(
                        f"Detected center px=({center[0]}, {center[1]}); correction X={correction_x:.2f} Y={correction_y:.2f} mm; gripper target X={target_x:.2f} Y={target_y:.2f} mm.",
                        source="camera",
                        correlation_id=correlation_id,
                    )
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
                        session["current_z"] = session["safe_z_mm"]
                    elif index == 1:
                        session["detection_sequence"] = camera_preview.status()["sequence"]
                    elif index == 4:
                        session["vacuum_enabled"] = True
                    elif index == 5:
                        session["current_z"] = session["pickup_height_mm"]
                    elif index == 6:
                        session["current_z"] = session["scan_height_mm"]
                    elif index == 7:
                        session["current_z"] = session["scan_height_mm"]
                        session["target_xy"] = session["tube_xy"]
                        session["vacuum_enabled"] = False

                step["state"] = "completed"
                session["next_step"] += 1
                if session["next_step"] == len(session["steps"]):
                    session["state"] = "completed"
                    self.events.publish(
                        "Pickup and deposit complete; tube returned to the selected slot and vacuum is off.",
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
                if index in {0, 5, 6, 7}:
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
            "detected_center_px": session["detected_center_px"],
            "correction_mm": dict(session["correction_mm"]) if session["correction_mm"] else None,
            "pickup_height_mm": session["pickup_height_mm"],
            "scan_height_mm": session["scan_height_mm"],
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
