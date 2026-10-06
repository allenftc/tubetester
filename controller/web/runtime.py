from __future__ import annotations

import asyncio
import math
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from controller.config.settings import ControllerSettings
from controller.motion.klipper_client import KlipperMotionClient
from controller.network.moonraker import MoonrakerClient, MoonrakerResponse
from controller.vision.camera_preview import CameraPreviewService
from controller.workflow.state_machine import ScanStep, TubeScanWorkflow
from controller.web.events import EventStore, utc_timestamp

_ACTIVE_STATES = {"starting", "running", "paused", "stopping"}
_TERMINAL_STATES = {"idle", "stopped", "completed", "failed"}


class QrBackend(Protocol):
    async def decode(self, row: int, column: int, yaw_angle_deg: float) -> Any: ...


class RuntimeConflict(RuntimeError):
    pass


class RuntimeUnavailable(RuntimeError):
    pass


def _package_version() -> str:
    try:
        return version("tube-tester")
    except PackageNotFoundError:
        return "0.1.0"


class WorkflowRuntime:
    """Authoritative dashboard state and cooperative background workflow executor."""

    def __init__(
        self,
        settings: ControllerSettings,
        moonraker: MoonrakerClient | None = None,
        events: EventStore | None = None,
        qr_backend: QrBackend | None = None,
    ) -> None:
        self.settings = settings
        self.moonraker = moonraker or MoonrakerClient(settings.network.moonraker)
        self.events = events or EventStore()
        self.qr_backend = qr_backend
        self.motion = KlipperMotionClient()
        self.workflow_builder = TubeScanWorkflow(settings)
        self._lock = asyncio.Lock()
        self._manual_macro_lock = asyncio.Lock()
        self._hardware_lock = asyncio.Lock()
        self._hardware_initialized = False
        self._debug_pickup: dict[str, Any] | None = None
        self._debug_pickup_lock = asyncio.Lock()
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._stop_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._held_tube = False
        self._machine = {
            "connected": False,
            "klipper_state": "offline",
            "state_message": "Moonraker has not been contacted.",
            "position_mm": None,
            "homed_axes": [],
        }
        self._workflow = self._empty_workflow()
        self._tubes = self._new_tubes()

    @property
    def debug_pickup_active(self) -> bool:
        return bool(self._debug_pickup and self._debug_pickup["state"] in {"running", "failed"})

    async def initialize(self) -> None:
        await self.refresh_machine()

    async def close(self) -> None:
        if self._task and not self._task.done():
            self._stop_requested.set()
            self._pause_gate.set()
            try:
                await asyncio.wait_for(self._task, timeout=self.settings.network.moonraker.timeout_seconds + 2)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def refresh_machine(self) -> None:
        try:
            response = await asyncio.to_thread(self.moonraker.get_printer_info)
        except Exception as exc:  # defensive boundary around hardware adapter
            response = MoonrakerResponse(False, 0, error_message=str(exc))
        async with self._lock:
            if not response.ok:
                changed = self._machine["connected"] or self._machine["klipper_state"] != "offline"
                self._machine.update(
                    connected=False,
                    klipper_state="offline",
                    state_message="Moonraker is unreachable.",
                    position_mm=None,
                    homed_axes=[],
                )
                self._hardware_initialized = False
                if changed:
                    self.events.publish(
                        f"Moonraker connection failed: {response.error_message or 'unavailable'}",
                        source="moonraker",
                        level="error",
                    )
            else:
                result = (response.payload or {}).get("result", response.payload or {})
                state = str(result.get("state", "ready")).lower()
                self._machine.update(
                    connected=True,
                    klipper_state=state,
                    state_message=str(result.get("state_message", "Printer information received.")),
                )
                if state != "ready":
                    self._hardware_initialized = False
        if response.ok and self._machine["klipper_state"] == "ready":
            await self._initialize_tooling()
        self._broadcast_status()

    async def _initialize_tooling(self) -> None:
        async with self._hardware_lock:
            if self._hardware_initialized or not self._machine["connected"] or self._machine["klipper_state"] != "ready":
                return
            script = self.motion.initialize_tooling_command()
            try:
                response = await asyncio.to_thread(self.moonraker.send_gcode, script)
            except Exception as exc:
                response = MoonrakerResponse(False, 0, error_message=str(exc))
            if not response.ok:
                self.events.publish(
                    response.error_message or "Tooling initialization failed.",
                    source="klipper",
                    level="error",
                    command=script,
                )
                return
            self._hardware_initialized = True
            self.events.publish("Solenoid and vacuum pump initialized off.", source="klipper", command=script)

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            workflow = deepcopy(self._workflow)
            machine = deepcopy(self._machine)
            tubes = deepcopy(self._tubes)
        issues = self._readiness_issues(machine)
        active = workflow["state"] in _ACTIVE_STATES
        debug_active = self.debug_pickup_active
        ready = not issues
        blocking_issues = [issue for issue in issues if not issue.get("overridable", False)]
        degraded_available = bool(issues) and not blocking_issues
        capabilities = {
            "home": machine["klipper_state"] == "ready" and workflow["state"] in _TERMINAL_STATES and not debug_active,
            "preview": not active,
            "start": not blocking_issues and not active and not debug_active and bool(tubes),
            "pause": workflow["state"] == "running",
            "resume": workflow["state"] == "paused",
            "stop": workflow["state"] in _ACTIVE_STATES,
            "send_gcode": machine["klipper_state"] == "ready" and not active and not debug_active,
            "tooling": machine["klipper_state"] == "ready" and workflow["state"] in _TERMINAL_STATES and not debug_active,
            "camera_pickup_stepper": machine["klipper_state"] == "ready" and not active,
            "qr": self.qr_backend is not None,
            "degraded_mode": degraded_available,
        }
        return {
            "schema_version": 1,
            "sequence": self.events.sequence,
            "generated_at": utc_timestamp(),
            "controller": {
                "state": workflow["state"],
                "ready": ready,
                "version": _package_version(),
            },
            "machine": machine,
            "workflow": workflow,
            "camera_pickup": deepcopy(self._debug_pickup_payload()) if self._debug_pickup else None,
            "rack": {
                "rows": self.settings.rack.rows,
                "columns": self.settings.rack.columns,
                "safe_z_mm": self.settings.rack.safe_z_mm,
                "camera_offset_mm": {
                    "x": self.settings.rack.camera_offset_mm.x,
                    "y": self.settings.rack.camera_offset_mm.y,
                    "z": self.settings.rack.camera_offset_mm.z,
                },
                "pixel_to_mm_multiplier": self.settings.rack.pixel_to_mm_multiplier,
                "pickup_height_mm": self.settings.rack.pickup_height_mm,
                "tubes": tubes,
            },
            "capabilities": capabilities,
            "readiness": {"workflow_ready": ready, "issues": issues},
        }

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
        snapshot = await self.snapshot()
        if not snapshot["capabilities"]["send_gcode"]:
            raise RuntimeUnavailable("Klipper must be ready and the workflow inactive before locating a tube.")

        async with self._debug_pickup_lock:
            if self.debug_pickup_active:
                raise RuntimeConflict("A camera-guided pickup is already in progress.")
            was_running = camera_preview.status()["state"] not in {"not_started", "stopped"}
            started_by_session = not was_running
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
                {"label": f"Lift to safe Z={safe_z:.2f} mm with vacuum maintained", "command": self.motion.synchronized_move_command(z=safe_z, feedrate=5000), "state": "pending"},
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
                        camera_preview.wait_for_detection,
                        session["detection_sequence"],
                        8.0,
                    )
                    if detection is None:
                        raise RuntimeUnavailable("No tube center detected in the camera ROI; no gripper correction or pickup was made.")
                    center = detection["detected_center"]
                    roi = detection["roi"]
                    correction_x = (center[0] - roi["center_x"]) * self.settings.rack.pixel_to_mm_multiplier
                    correction_y = (center[1] - roi["center_y"]) * self.settings.rack.pixel_to_mm_multiplier
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
                        session["current_z"] = session["safe_z_mm"]

                step["state"] = "completed"
                session["next_step"] += 1
                if session["next_step"] == len(session["steps"]):
                    session["state"] = "completed"
                    self.events.publish(
                        "Pickup step-through complete; vacuum remains on.",
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
                if index in {0, 5, 6}:
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
            "safe_z_mm": session["safe_z_mm"],
            "vacuum_enabled": session["vacuum_enabled"],
        }

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
                ("gcode", self.motion.synchronized_move_command(z=20, feedrate=10000)),
                ("gcode", self.motion.release_command()),
                ("gcode", self.motion.synchronized_move_command(z=50, feedrate=10000)),
            ),
        )

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

    async def _run(self, workflow_id: str, plan: list[ScanStep], degraded_mode: bool = False) -> None:
        try:
            await self._set_workflow_state("running")
            for index, step in enumerate(plan, start=1):
                if self._stop_requested.is_set():
                    break
                await self._cooperative_pause(workflow_id)
                if self._stop_requested.is_set():
                    break
                await self._before_step(step, index, len(plan))
                if degraded_mode and self._phase_for(step) == "scan":
                    response = MoonrakerResponse(True, 204)
                else:
                    response = await asyncio.to_thread(self.moonraker.send_gcode, self._command_for(step))
                if not response.ok:
                    raise RuntimeError(response.error_message or f"Moonraker returned HTTP {response.status_code}")
                await self._after_step(step, index)
            if self._stop_requested.is_set():
                await self._safe_stop(workflow_id)
                await self._finish("stopped")
                self.events.publish("Scan stopped cooperatively.", source="workflow", correlation_id=workflow_id)
            else:
                await self._finish("completed")
                self.events.publish("Scan completed.", source="workflow", correlation_id=workflow_id)
        except asyncio.CancelledError:
            await self._finish("stopped")
            raise
        except Exception as exc:
            await self._mark_failure(str(exc))
            self.events.publish(f"Scan failed: {exc}", source="workflow", level="error", correlation_id=workflow_id)
        finally:
            await self._ensure_tooling_off(workflow_id)
            self._held_tube = False
            self._broadcast_status()

    async def _ensure_tooling_off(self, workflow_id: str) -> None:
        if self._machine["klipper_state"] != "ready":
            return
        script = self.motion.shutdown_tooling_command()
        try:
            response = await asyncio.to_thread(self.moonraker.send_gcode, script)
        except Exception as exc:
            response = MoonrakerResponse(False, 0, error_message=str(exc))
        if not response.ok:
            self.events.publish(
                response.error_message or "Could not switch the vacuum pump and solenoid off.",
                source="klipper",
                level="warning",
                correlation_id=workflow_id,
            )
        else:
            self.events.publish("Vacuum pump and solenoid switched off after workflow.", source="klipper", command=script, correlation_id=workflow_id)

    async def _cooperative_pause(self, workflow_id: str) -> None:
        if self._pause_gate.is_set():
            return
        await self._set_workflow_state("paused")
        self.events.publish("Scan paused.", source="workflow", correlation_id=workflow_id)
        await self._pause_gate.wait()

    async def _before_step(self, step: ScanStep, index: int, total: int) -> None:
        phase = self._phase_for(step)
        async with self._lock:
            now = utc_timestamp()
            self._workflow["current"] = {
                "step_id": step.name,
                "phase": phase,
                "description": step.description,
                "row": step.row,
                "column": step.column,
                "yaw_angle_deg": step.yaw_angle_deg,
                "step_index": index,
                "step_total": total,
            }
            self._workflow["updated_at"] = now
            if step.row is not None and step.column is not None:
                tube = self._tube(step.row, step.column)
                tube["phase"] = phase
                tube["status"] = {"approach": "approaching", "pickup": "picked_up", "scan": "scanning", "release": tube["status"]}.get(phase, tube["status"])
                tube["started_at"] = tube["started_at"] or now
                if phase == "scan":
                    tube["yaw_attempt"] += 1
        self._broadcast_status()

    async def _after_step(self, step: ScanStep, index: int) -> None:
        phase = self._phase_for(step)
        async with self._lock:
            self._workflow["progress"]["completed_steps"] = index
            if phase == "pickup":
                self._held_tube = True
            if phase == "release" and step.row and step.column:
                self._held_tube = False
                tube = self._tube(step.row, step.column)
                if tube["status"] != "decoded":
                    tube["status"] = "released_without_decode"
                tube["released_at"] = utc_timestamp()
                self._workflow["progress"]["completed_tubes"] += 1
                self._update_percent_and_summary()
        self._broadcast_status()

    async def _safe_stop(self, workflow_id: str) -> None:
        if self._held_tube and self._machine["klipper_state"] == "ready":
            response = await asyncio.to_thread(
                self.moonraker.send_gcode,
                f"{self.motion.release_command()}\n{self.motion.move_command(z=self.settings.rack.safe_z_mm, feedrate=10000)}",
            )
            if not response.ok:
                self.events.publish("Safe release could not be completed.", source="workflow", level="warning", correlation_id=workflow_id)
        async with self._lock:
            for tube in self._tubes:
                if tube["status"] in {"approaching", "picked_up", "scanning"}:
                    tube["status"] = "stopped"

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

    def _command_for(self, step: ScanStep) -> str:
        if step.name == "home":
            return self.motion.home_command()
        if step.name.startswith("approach_"):
            return f"{self.motion.move_command(z=self.settings.rack.safe_z_mm, feedrate=10000)}\n{self.motion.move_command(x=step.x_mm, y=step.y_mm, feedrate=10000)}"
        if step.name.startswith("pickup_"):
            return f"{self.motion.move_command(z=step.z_mm, feedrate=10000)}\n{self.motion.pickup_command()}"
        if step.name.startswith("scan_") and step.yaw_angle_deg is not None:
            return self.motion.set_yaw_command(step.yaw_angle_deg)
        if step.name.startswith("release_"):
            return f"{self.motion.release_command()}\n{self.motion.move_command(z=self.settings.rack.safe_z_mm, feedrate=10000)}"
        raise RuntimeError(f"Unknown workflow step: {step.name}")

    async def _set_workflow_state(self, state: str) -> None:
        async with self._lock:
            self._workflow["state"] = state
            self._workflow["updated_at"] = utc_timestamp()
        self._broadcast_status()

    async def _finish(self, state: str) -> None:
        async with self._lock:
            now = utc_timestamp()
            self._workflow.update(state=state, updated_at=now, finished_at=now, pause_requested=False)
            self._workflow["current"] = None
            self._update_percent_and_summary()

    async def _mark_failure(self, error: str) -> None:
        async with self._lock:
            if self._workflow["current"] and self._workflow["current"]["row"]:
                self._tube(self._workflow["current"]["row"], self._workflow["current"]["column"])["status"] = "failed"
            self._workflow["last_error"] = error[:500]
        await self._finish("failed")

    def _new_tubes(self, selected: set[tuple[int, int]] | None = None) -> list[dict[str, Any]]:
        tubes = []
        for row in range(1, self.settings.rack.rows + 1):
            for column in range(1, self.settings.rack.columns + 1):
                position = self.settings.rack.tube_position(row - 1, column - 1)
                status = "pending" if selected is None or (row, column) in selected else "skipped"
                tubes.append({
                    "row": row,
                    "column": column,
                    "position_mm": {"x": position.x, "y": position.y, "z": position.z},
                    "status": status,
                    "phase": None,
                    "yaw_attempt": 0,
                    "yaw_attempt_total": len(self.settings.yaw.sweep_angles()),
                    "decoded_payload": None,
                    "confidence": None,
                    "frame_id": None,
                    "error": None,
                    "started_at": None,
                    "decoded_at": None,
                    "released_at": None,
                })
        return tubes

    def _empty_workflow(self) -> dict[str, Any]:
        total = self.settings.rack.rows * self.settings.rack.columns
        return {
            "id": None,
            "state": "idle",
            "started_at": None,
            "updated_at": utc_timestamp(),
            "finished_at": None,
            "pause_requested": False,
            "stop_requested": False,
            "current": None,
            "progress": {"completed_tubes": 0, "total_tubes": total, "percent": 0.0, "completed_steps": 0, "total_steps": len(self.workflow_builder.build_plan().steps)},
            "summary": {"pending": total, "active": 0, "decoded": 0, "failed": 0, "released_without_decode": 0, "skipped": 0, "stopped": 0},
            "last_error": None,
        }

    def _readiness_issues(self, machine: dict[str, Any]) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if not machine["connected"]:
            issues.append({"level": "error", "code": "moonraker_offline", "message": "Moonraker is offline.", "overridable": False})
        elif machine["klipper_state"] != "ready":
            issues.append({"level": "error", "code": "klipper_not_ready", "message": "Klipper is not ready.", "overridable": False})
        if self.qr_backend is None:
            issues.append({"level": "warning", "code": "qr_backend_unavailable", "message": "Camera/QR acquisition is unavailable. Degraded mode can record no-decode results without camera work.", "overridable": True})
        return issues

    def _phase_for(self, step: ScanStep) -> str:
        return "home" if step.name == "home" else step.name.split("_", 1)[0]

    def _tube(self, row: int, column: int) -> dict[str, Any]:
        return self._tubes[(row - 1) * self.settings.rack.columns + column - 1]

    def _update_percent_and_summary(self) -> None:
        progress = self._workflow["progress"]
        total = progress["total_tubes"]
        progress["percent"] = round(progress["completed_tubes"] * 100 / total, 1) if total else 0.0
        statuses = [tube["status"] for tube in self._tubes]
        self._workflow["summary"] = {
            "pending": statuses.count("pending"),
            "active": sum(item in {"approaching", "picked_up", "scanning"} for item in statuses),
            "decoded": statuses.count("decoded"),
            "failed": statuses.count("failed"),
            "released_without_decode": statuses.count("released_without_decode"),
            "skipped": statuses.count("skipped"),
            "stopped": statuses.count("stopped"),
        }

    def _action(self, action: str, message: str, workflow_id: str | None = None) -> dict[str, Any]:
        result = {"ok": True, "action": action, "accepted_at": utc_timestamp(), "message": message}
        if workflow_id:
            result["workflow_id"] = workflow_id
        return result

    def _broadcast_status(self) -> None:
        self.events.broadcast("status.changed", {})
