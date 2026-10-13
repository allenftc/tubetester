from __future__ import annotations

import asyncio
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from controller.config.settings import ControllerSettings
from controller.motion.klipper_client import KlipperMotionClient
from controller.network.moonraker import MoonrakerClient, MoonrakerResponse
from controller.workflow.state_machine import ScanStep, TubeScanWorkflow
from controller.web.events import EventStore, utc_timestamp
from controller.web.actions import CameraPickupActions, MachineActions, WorkflowControlActions

_ACTIVE_STATES = {"starting", "running", "paused", "stopping"}
_TERMINAL_STATES = {"idle", "stopped", "completed", "failed"}


class QrBackend(Protocol):
    async def decode(self, row: int, column: int, yaw_angle_deg: float) -> Any: ...


def _package_version() -> str:
    try:
        return version("tube-tester")
    except PackageNotFoundError:
        return "0.1.0"


class WorkflowController(MachineActions, CameraPickupActions, WorkflowControlActions):
    """Own machine state and execute dashboard actions and scan workflows."""

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
                "scan_height_mm": self.settings.rack.scan_height_mm,
                "tubes": tubes,
            },
            "capabilities": capabilities,
            "readiness": {"workflow_ready": ready, "issues": issues},
        }

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

    def _broadcast_status(self) -> None:
        self.events.broadcast("status.changed", {})
