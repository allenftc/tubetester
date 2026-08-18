from __future__ import annotations

import asyncio
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

from controller.config.settings import ControllerSettings
from controller.motion.klipper_client import KlipperMotionClient
from controller.network.moonraker import MoonrakerClient, MoonrakerResponse
from controller.workflow.state_machine import ScanStep, TubeScanWorkflow
from controller.web.events import EventStore, utc_timestamp

_ACTIVE_STATES = {"starting", "running", "paused", "stopping"}
_TERMINAL_STATES = {"idle", "stopped", "completed", "failed"}
_REQUIRED_MACROS = ("TUBE_PICKUP", "TUBE_RELEASE", "TUBE_SET_YAW")


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
        macros_available: bool = False,
    ) -> None:
        self.settings = settings
        self.moonraker = moonraker or MoonrakerClient(settings.network.moonraker)
        self.events = events or EventStore()
        self.qr_backend = qr_backend
        self.macros_available = macros_available
        self.motion = KlipperMotionClient()
        self.workflow_builder = TubeScanWorkflow(settings)
        self._lock = asyncio.Lock()
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
        self._broadcast_status()

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            workflow = deepcopy(self._workflow)
            machine = deepcopy(self._machine)
            tubes = deepcopy(self._tubes)
        issues = self._readiness_issues(machine)
        active = workflow["state"] in _ACTIVE_STATES
        ready = not issues
        blocking_issues = [issue for issue in issues if not issue.get("overridable", False)]
        degraded_available = bool(issues) and not blocking_issues
        capabilities = {
            "home": machine["klipper_state"] == "ready" and workflow["state"] in _TERMINAL_STATES,
            "preview": not active,
            "start": not blocking_issues and not active and bool(tubes),
            "pause": workflow["state"] == "running",
            "resume": workflow["state"] == "paused",
            "stop": workflow["state"] in _ACTIVE_STATES,
            "send_gcode": machine["klipper_state"] == "ready" and not active,
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
            "rack": {
                "rows": self.settings.rack.rows,
                "columns": self.settings.rack.columns,
                "safe_z_mm": self.settings.rack.safe_z_mm,
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

    async def start(
        self,
        selection: list[tuple[int, int]] | None = None,
        *,
        degraded_mode: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._task and not self._task.done():
                raise RuntimeConflict("A workflow is already running.")
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
                "Degraded mode enabled: pickup, QR/yaw, and release macro steps will be simulated without sending commands.",
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
                if degraded_mode and self._phase_for(step) in {"pickup", "scan", "release"}:
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
            self._held_tube = False
            self._broadcast_status()

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
        if not self.macros_available:
            issues.append({"level": "warning", "code": "missing_klipper_macro", "message": f"Required macros are not verified: {', '.join(_REQUIRED_MACROS)}. Degraded mode can skip these steps.", "overridable": True})
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
