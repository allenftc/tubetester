from __future__ import annotations

from typing import Any

from controller.config.settings import ControllerSettings
from controller.network.moonraker import MoonrakerClient
from controller.vision.camera_preview import CameraPreviewService
from controller.web.controller import QrBackend, WorkflowController
from controller.web.events import EventStore
from controller.web.errors import RuntimeConflict, RuntimeUnavailable


class WorkflowRuntime:
    """Web-facing adapter that delegates controls and state to WorkflowController."""

    def __init__(
        self,
        settings: ControllerSettings,
        moonraker: MoonrakerClient | None = None,
        events: EventStore | None = None,
        qr_backend: QrBackend | None = None,
    ) -> None:
        self._controller = WorkflowController(settings, moonraker, events, qr_backend)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._controller, name)

    @property
    def events(self) -> EventStore:
        return self._controller.events

    @property
    def debug_pickup_active(self) -> bool:
        return self._controller.debug_pickup_active

    async def initialize(self) -> None:
        await self._controller.initialize()

    async def close(self) -> None:
        await self._controller.close()

    async def refresh_machine(self) -> None:
        await self._controller.refresh_machine()

    async def snapshot(self) -> dict[str, Any]:
        return await self._controller.snapshot()

    def preview(self) -> dict[str, Any]:
        return self._controller.preview()

    async def home(self) -> dict[str, Any]:
        return await self._controller.home()

    async def send_gcode(self, script: str) -> dict[str, Any]:
        return await self._controller.send_gcode(script)

    async def release_tube(self) -> dict[str, Any]:
        return await self._controller.release_tube()

    async def start_vacuum(self) -> dict[str, Any]:
        return await self._controller.start_vacuum()

    async def stop_vacuum(self) -> dict[str, Any]:
        return await self._controller.stop_vacuum()

    async def zero_rotary(self) -> dict[str, Any]:
        return await self._controller.zero_rotary()

    async def rotate_to_position(self, degrees: int) -> dict[str, Any]:
        return await self._controller.rotate_to_position(degrees)

    async def begin_camera_pickup(
        self,
        row: int,
        column: int,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        return await self._controller.begin_camera_pickup(row, column, camera_preview)

    async def run_camera_pickup_step(
        self,
        session_id: str,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        return await self._controller.run_camera_pickup_step(session_id, camera_preview)

    async def cancel_camera_pickup(
        self,
        session_id: str,
        camera_preview: CameraPreviewService,
    ) -> dict[str, Any]:
        return await self._controller.cancel_camera_pickup(session_id, camera_preview)

    async def calibrate_rotary_macro(self) -> dict[str, Any]:
        return await self._controller.calibrate_rotary_macro()

    async def pickup_macro(self) -> dict[str, Any]:
        return await self._controller.pickup_macro()

    async def deposit_macro(self) -> dict[str, Any]:
        return await self._controller.deposit_macro()

    async def start(
        self,
        selection: list[tuple[int, int]] | None = None,
        *,
        degraded_mode: bool = False,
    ) -> dict[str, Any]:
        return await self._controller.start(selection, degraded_mode=degraded_mode)

    async def pause(self) -> dict[str, Any]:
        return await self._controller.pause()

    async def resume(self) -> dict[str, Any]:
        return await self._controller.resume()

    async def stop(self) -> dict[str, Any]:
        return await self._controller.stop()
