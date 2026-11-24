from __future__ import annotations

import asyncio
import contextlib
import math
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import WSMsgType, web

from controller.config.settings import ControllerSettings, load_settings
from controller.hot_reload import SourceWatcher, reload_modules, touched_vision
from controller.vision.camera_preview import CameraPreviewService
from controller.vision.tubeLocator.cap_matching import (
    MergedCircle,
    assign_caps_to_slots,
    locate_caps,
)
from controller.vision.tubeLocator.tube_locator import (
    CALIBRATION_GANTRY_Z_MM,
    rack_camera_pose,
)
from controller.web.events import EventStore, utc_timestamp
from controller.web.runtime import RuntimeConflict, RuntimeUnavailable, WorkflowRuntime

_WEB_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _WEB_ROOT.parents[1]
_MAX_BODY = 16 * 1024
_WATCH_INTERVAL_SECONDS = 0.75
_DEBOUNCE_SECONDS = 0.25


def create_control_app(
    settings: ControllerSettings,
    *,
    runtime: WorkflowRuntime | None = None,
    camera_preview: CameraPreviewService | None = None,
    events: EventStore | None = None,
    initialize_hardware: bool = True,
    reload_request: Callable[[str, str], None] | None = None,
    close_camera_on_cleanup: bool = True,
) -> web.Application:
    events = events or (runtime.events if runtime else EventStore())
    controller = runtime or WorkflowRuntime(settings, events=events)
    preview = camera_preview or CameraPreviewService(settings.camera)
    app = web.Application(client_max_size=_MAX_BODY, middlewares=[_error_middleware])
    app["settings"] = settings
    app["runtime"] = controller
    app["camera_preview"] = preview
    app["events"] = events
    app["initialize_hardware"] = initialize_hardware
    app["reload_request"] = reload_request
    app["close_camera_on_cleanup"] = close_camera_on_cleanup
    app["background_tasks"] = set()
    app["websockets"] = set()

    app.router.add_get("/", _index)
    app.router.add_get("/api/status", _status)
    app.router.add_get("/api/camera/status", _camera_status)
    app.router.add_get("/api/camera/caps", _camera_caps)
    app.router.add_get("/api/camera/stream", _camera_stream)
    app.router.add_get("/api/camera/mask", _camera_mask_stream)
    app.router.add_get("/api/camera/debug", _camera_debug_stream)
    app.router.add_post("/api/camera/preview", _camera_preview_toggle)
    app.router.add_post("/api/camera/detection-mode", _camera_detection_mode)
    app.router.add_post("/api/camera/circle-parameters", _camera_circle_parameters)
    app.router.add_post("/api/camera/color-range-recording", _camera_color_range_recording)
    app.router.add_post("/api/camera/photo", _camera_photo)
    app.router.add_post("/api/actions/home", _home)
    app.router.add_post("/api/actions/preview", _preview)
    app.router.add_post("/api/actions/locate-tube", _locate_tube)
    app.router.add_post("/api/actions/stereo-test", _stereo_test)
    app.router.add_post("/api/actions/locate-tube/step", _locate_tube_step)
    app.router.add_post("/api/actions/locate-tube/cancel", _cancel_locate_tube)
    app.router.add_post("/api/tooling/release", _tooling_release)
    app.router.add_post("/api/tooling/vacuum", _tooling_vacuum)
    app.router.add_post("/api/tooling/vacuum/off", _tooling_vacuum_off)
    app.router.add_post("/api/tooling/rotary/zero", _rotary_zero)
    app.router.add_post("/api/tooling/rotary/move", _rotary_move)
    app.router.add_post("/api/macros/calibrate", _macro_calibrate)
    app.router.add_post("/api/macros/pickup", _macro_pickup)
    app.router.add_post("/api/macros/deposit", _macro_deposit)
    app.router.add_post("/api/workflow/start", _start)
    app.router.add_post("/api/workflow/pause", _pause)
    app.router.add_post("/api/workflow/resume", _resume)
    app.router.add_post("/api/workflow/stop", _stop)
    app.router.add_post("/api/gcode", _gcode)
    app.router.add_post("/api/system/reload", _system_reload)
    app.router.add_get("/ws", _websocket)
    app.router.add_static("/static/", _WEB_ROOT / "static", show_index=False)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def serve_control_server(
    settings: ControllerSettings,
    *,
    config_dir: Path | None = None,
    watch: bool = False,
) -> None:
    server = ControlServer(
        settings,
        config_dir or (_REPO_ROOT / "calibration"),
        watch=watch,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass


class ControlServer:
    """Own the HTTP listener and rebuild it in place when sources change."""

    def __init__(
        self,
        settings: ControllerSettings,
        config_dir: Path,
        *,
        watch: bool = False,
        watch_interval_seconds: float = _WATCH_INTERVAL_SECONDS,
    ) -> None:
        self._settings = settings
        self._config_dir = Path(config_dir).resolve()
        self._watch = watch
        self._watch_interval_seconds = watch_interval_seconds
        self._requests: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._code_paths: tuple[Path, ...] = ()

    @property
    def settings(self) -> ControllerSettings:
        return self._settings

    def request_reload(
        self,
        scope: str,
        reason: str = "manual request",
        paths: tuple[Path, ...] | None = None,
    ) -> None:
        if scope not in {"config", "code"}:
            raise ValueError("scope must be 'config' or 'code'")
        if scope == "code":
            # Unknown paths mean "everything", which forces a camera restart.
            self._code_paths = tuple(paths) if paths is not None else ()
        self._requests.put_nowait((scope, reason))

    async def run(self) -> None:
        events = EventStore()
        preview = CameraPreviewService(self._settings.camera)
        watcher = (
            asyncio.create_task(self._watch_sources(), name="source-watcher")
            if self._watch
            else None
        )
        try:
            while True:
                app, runner = await self._start(events, preview)
                replace_app = False
                try:
                    while True:
                        scope, reason = await self._requests.get()
                        if scope == "config":
                            await self._reload_config(app, events, preview, reason)
                            continue
                        if await self._prepare_code_reload(app, events, reason):
                            replace_app = True
                            break
                finally:
                    await runner.cleanup()
                if replace_app:
                    preview = await self._swap_code(events, preview, reason)
        finally:
            if watcher is not None:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
            await asyncio.to_thread(preview.stop)

    async def _start(
        self,
        events: EventStore,
        preview: CameraPreviewService,
    ) -> tuple[web.Application, web.AppRunner]:
        runtime = WorkflowRuntime(self._settings, events=events)
        app = create_control_app(
            self._settings,
            runtime=runtime,
            camera_preview=preview,
            events=events,
            reload_request=self.request_reload,
            close_camera_on_cleanup=False,
        )
        runner = web.AppRunner(app, shutdown_timeout=3.0)
        await runner.setup()
        site = web.TCPSite(
            runner,
            self._settings.network.web.host,
            self._settings.network.web.port,
            reuse_address=True,
        )
        await site.start()
        print(
            f"Serving control UI on http://{self._settings.network.web.host}:"
            f"{self._settings.network.web.port}/"
        )
        if self._watch:
            print("Watching Python and calibration files; edits apply without restarting.")
        return app, runner

    async def _reload_config(
        self,
        app: web.Application,
        events: EventStore,
        preview: CameraPreviewService,
        reason: str,
    ) -> None:
        await self._wait_until_idle(app["runtime"], events)
        try:
            settings = load_settings(self._config_dir)
        except Exception as exc:
            events.publish(f"Calibration reload failed: {exc}", source="controller", level="error")
            return
        try:
            await app["runtime"].apply_settings(settings)
        except RuntimeConflict as exc:
            events.publish(f"Calibration reload skipped: {exc}", source="controller", level="warning")
            return
        app["settings"] = settings
        self._settings = settings
        await asyncio.to_thread(preview.apply_settings, settings.camera)
        events.publish(
            f"Reloaded calibration ({reason}): rack {settings.rack.rows}x{settings.rack.columns}, "
            f"safe Z {settings.rack.safe_z_mm:g} mm.",
            source="controller",
        )

    async def _prepare_code_reload(
        self,
        app: web.Application,
        events: EventStore,
        reason: str,
    ) -> bool:
        """Re-import modules and stop the old runtime; True when ready to rebuild."""
        await self._wait_until_idle(app["runtime"], events)
        try:
            names = await asyncio.to_thread(reload_modules)
        except RuntimeError as exc:
            events.publish(str(exc), source="controller", level="error")
            print(f"Hot reload failed: {exc}")
            return False
        await app["runtime"].close()
        print(f"Reloaded {len(names)} modules ({reason}).")
        return True

    async def _swap_code(
        self,
        events: EventStore,
        preview: CameraPreviewService,
        reason: str,
    ) -> CameraPreviewService:
        try:
            settings = load_settings(self._config_dir)
        except Exception as exc:
            settings = self._settings
            events.publish(
                f"Calibration files could not be read; keeping previous values: {exc}",
                source="controller",
                level="warning",
            )
        self._settings = settings
        if not self._code_paths or touched_vision(self._code_paths, _REPO_ROOT):
            await asyncio.to_thread(preview.stop)
            preview = CameraPreviewService(settings.camera)
        else:
            await asyncio.to_thread(preview.apply_settings, settings.camera)
        events.publish(
            f"Reloaded source ({reason}) without restarting the process.",
            source="controller",
        )
        return preview

    async def _wait_until_idle(self, runtime: WorkflowRuntime, events: EventStore) -> None:
        if not runtime.busy:
            return
        events.publish(
            "Change detected; waiting for the active workflow to finish before reloading.",
            source="controller",
        )
        while runtime.busy:
            await asyncio.sleep(1.0)

    async def _watch_sources(self) -> None:
        watcher = SourceWatcher([self._config_dir], _REPO_ROOT)
        while True:
            await asyncio.sleep(self._watch_interval_seconds)
            try:
                changes = await asyncio.to_thread(watcher.poll)
            except Exception:
                continue
            if changes.empty:
                continue
            await asyncio.sleep(_DEBOUNCE_SECONDS)
            changes = changes.merged(await asyncio.to_thread(watcher.poll))
            if changes.code_paths:
                print(f"Detected source change: {changes.describe()}")
                self.request_reload("code", changes.describe(), paths=changes.code_paths)
            elif changes.config_paths:
                print(f"Detected calibration change: {changes.describe()}")
                self.request_reload("config", changes.describe())


@web.middleware
async def _error_middleware(request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]) -> web.StreamResponse:
    correlation_id = f"req_{uuid.uuid4().hex}"
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except RuntimeConflict as exc:
        return _error("workflow_conflict", str(exc), correlation_id, 409)
    except RuntimeUnavailable as exc:
        return _error("service_unavailable", str(exc), correlation_id, 503)
    except (ValueError, TypeError) as exc:
        return _error("invalid_request", str(exc), correlation_id, 400)
    except Exception:
        request.app["events"].publish(
            f"Controller request failed (reference {correlation_id}).",
            source="controller",
            level="error",
            correlation_id=correlation_id,
        )
        return _error("internal_error", "The controller could not complete the request.", correlation_id, 500)


async def _on_startup(app: web.Application) -> None:
    if app["initialize_hardware"]:
        task = asyncio.create_task(app["runtime"].initialize(), name="machine-initialize")
        app["background_tasks"].add(task)
        task.add_done_callback(app["background_tasks"].discard)


async def _on_cleanup(app: web.Application) -> None:
    for socket in list(app["websockets"]):
        with contextlib.suppress(Exception):
            await socket.close(code=1001, message=b"controller reloading")
    if app["close_camera_on_cleanup"]:
        await asyncio.to_thread(app["camera_preview"].stop)
    await app["runtime"].close()
    tasks = list(app["background_tasks"])
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _index(_: web.Request) -> web.FileResponse:
    response = web.FileResponse(_WEB_ROOT / "templates" / "index.html")
    response.headers["Cache-Control"] = "no-cache"
    return response


async def _status(request: web.Request) -> web.Response:
    return web.json_response(await request.app["runtime"].snapshot(), headers={"Cache-Control": "no-store"})


async def _camera_status(request: web.Request) -> web.Response:
    return web.json_response(request.app["camera_preview"].status(), headers={"Cache-Control": "no-store"})


async def _camera_caps(request: web.Request) -> web.Response:
    """Place the latest cap detections in rack-frame 3D using colours and pixel_to_real.

    Positions come from :func:`locate_caps`, which applies the per-colour cap
    height and the live gantry Z to the pixel-to-real homography. The gantry
    position is read from the machine snapshot; ``x``/``y``/``z`` query
    parameters override it for bench testing without Klipper.
    """
    preview: CameraPreviewService = request.app["camera_preview"]
    runtime: WorkflowRuntime = request.app["runtime"]
    settings: ControllerSettings = request.app["settings"]
    status = preview.status()
    snapshot = await runtime.snapshot()
    position = (snapshot.get("machine") or {}).get("position_mm") or {}

    x = _float_param(request.query.get("x"), "x")
    y = _float_param(request.query.get("y"), "y")
    z = _float_param(request.query.get("z"), "z")
    if x is None:
        x = _optional_number(position.get("x"))
    if y is None:
        y = _optional_number(position.get("y"))
    if z is None:
        z = _optional_number(position.get("z"))
    known = x is not None and y is not None and z is not None

    circles: list[MergedCircle] = []
    for item in status.get("detected_circles") or []:
        if not isinstance(item, dict):
            continue
        circle_x = _optional_number(item.get("x"))
        circle_y = _optional_number(item.get("y"))
        radius = _optional_number(item.get("radius"))
        if circle_x is None or circle_y is None or radius is None:
            continue
        circles.append(MergedCircle(
            x=circle_x,
            y=circle_y,
            radius=radius,
            color=item.get("color"),
            frames_seen=1,
        ))

    rack = settings.rack
    caps: list[dict[str, Any]] = []
    placed_caps = None
    if known:
        try:
            placed_caps = locate_caps(circles, (float(x), float(y)), float(z), rack.cap_height_for_color)
        except ValueError:
            placed_caps = None  # A cap height at/above the camera; report unplaced instead.
    if placed_caps is not None:
        slot_centers = {
            (row, column): (position.x, position.y)
            for row in range(rack.rows)
            for column in range(rack.columns)
            for position in (rack.tube_position(row, column),)
        }
        assigned = assign_caps_to_slots(placed_caps, slot_centers, rack.slot_tolerance_mm)
        slot_for_cap = {id(cap): slot for slot, cap in assigned.items()}
        for cap in placed_caps:
            entry = cap.as_dict()
            entry["placed"] = cap.x_mm is not None
            slot = slot_for_cap.get(id(cap))
            entry["slot"] = (
                {"row": slot[0] + 1, "column": slot[1] + 1} if slot is not None else None
            )
            caps.append(entry)
    else:
        for circle in circles:
            caps.append({
                "pixel": [circle.x, circle.y],
                "radius_px": circle.radius,
                "color": circle.color,
                "x_mm": None,
                "y_mm": None,
                "cap_height_mm": None,
                "placed": False,
            })

    frame_width = _optional_number(status.get("width"))
    frame_height = _optional_number(status.get("height"))
    if caps and frame_width and frame_height:
        focus = min(
            range(len(caps)),
            key=lambda index: (
                (caps[index]["pixel"][0] - frame_width / 2) ** 2
                + (caps[index]["pixel"][1] - frame_height / 2) ** 2
            ),
        )
        caps[focus]["focus"] = True

    camera = None
    if known:
        try:
            pose = rack_camera_pose()
        except Exception:
            pose = None
        if pose is not None:
            camera = {
                "x": float(x) + pose.camera_xy_mm[0],
                "y": float(y) + pose.camera_xy_mm[1],
                "z": pose.camera_height_mm + (float(z) - CALIBRATION_GANTRY_Z_MM),
                "height_mm": pose.camera_height_mm,
                "tilt_deg": pose.tilt_deg,
                "raise_mm": float(z) - CALIBRATION_GANTRY_Z_MM,
            }

    return web.json_response({
        "ok": True,
        "available": bool(status.get("available")),
        "state": status.get("state"),
        "sequence": status.get("sequence"),
        "machine": {
            "x": x,
            "y": y,
            "z": z,
            "known": known,
        },
        "camera": camera,
        "rack": {
            "rows": rack.rows,
            "columns": rack.columns,
            "pitch_mm": {"x": rack.tube_pitch_mm.x, "y": rack.tube_pitch_mm.y},
            "origin_mm": {
                "x": rack.origin_mm.x + rack.pickup_offset_mm.x,
                "y": rack.origin_mm.y + rack.pickup_offset_mm.y,
                "z": rack.origin_mm.z + rack.pickup_offset_mm.z,
            },
            "safe_z_mm": rack.safe_z_mm,
            "scan_height_mm": rack.scan_height_mm,
            "calibration_gantry_z_mm": CALIBRATION_GANTRY_Z_MM,
        },
        "cap_heights_mm": dict(rack.pickup_height_offset_mm_by_color),
        "caps": caps,
        "placed_count": sum(1 for cap in caps if cap["placed"]),
        "unplaced_count": sum(1 for cap in caps if not cap["placed"]),
    }, headers={"Cache-Control": "no-store"})


async def _camera_preview_toggle(request: web.Request) -> web.Response:
    body = await _json_body(request)
    enabled = body.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    if not enabled and request.app["runtime"].debug_pickup_active:
        raise RuntimeConflict("Camera preview is needed until the pickup step-through is completed or cancelled.")
    preview: CameraPreviewService = request.app["camera_preview"]
    await asyncio.to_thread(preview.start if enabled else preview.stop)
    return web.json_response({"ok": True, "enabled": enabled, "camera": preview.status()})


async def _camera_detection_mode(request: web.Request) -> web.Response:
    body = await _json_body(request)
    mode = body.get("detection_mode")
    if mode not in {"hsv", "hough"}:
        raise ValueError("detection_mode must be 'hsv' or 'hough'")
    preview: CameraPreviewService = request.app["camera_preview"]
    preview.set_detection_mode(mode)
    return web.json_response({"ok": True, "detection_mode": mode})


async def _camera_circle_parameters(request: web.Request) -> web.Response:
    body = await _json_body(request)
    min_distance = body.get("circle_min_distance")
    edge_threshold = body.get("circle_edge_threshold")
    canny_threshold = body.get("circle_canny_threshold")
    method = body.get("circle_method")
    if isinstance(min_distance, bool) or not isinstance(min_distance, (int, float)):
        raise ValueError("circle_min_distance must be a number")
    if isinstance(edge_threshold, bool) or not isinstance(edge_threshold, (int, float)):
        raise ValueError("circle_edge_threshold must be a number")
    if isinstance(canny_threshold, bool) or not isinstance(canny_threshold, (int, float)):
        raise ValueError("circle_canny_threshold must be a number")
    if method not in {"gradient", "gradient_alt"}:
        raise ValueError("circle_method must be 'gradient' or 'gradient_alt'")
    preview: CameraPreviewService = request.app["camera_preview"]
    preview.set_circle_parameters(
        min_distance=float(min_distance),
        edge_threshold=float(edge_threshold),
        canny_threshold=float(canny_threshold),
        method=method,
    )
    return web.json_response({
        "ok": True,
        "circle_min_distance": float(min_distance),
        "circle_edge_threshold": float(edge_threshold),
        "circle_canny_threshold": float(canny_threshold),
        "circle_method": method,
    })


async def _camera_color_range_recording(request: web.Request) -> web.Response:
    body = await _json_body(request)
    recording = body.get("recording")
    if not isinstance(recording, bool):
        raise ValueError("recording must be a boolean")
    preview: CameraPreviewService = request.app["camera_preview"]
    if body.get("reset") is True:
        preview.reset_color_range_recording()
    preview.set_color_range_recording(recording)
    return web.json_response({"ok": True, **preview.status()})


async def _camera_photo(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    preview: CameraPreviewService = request.app["camera_preview"]
    was_running = preview.status()["state"] not in {"not_started", "stopped", "stopping"}
    started_by_request = not was_running
    if started_by_request:
        await asyncio.to_thread(preview.start)
    try:
        photo = await asyncio.to_thread(preview.take_photo)
    except RuntimeError as exc:
        raise RuntimeUnavailable(str(exc)) from exc
    finally:
        if started_by_request:
            await asyncio.to_thread(preview.stop)
    return web.json_response({"ok": True, "action": "camera.photo", **photo})


async def _camera_stream(request: web.Request) -> web.StreamResponse:
    return await _camera_mjpeg_stream(request, mask=False)


async def _camera_mask_stream(request: web.Request) -> web.StreamResponse:
    return await _camera_mjpeg_stream(request, mask=True)


async def _camera_debug_stream(request: web.Request) -> web.StreamResponse:
    return await _camera_mjpeg_stream(request, debug=True)


async def _camera_mjpeg_stream(
    request: web.Request,
    *,
    mask: bool = False,
    debug: bool = False,
) -> web.StreamResponse:
    preview: CameraPreviewService = request.app["camera_preview"]
    status = preview.status()
    if status["state"] in {"not_started", "stopped", "stopping"}:
        return web.json_response(status, status=409, headers={"Cache-Control": "no-store"})
    if not status["available"]:
        return web.json_response(status, status=503, headers={"Cache-Control": "no-store"})

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )
    await response.prepare(request)
    sequence = 0
    wait_for_frame = (
        preview.wait_for_debug_frame if debug
        else preview.wait_for_mask_frame if mask
        else preview.wait_for_frame
    )
    try:
        while True:
            sequence, jpeg = await asyncio.to_thread(wait_for_frame, sequence, 5.0)
            if jpeg is None:
                if not preview.status()["available"]:
                    break
                continue
            await response.write(
                b"--frame\r\nContent-Type: image/jpeg\r\n"
                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                + jpeg
                + b"\r\n"
            )
    except ConnectionResetError:
        pass
    return response


async def _home(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].home())


async def _preview(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(request.app["runtime"].preview())


async def _locate_tube(request: web.Request) -> web.Response:
    body = await _json_body(request)
    row = body.get("row")
    column = body.get("column")
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("row must be an integer")
    if isinstance(column, bool) or not isinstance(column, int):
        raise ValueError("column must be an integer")
    return web.json_response(
        await request.app["runtime"].begin_camera_pickup(
            row,
            column,
            request.app["camera_preview"],
        )
    )


async def _stereo_test(request: web.Request) -> web.Response:
    body = await _json_body(request)
    row = body.get("row")
    column = body.get("column")
    if isinstance(row, bool) or not isinstance(row, int):
        raise ValueError("row must be an integer")
    if isinstance(column, bool) or not isinstance(column, int):
        raise ValueError("column must be an integer")
    return web.json_response(
        await request.app["runtime"].test_camera_stereo(
            row,
            column,
            request.app["camera_preview"],
        )
    )


async def _locate_tube_step(request: web.Request) -> web.Response:
    body = await _json_body(request)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    return web.json_response(
        await request.app["runtime"].run_camera_pickup_step(
            session_id,
            request.app["camera_preview"],
        )
    )


async def _cancel_locate_tube(request: web.Request) -> web.Response:
    body = await _json_body(request)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")
    return web.json_response(
        await request.app["runtime"].cancel_camera_pickup(
            session_id,
            request.app["camera_preview"],
        )
    )


async def _tooling_release(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].release_tube())


async def _tooling_vacuum(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].start_vacuum())


async def _tooling_vacuum_off(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].stop_vacuum())


async def _rotary_zero(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].zero_rotary())


async def _rotary_move(request: web.Request) -> web.Response:
    body = await _json_body(request)
    degrees = body.get("degrees")
    if isinstance(degrees, bool) or not isinstance(degrees, int):
        raise ValueError("degrees must be an integer")
    return web.json_response(await request.app["runtime"].rotate_to_position(degrees))


async def _macro_calibrate(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].calibrate_rotary_macro())


async def _macro_pickup(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].pickup_macro())


async def _macro_deposit(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].deposit_macro())


async def _start(request: web.Request) -> web.Response:
    body = await _json_body(request, allow_empty=True)
    selection = _parse_selection(body, request.app["settings"])
    degraded_mode = body.get("degraded_mode", False)
    if not isinstance(degraded_mode, bool):
        raise ValueError("degraded_mode must be a boolean")
    payload = await request.app["runtime"].start(selection, degraded_mode=degraded_mode)
    return web.json_response(payload, status=202)


async def _pause(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].pause())


async def _resume(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].resume())


async def _stop(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].stop())


async def _gcode(request: web.Request) -> web.Response:
    body = await _json_body(request)
    script = body.get("script")
    if not isinstance(script, str):
        raise ValueError("script must be a string")
    script = script.strip()
    if not script:
        raise ValueError("script must not be blank")
    if len(script) > 1000:
        raise ValueError("script must be at most 1,000 characters")
    if "\x00" in script:
        raise ValueError("script must not contain NUL characters")
    return web.json_response(await request.app["runtime"].send_gcode(script))


async def _system_reload(request: web.Request) -> web.Response:
    body = await _json_body(request, allow_empty=True)
    scope = body.get("scope", "config")
    if scope not in {"config", "code"}:
        raise ValueError("scope must be 'config' or 'code'")
    callback = request.app["reload_request"]
    if callback is None:
        raise RuntimeUnavailable(
            "Hot reload is disabled. Start the controller with --reload to enable it."
        )
    callback(scope, "manual reload requested from the dashboard")
    return web.json_response({"ok": True, "scope": scope}, status=202)


async def _websocket(request: web.Request) -> web.WebSocketResponse:
    _validate_origin(request)
    socket = web.WebSocketResponse(heartbeat=20, receive_timeout=60)
    await socket.prepare(request)
    runtime: WorkflowRuntime = request.app["runtime"]
    events: EventStore = request.app["events"]
    queue = events.subscribe()
    request.app["websockets"].add(socket)
    client_id = f"client_{uuid.uuid4().hex}"
    await _ws_send(socket, events, "hello", {"client_id": client_id, "heartbeat_seconds": 20, "latest_sequence": events.sequence})
    await _ws_send(socket, events, "status.snapshot", await runtime.snapshot())
    for event in reversed(events.history(200)):
        await socket.send_json(_envelope(events, "console.event", event, sequence=event["sequence"]))

    sender = asyncio.create_task(_ws_sender(socket, queue, runtime, events), name=f"ws-{client_id}")
    try:
        async for message in socket:
            if message.type == WSMsgType.TEXT:
                data = message.json(loads=__import__("json").loads)
                if data.get("type") == "resync":
                    await _ws_send(socket, events, "status.snapshot", await runtime.snapshot())
            elif message.type in {WSMsgType.ERROR, WSMsgType.CLOSE}:
                break
    finally:
        request.app["websockets"].discard(socket)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        events.unsubscribe(queue)
    return socket


async def _ws_sender(socket: web.WebSocketResponse, queue: asyncio.Queue[dict[str, Any]], runtime: WorkflowRuntime, events: EventStore) -> None:
    while not socket.closed:
        item = await queue.get()
        event_type = item["type"]
        if event_type == "status.changed":
            await _ws_send(socket, events, "status.snapshot", await runtime.snapshot())
        else:
            payload = item["payload"]
            sequence = payload.get("sequence", item.get("sequence"))
            await socket.send_json(_envelope(events, event_type, payload, sequence=sequence))


async def _ws_send(socket: web.WebSocketResponse, events: EventStore, event_type: str, payload: dict[str, Any]) -> None:
    await socket.send_json(_envelope(events, event_type, payload))


def _envelope(events: EventStore, event_type: str, payload: dict[str, Any], sequence: int | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence if sequence is not None else events.next_sequence(),
        "type": event_type,
        "sent_at": utc_timestamp(),
        "payload": payload,
    }


def _optional_number(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _float_param(value: str | None, name: str) -> float | None:
    """Parse an optional finite float query parameter."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


async def _json_body(request: web.Request, allow_empty: bool = False) -> dict[str, Any]:
    if not request.can_read_body:
        if allow_empty:
            return {}
        raise ValueError("A JSON request body is required")
    if request.content_type != "application/json":
        raise ValueError("Content-Type must be application/json")
    try:
        body = await request.json()
    except Exception as exc:
        raise ValueError("Request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")
    return body


def _parse_selection(body: dict[str, Any], settings: ControllerSettings) -> list[tuple[int, int]] | None:
    selection = body.get("selection", {"mode": "all", "tubes": []})
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    mode = selection.get("mode", "all")
    if mode == "all":
        return None
    if mode != "selected" or not isinstance(selection.get("tubes"), list):
        raise ValueError("selection mode must be all or selected")
    result: set[tuple[int, int]] = set()
    for tube in selection["tubes"]:
        if not isinstance(tube, dict) or not isinstance(tube.get("row"), int) or not isinstance(tube.get("column"), int):
            raise ValueError("Each selected tube requires integer row and column")
        row, column = tube["row"], tube["column"]
        if not (1 <= row <= settings.rack.rows and 1 <= column <= settings.rack.columns):
            raise ValueError("Selected tube is outside rack bounds")
        result.add((row, column))
    if not result:
        raise ValueError("At least one selected tube is required")
    return sorted(result)


def _validate_origin(request: web.Request) -> None:
    origin = request.headers.get("Origin")
    if not origin:
        return
    expected = f"{request.scheme}://{request.host}"
    if origin.rstrip("/") != expected:
        raise web.HTTPForbidden(text="WebSocket origin is not allowed")


def _error(code: str, message: str, correlation_id: str, status: int) -> web.Response:
    return web.json_response(
        {"ok": False, "error": {"code": code, "message": message, "correlation_id": correlation_id}},
        status=status,
    )
