from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from aiohttp import WSMsgType, web

from controller.config.settings import ControllerSettings
from controller.web.events import EventStore, utc_timestamp
from controller.web.runtime import RuntimeConflict, RuntimeUnavailable, WorkflowRuntime

_WEB_ROOT = Path(__file__).resolve().parent
_MAX_BODY = 16 * 1024


def create_control_app(
    settings: ControllerSettings,
    *,
    runtime: WorkflowRuntime | None = None,
    initialize_hardware: bool = True,
) -> web.Application:
    events = runtime.events if runtime else EventStore()
    controller = runtime or WorkflowRuntime(settings, events=events)
    app = web.Application(client_max_size=_MAX_BODY, middlewares=[_error_middleware])
    app["settings"] = settings
    app["runtime"] = controller
    app["events"] = events
    app["initialize_hardware"] = initialize_hardware
    app["background_tasks"] = set()

    app.router.add_get("/", _index)
    app.router.add_get("/api/status", _status)
    app.router.add_post("/api/actions/home", _home)
    app.router.add_post("/api/actions/preview", _preview)
    app.router.add_post("/api/workflow/start", _start)
    app.router.add_post("/api/workflow/pause", _pause)
    app.router.add_post("/api/workflow/resume", _resume)
    app.router.add_post("/api/workflow/stop", _stop)
    app.router.add_post("/api/gcode", _gcode)
    app.router.add_get("/ws", _websocket)
    app.router.add_static("/static/", _WEB_ROOT / "static", show_index=False)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def serve_control_server(settings: ControllerSettings) -> None:
    print(f"Serving control UI on http://{settings.network.web.host}:{settings.network.web.port}/")
    web.run_app(
        create_control_app(settings),
        host=settings.network.web.host,
        port=settings.network.web.port,
        print=None,
    )


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


async def _home(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(await request.app["runtime"].home())


async def _preview(request: web.Request) -> web.Response:
    await _json_body(request, allow_empty=True)
    return web.json_response(request.app["runtime"].preview())


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


async def _websocket(request: web.Request) -> web.WebSocketResponse:
    _validate_origin(request)
    socket = web.WebSocketResponse(heartbeat=20, receive_timeout=60)
    await socket.prepare(request)
    runtime: WorkflowRuntime = request.app["runtime"]
    events: EventStore = request.app["events"]
    queue = events.subscribe()
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
