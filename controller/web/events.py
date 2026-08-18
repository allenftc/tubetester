from __future__ import annotations

import asyncio
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

_LEVELS = {"debug", "info", "warning", "error"}
_SOURCES = {"controller", "workflow", "moonraker", "klipper", "camera", "qr", "user"}
_ERROR_RE = re.compile(r"(?:\berror\b|!!|unknown command|unable|failed)", re.IGNORECASE)
_WARNING_RE = re.compile(r"(?:\bwarning\b|\bwarn\b|not ready|timeout)", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"(?i)((?:x-api-key|authorization|api[_ -]?key)\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)
_SPACES_RE = re.compile(r"[ \t]+")


def utc_timestamp(value: datetime | None = None) -> str:
    moment = value or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def classify_level(message: str, explicit: str | None = None) -> str:
    if explicit in _LEVELS:
        return explicit
    if _ERROR_RE.search(message):
        return "error"
    if _WARNING_RE.search(message):
        return "warning"
    return "info"


def sanitize_message(message: object) -> str:
    text = str(message).replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _SECRET_RE.sub(r"\1[REDACTED]", text)
    # Preserve line boundaries and quoted values while normalizing ordinary spacing.
    lines = [_collapse_unquoted_spaces(line) for line in text.split("\n")]
    normalized = "\n".join(lines)
    return normalized if len(normalized) <= 2000 else normalized[:1999] + "…"


def _collapse_unquoted_spaces(text: str) -> str:
    result: list[str] = []
    chunk: list[str] = []
    quote: str | None = None
    escaped = False
    for character in text:
        if quote is not None:
            chunk.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            if chunk:
                result.append(_SPACES_RE.sub(" ", "".join(chunk)))
                chunk = []
            quote = character
            chunk.append(character)
        else:
            chunk.append(character)
    if chunk:
        result.append(_SPACES_RE.sub(" ", "".join(chunk)))
    return "".join(result).strip()


class EventStore:
    """In-memory, bounded console history with consecutive repeat collapsing."""

    def __init__(self, max_events: int = 500, collapse_window_seconds: float = 30.0) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._collapse_window = collapse_window_seconds
        self._sequence = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    @property
    def sequence(self) -> int:
        return self._sequence

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def publish(
        self,
        message: object,
        *,
        level: str | None = None,
        source: str = "controller",
        correlation_id: str | None = None,
        command: str | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        normalized = sanitize_message(message)
        actual_source = source if source in _SOURCES else "controller"
        actual_level = classify_level(normalized, level)
        now = timestamp or datetime.now(timezone.utc)
        stamp = utc_timestamp(now)
        newest = self._events[-1] if self._events else None
        command_value = sanitize_message(command) if command is not None else None

        if newest and self._can_collapse(
            newest, normalized, actual_level, actual_source, correlation_id, command_value, now
        ):
            newest["repeat_count"] += 1
            newest["timestamp"] = stamp
            newest["last_timestamp"] = stamp
            newest["sequence"] = self.next_sequence()
            event = dict(newest)
        else:
            sequence = self.next_sequence()
            event = {
                "id": f"evt_{uuid.uuid4().hex}",
                "sequence": sequence,
                "timestamp": stamp,
                "level": actual_level,
                "source": actual_source,
                "message": normalized,
                "repeat_count": 1,
                "first_timestamp": stamp,
                "last_timestamp": stamp,
                "correlation_id": correlation_id,
                "command": command_value,
            }
            self._events.append(event)

        self._fan_out({"type": "console.event", "payload": dict(event)})
        return dict(event)

    def history(self, limit: int = 200) -> list[dict[str, Any]]:
        bounded = max(0, min(limit, 500))
        return [dict(item) for item in list(self._events)[-bounded:]][::-1]

    def broadcast(self, event_type: str, payload: dict[str, Any]) -> int:
        sequence = self.next_sequence()
        self._fan_out({"type": event_type, "payload": payload, "sequence": sequence})
        return sequence

    def subscribe(self, max_queue: int = 256) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=max_queue)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _can_collapse(
        self,
        newest: dict[str, Any],
        message: str,
        level: str,
        source: str,
        correlation_id: str | None,
        command: str | None,
        now: datetime,
    ) -> bool:
        if (
            newest["message"] != message
            or newest["level"] != level
            or newest["source"] != source
            or newest["correlation_id"] != correlation_id
            or newest["command"] != command
        ):
            return False
        last = datetime.fromisoformat(newest["last_timestamp"].replace("Z", "+00:00"))
        aware_now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return 0 <= (aware_now.astimezone(timezone.utc) - last).total_seconds() <= self._collapse_window

    def _fan_out(self, message: dict[str, Any]) -> None:
        stale: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self._subscribers.discard(queue)
