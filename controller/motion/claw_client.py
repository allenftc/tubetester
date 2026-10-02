from __future__ import annotations

import math
import os
import select
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path


class ClawCommunicationError(RuntimeError):
    """The dedicated USB CDC claw controller could not complete a command."""


@dataclass(frozen=True)
class ClawUsbCdcClient:
    """Sends a USB CDC command and waits for the claw's `D` completion byte."""

    device: Path
    completion_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.completion_timeout_seconds <= 0:
            raise ValueError("completion_timeout_seconds must be positive")

    def open(self) -> str:
        return self._send("O")

    def close(self) -> str:
        return self._send("C")

    def calibrate(self) -> str:
        return self._send("H")

    def turn_to_position(self, degrees: int | float) -> str:
        if isinstance(degrees, bool) or not isinstance(degrees, (int, float)):
            raise ValueError("degrees must be a whole number")
        if not math.isfinite(degrees) or not float(degrees).is_integer():
            raise ValueError("degrees must be a finite whole number")
        return self._send(f"T{int(degrees)}")

    def _send(self, command: str) -> str:
        payload = command.encode("ascii")
        try:
            descriptor = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as exc:
            raise ClawCommunicationError(f"Unable to open claw USB CDC device {self.device}: {exc.strerror or exc}") from exc
        original_terminal_settings: list[object] | None = None
        try:
            try:
                original_terminal_settings = termios.tcgetattr(descriptor)
                tty.setraw(descriptor)
            except termios.error as exc:
                raise ClawCommunicationError(f"Unable to configure claw USB CDC device {self.device}: {exc}") from exc
            self._drain_input(descriptor)
            self._write_all(descriptor, payload, command)
            self._wait_for_completion(descriptor, command)
        finally:
            if original_terminal_settings is not None:
                termios.tcsetattr(descriptor, termios.TCSANOW, original_terminal_settings)
            os.close(descriptor)
        return command

    @staticmethod
    def _drain_input(descriptor: int) -> None:
        while select.select([descriptor], [], [], 0)[0]:
            try:
                if not os.read(descriptor, 1024):
                    return
            except BlockingIOError:
                return

    @staticmethod
    def _write_all(descriptor: int, payload: bytes, command: str) -> None:
        sent = 0
        while sent < len(payload):
            try:
                written = os.write(descriptor, payload[sent:])
            except BlockingIOError:
                select.select([], [descriptor], [], 0.1)
                continue
            except OSError as exc:
                raise ClawCommunicationError(f"Unable to send claw command {command!r}: {exc.strerror or exc}") from exc
            if written <= 0:
                raise ClawCommunicationError(f"Unable to send claw command {command!r}: USB CDC device accepted no data")
            sent += written

    def _wait_for_completion(self, descriptor: int, command: str) -> None:
        deadline = time.monotonic() + self.completion_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ClawCommunicationError(
                    f"Timed out waiting {self.completion_timeout_seconds:g} seconds for claw command {command!r} to finish."
                )
            readable, _, _ = select.select([descriptor], [], [], remaining)
            if not readable:
                continue
            try:
                response = os.read(descriptor, 1024)
            except BlockingIOError:
                continue
            except OSError as exc:
                raise ClawCommunicationError(f"Unable to read claw completion for {command!r}: {exc.strerror or exc}") from exc
            if not response:
                raise ClawCommunicationError(f"Claw USB CDC device disconnected while waiting for command {command!r} to finish.")
            if b"D" in response:
                return
