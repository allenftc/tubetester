from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QrDecodeResult:
    payload: str | None
    confidence: float
    frame_id: int | None = None


class QrDecoder:
    """Placeholder interface for a future QR decoding backend."""

    def decode(self, frame_bytes: bytes, frame_id: int | None = None) -> QrDecodeResult:
        raise NotImplementedError("QR decoding backend is not wired yet.")