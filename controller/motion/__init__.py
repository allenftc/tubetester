"""Hardware adapter layer for the Klipper-controlled scanner."""

from .klipper_client import KlipperMotionClient

__all__ = ["KlipperMotionClient"]