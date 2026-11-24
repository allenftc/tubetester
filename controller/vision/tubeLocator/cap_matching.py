"""Turn whole-frame cap circles into rack positions and pick the one for a slot.

Every circle is mapped to gantry XY using the rack calibration plus the cap's
height (looked up from its color), so tilted or crowded tubes are judged by
where their cap actually is, not by where it happens to fall in the image.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Optional, Sequence

from controller.vision.tubeLocator.tube_locator import (
    CALIBRATION_GANTRY_Z_MM,
    CapCircle,
    pixel_to_real,
)


@dataclass(frozen=True)
class MergedCircle:
    """A cap seen consistently over several frames (pixel coordinates)."""

    x: float
    y: float
    radius: float
    color: Optional[str]
    frames_seen: int


@dataclass(frozen=True)
class LocatedCap:
    """A cap with its estimated gantry-frame position (mm)."""

    pixel: tuple[float, float]
    radius_px: float
    color: Optional[str]
    x_mm: Optional[float]
    y_mm: Optional[float]
    cap_height_mm: Optional[float]

    def distance_to(self, x: float, y: float) -> float:
        if self.x_mm is None or self.y_mm is None:
            return math.inf
        return math.hypot(self.x_mm - x, self.y_mm - y)

    def as_dict(self) -> dict[str, object]:
        return {
            "pixel": [self.pixel[0], self.pixel[1]],
            "radius_px": self.radius_px,
            "color": self.color,
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "cap_height_mm": self.cap_height_mm,
        }


def merge_frame_detections(
    frames: Sequence[Iterable[CapCircle]],
    match_distance_px: float = 12.0,
    min_fraction: float = 0.5,
) -> list[MergedCircle]:
    """Cluster circles across frames; keep caps seen in at least ``min_fraction`` of them.

    Centers and radii are averaged, and the color is the majority vote of the
    frames that classified it.
    """
    if not frames:
        return []
    clusters: list[list[CapCircle]] = []
    for circles in frames:
        for circle in circles:
            best: Optional[list[CapCircle]] = None
            best_distance = match_distance_px
            for cluster in clusters:
                cx = sum(c.x for c in cluster) / len(cluster)
                cy = sum(c.y for c in cluster) / len(cluster)
                distance = math.hypot(circle.x - cx, circle.y - cy)
                if distance <= best_distance:
                    best, best_distance = cluster, distance
            if best is None:
                clusters.append([circle])
            else:
                best.append(circle)
    needed = max(1, math.ceil(len(frames) * min_fraction))
    merged: list[MergedCircle] = []
    for cluster in clusters:
        if len(cluster) < needed:
            continue
        colors = Counter(c.color for c in cluster if c.color)
        merged.append(MergedCircle(
            x=sum(c.x for c in cluster) / len(cluster),
            y=sum(c.y for c in cluster) / len(cluster),
            radius=sum(c.radius for c in cluster) / len(cluster),
            color=colors.most_common(1)[0][0] if colors else None,
            frames_seen=len(cluster),
        ))
    return merged


def locate_caps(
    circles: Iterable[MergedCircle],
    gantry_xy: tuple[float, float],
    gantry_z_mm: float,
    cap_height_for_color: Mapping[str, float] | Callable[[str], Optional[float]],
) -> list[LocatedCap]:
    """Map each circle to gantry XY using its color's cap height above the rack.

    ``gantry_xy`` and ``gantry_z_mm`` are where the gantry was when the frames
    were taken. Circles whose color has no known height cannot be placed and
    get ``None`` coordinates.
    """
    lookup = cap_height_for_color.get if isinstance(cap_height_for_color, Mapping) else cap_height_for_color
    raise_mm = gantry_z_mm - CALIBRATION_GANTRY_Z_MM
    located: list[LocatedCap] = []
    for circle in circles:
        height = lookup(circle.color) if circle.color else None
        x_mm = y_mm = None
        if height is not None:
            offset_x, offset_y = pixel_to_real(
                circle.x, circle.y, height_mm=float(height), camera_raise_mm=raise_mm
            )
            x_mm, y_mm = gantry_xy[0] + offset_x, gantry_xy[1] + offset_y
        located.append(LocatedCap(
            pixel=(circle.x, circle.y),
            radius_px=circle.radius,
            color=circle.color,
            x_mm=x_mm,
            y_mm=y_mm,
            cap_height_mm=None if height is None else float(height),
        ))
    return located


def assign_caps_to_slots(
    caps: Iterable[LocatedCap],
    slots: Mapping[tuple[int, int], tuple[float, float]],
    max_distance_mm: float,
) -> dict[tuple[int, int], LocatedCap]:
    """One-to-one cap/slot assignment, closest pairs first.

    A tube leaning toward a neighbour is still given its own slot when the
    neighbour's cap is the better match there, which a plain nearest-cap test
    cannot do once tilt approaches half a pitch.
    """
    caps_list = list(caps)
    pairs = sorted(
        (cap.distance_to(*center), index, slot)
        for index, cap in enumerate(caps_list)
        for slot, center in slots.items()
        if cap.distance_to(*center) <= max_distance_mm
    )
    assigned: dict[tuple[int, int], LocatedCap] = {}
    used: set[int] = set()
    for _, index, slot in pairs:
        if slot in assigned or index in used:
            continue
        assigned[slot] = caps_list[index]
        used.add(index)
    return assigned


def select_cap(
    caps: Iterable[LocatedCap],
    target_xy: tuple[float, float],
    tolerance_mm: float,
) -> Optional[LocatedCap]:
    """Return the placed cap nearest ``target_xy`` if it is within ``tolerance_mm``."""
    best = min(caps, key=lambda cap: cap.distance_to(*target_xy), default=None)
    if best is None or best.distance_to(*target_xy) > tolerance_mm:
        return None
    return best
