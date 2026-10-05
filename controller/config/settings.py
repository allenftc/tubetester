from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class RackSettings:
    origin_mm: Point3D
    tube_pitch_mm: Point2D
    pickup_offset_mm: Point3D
    camera_offset_mm: Point3D
    pixel_to_mm_multiplier: float
    pickup_height_mm: float
    safe_z_mm: float
    columns: int = 1
    rows: int = 1

    def tube_position(self, row: int, column: int) -> Point3D:
        return Point3D(
            x=self.origin_mm.x + self.pickup_offset_mm.x + self.tube_pitch_mm.x * column,
            y=self.origin_mm.y + self.pickup_offset_mm.y + self.tube_pitch_mm.y * row,
            z=self.origin_mm.z + self.pickup_offset_mm.z,
        )


@dataclass(frozen=True)
class YawSettings:
    start_deg: float
    step_deg: float
    stop_deg: float

    def sweep_angles(self) -> tuple[float, ...]:
        if self.step_deg == 0:
            return (self.start_deg,)

        angles: list[float] = []
        current = self.start_deg
        if self.step_deg > 0:
            while current <= self.stop_deg + 1e-9:
                angles.append(round(current, 10))
                current += self.step_deg
        else:
            while current >= self.stop_deg - 1e-9:
                angles.append(round(current, 10))
                current += self.step_deg
        return tuple(angles)


@dataclass(frozen=True)
class CameraSettings:
    device_index: int
    resolution: ImageSize
    qr_library: str
    roi_center_x: int
    roi_center_y: int
    roi_width: int
    roi_height: int
    roi_rotation_deg: float


@dataclass(frozen=True)
class MoonrakerSettings:
    base_url: str
    timeout_seconds: float
    api_key: str | None = None


@dataclass(frozen=True)
class WebSettings:
    host: str
    port: int
    title: str


@dataclass(frozen=True)
class NetworkSettings:
    moonraker: MoonrakerSettings
    web: WebSettings


@dataclass(frozen=True)
class ControllerSettings:
    rack: RackSettings
    yaw: YawSettings
    camera: CameraSettings
    network: NetworkSettings


def load_settings(config_dir: Path) -> ControllerSettings:
    rack_data = _read_json(config_dir / "rack.json")
    yaw_data = _read_json(config_dir / "yaw.json")
    camera_data = _read_json(config_dir / "camera.json")
    network_data = _read_json(config_dir / "network.json")

    rack = RackSettings(
        origin_mm=_point3d(rack_data["origin_mm"]),
        tube_pitch_mm=_point2d(rack_data["tube_pitch_mm"]),
        pickup_offset_mm=_point3d(rack_data["pickup_offset_mm"]),
        camera_offset_mm=_point3d(rack_data.get("camera_offset_mm", {"x": 0, "y": 0, "z": 0})),
        pixel_to_mm_multiplier=float(rack_data.get("pixel_to_mm_multiplier", 0.1)),
        pickup_height_mm=float(rack_data.get(
            "pickup_height_mm",
            float(rack_data["origin_mm"]["z"]) + float(rack_data["pickup_offset_mm"]["z"]),
        )),
        safe_z_mm=float(rack_data["safe_z_mm"]),
        columns=int(rack_data.get("columns", 1)),
        rows=int(rack_data.get("rows", 1)),
    )
    yaw = YawSettings(
        start_deg=float(yaw_data["start_deg"]),
        step_deg=float(yaw_data["step_deg"]),
        stop_deg=float(yaw_data["stop_deg"]),
    )
    camera_resolution = ImageSize(
        width=int(camera_data["resolution"]["width"]),
        height=int(camera_data["resolution"]["height"]),
    )
    roi_data = camera_data.get("roi", {})
    camera = CameraSettings(
        device_index=int(camera_data["device_index"]),
        resolution=camera_resolution,
        qr_library=str(camera_data["qr_library"]),
        roi_center_x=int(roi_data.get("center_x", camera_resolution.width // 2)),
        roi_center_y=int(roi_data.get("center_y", camera_resolution.height // 2)),
        roi_width=int(roi_data.get("width", round(camera_resolution.width * 100 / 320))),
        roi_height=int(roi_data.get("height", round(camera_resolution.height * 100 / 240))),
        roi_rotation_deg=float(roi_data.get("rotation_deg", 45.0)),
    )
    network = NetworkSettings(
        moonraker=MoonrakerSettings(
            base_url=str(network_data["moonraker"]["base_url"]),
            timeout_seconds=float(network_data["moonraker"]["timeout_seconds"]),
            api_key=network_data["moonraker"].get("api_key"),
        ),
        web=WebSettings(
            host=str(network_data["web"]["host"]),
            port=int(network_data["web"]["port"]),
            title=str(network_data["web"]["title"]),
        ),
    )
    return ControllerSettings(rack=rack, yaw=yaw, camera=camera, network=network)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _point2d(data: dict) -> Point2D:
    return Point2D(x=float(data["x"]), y=float(data["y"]))


def _point3d(data: dict) -> Point3D:
    return Point3D(x=float(data["x"]), y=float(data["y"]), z=float(data["z"]))