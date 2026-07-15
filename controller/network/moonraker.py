from __future__ import annotations

import json
from dataclasses import dataclass
from urllib import error, request

from controller.config.settings import MoonrakerSettings


@dataclass(frozen=True)
class MoonrakerResponse:
    ok: bool
    status_code: int
    payload: dict | None = None
    error_message: str | None = None


class MoonrakerClient:
    def __init__(self, settings: MoonrakerSettings) -> None:
        self.settings = settings

    def get_server_info(self) -> MoonrakerResponse:
        return self._request("GET", "/server/info")

    def get_printer_info(self) -> MoonrakerResponse:
        return self._request("GET", "/printer/info")

    def send_gcode(self, script: str) -> MoonrakerResponse:
        return self._request("POST", "/printer/gcode/script", json_body={"script": script})

    def home(self) -> MoonrakerResponse:
        return self.send_gcode("G28")

    def _request(self, method: str, path: str, json_body: dict | None = None) -> MoonrakerResponse:
        url = f"{self.settings.base_url.rstrip('/')}" + path
        data = None
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["X-Api-Key"] = self.settings.api_key
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")

        http_request = request.Request(url=url, data=data, method=method, headers=headers)
        try:
            with request.urlopen(http_request, timeout=self.settings.timeout_seconds) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body) if body else None
                return MoonrakerResponse(ok=True, status_code=response.status, payload=payload)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return MoonrakerResponse(ok=False, status_code=exc.code, error_message=body or exc.reason)
        except error.URLError as exc:
            return MoonrakerResponse(ok=False, status_code=0, error_message=str(exc.reason))