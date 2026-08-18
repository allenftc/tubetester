from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from controller.config.settings import ControllerSettings
from controller.motion.klipper_client import KlipperMotionClient
from controller.network.moonraker import MoonrakerClient, MoonrakerResponse
from controller.workflow.state_machine import TubeScanWorkflow


def create_control_server(settings: ControllerSettings) -> ThreadingHTTPServer:
    moonraker_client = MoonrakerClient(settings.network.moonraker)
    workflow = TubeScanWorkflow(settings)

    class ControlHandler(BaseHTTPRequestHandler):
        server_version = "TubeTesterControl/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_html(_render_index(settings, workflow))
                return
            if self.path == "/api/status":
                self._send_json(
                    {
                        "moonraker": moonraker_client.get_server_info().__dict__,
                        "printer": moonraker_client.get_printer_info().__dict__,
                        "workflow": [step.__dict__ for step in workflow.build_plan().steps],
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/home":
                self._send_json(moonraker_client.home().__dict__)
                return
            if self.path == "/api/dry-run":
                self._send_json({"workflow": [step.__dict__ for step in workflow.build_plan().steps]})
                return
            if self.path == "/api/run-workflow":
                self._send_json(_execute_workflow(settings, workflow, moonraker_client))
                return
            if self.path == "/api/gcode":
                body = self._read_json_body()
                script = str(body.get("script", ""))
                self._send_json(moonraker_client.send_gcode(script).__dict__)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

        def _read_json_body(self) -> dict:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            body = self.rfile.read(content_length)
            if not body:
                return {}
            return json.loads(body.decode("utf-8"))

        def _send_json(self, payload: dict) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer((settings.network.web.host, settings.network.web.port), ControlHandler)


def serve_control_server(settings: ControllerSettings) -> None:
    server = create_control_server(settings)
    host, port = server.server_address
    print(f"Serving control UI on http://{host}:{port}/")
    server.serve_forever()


def _execute_workflow(settings: ControllerSettings, workflow: TubeScanWorkflow, moonraker_client: MoonrakerClient) -> dict:
    motion = KlipperMotionClient()
    plan = workflow.build_plan()
    results: list[dict[str, object]] = []

    safe_z = settings.rack.safe_z_mm

    for step in plan.steps:
        if step.name == "home":
            command = motion.home_command()
            response = moonraker_client.home()
        elif step.name.startswith("approach_r") and step.x_mm is not None and step.y_mm is not None and step.z_mm is not None:
            command = (
                motion.move_command(z=safe_z, feedrate=10000)
                + "\n"
                + motion.move_command(x=step.x_mm, y=step.y_mm, feedrate=10000)
            )
            response = moonraker_client.send_gcode(command)
        elif step.name.startswith("pickup_r") and step.z_mm is not None:
            command = (
                motion.move_command(z=step.z_mm, feedrate=10000)
                + "\n"
                + motion.pickup_command()
            )
            response = moonraker_client.send_gcode(command)
        elif step.name.startswith("scan_r") and step.yaw_angle_deg is not None:
            command = motion.set_yaw_command(step.yaw_angle_deg)
            response = moonraker_client.send_gcode(command)
        elif step.name.startswith("release_r"):
            command = (
                motion.move_command(z=safe_z, feedrate=10000)
                + "\n"
                + motion.release_command()
            )
            response = moonraker_client.send_gcode(command)
        else:
            command = ""
            response = MoonrakerResponse(
                ok=False,
                status_code=0,
                payload=None,
                error_message=f"Unknown workflow step: {step.name}",
            )

        results.append(
            {
                "step": step.__dict__,
                "command": command,
                "response": response.__dict__,
            }
        )

        if not response.ok:
            break

    return {"results": results}


def _render_index(settings: ControllerSettings, workflow: TubeScanWorkflow) -> str:
    workflow_preview = "\n".join(f"<li>{step.name}: {step.description}</li>" for step in workflow.build_plan().steps)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{settings.network.web.title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #111827; color: #e5e7eb; }}
    main {{ max-width: 960px; margin: 0 auto; }}
    .panel {{ background: #1f2937; border: 1px solid #374151; border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; }}
    button, input {{ font: inherit; }}
    button {{ margin-right: 0.5rem; margin-bottom: 0.5rem; padding: 0.75rem 1rem; border-radius: 10px; border: 0; background: #22c55e; color: #052e16; cursor: pointer; }}
    pre {{ white-space: pre-wrap; background: #0f172a; padding: 1rem; border-radius: 12px; overflow-x: auto; }}
    input[type=\"text\"] {{ width: 100%; max-width: 100%; padding: 0.75rem; border-radius: 10px; border: 1px solid #374151; background: #0f172a; color: #e5e7eb; margin-top: 0.5rem; }}
    ul {{ line-height: 1.6; }}
  </style>
</head>
<body>
  <main>
    <h1>{settings.network.web.title}</h1>
    <p>Moonraker: {settings.network.moonraker.base_url}</p>
    <section class=\"panel\">
      <h2>Control</h2>
      <button onclick=\"post('/api/home')\">Home</button>
      <button onclick=\"post('/api/dry-run')\">Dry Run</button>
      <button onclick=\"post('/api/run-workflow')\">Run Workflow</button>
      <button onclick=\"refreshStatus()\">Refresh Status</button>
      <label for=\"gcode\">Raw G-code</label>
      <input id=\"gcode\" type=\"text\" placeholder=\"M105\" />
      <button onclick=\"sendGcode()\">Send G-code</button>
    </section>
    <section class=\"panel\">
      <h2>Workflow Preview</h2>
      <ul>{workflow_preview}</ul>
    </section>
    <section class=\"panel\">
      <h2>Status</h2>
      <pre id=\"status\">Loading...</pre>
    </section>
  </main>
  <script>
    async function post(path, payload) {{
      const response = await fetch(path, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: payload ? JSON.stringify(payload) : JSON.stringify({{}}),
      }});
      document.getElementById('status').textContent = JSON.stringify(await response.json(), null, 2);
    }}
    async function refreshStatus() {{
      const response = await fetch('/api/status');
      document.getElementById('status').textContent = JSON.stringify(await response.json(), null, 2);
    }}
    async function sendGcode() {{
      const script = document.getElementById('gcode').value;
      await post('/api/gcode', {{ script }});
    }}
    refreshStatus();
  </script>
</body>
</html>"""