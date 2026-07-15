from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from controller.config.settings import ControllerSettings
from controller.network.moonraker import MoonrakerClient
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