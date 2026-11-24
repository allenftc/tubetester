"use strict";

const ACTIVE_TUBES = new Set(["approaching", "picked_up", "scanning"]);
const state = {
  snapshot: null,
  selectedTube: null,
  lastSequence: 0,
  reconnectDelay: 500,
  socket: null,
  cameraEnabled: false,
  cameraAvailable: false,
  cameraImageFailed: false,
  cameraPollTimer: null,
  cameraLocateBusy: false,
  stereoTestBusy: false,
  cameraPickupSession: null,
  cameraPickupPreviewWasEnabled: true,
  caps3d: {
    data: null,
    error: null,
    target: null,
    yaw: -2.35,
    pitch: 0.92,
    distance: 620,
    dragging: false,
    lastX: 0,
    lastY: 0,
    resetOnNextFrame: true,
    pollTimer: null,
    polling: false,
  },
};
const el = id => document.getElementById(id);

function setText(id, text) { el(id).textContent = String(text); }
function classToken(value) { return String(value || "neutral").toLowerCase().replace(/[^a-z0-9_-]/g, "-"); }

async function api(path, body = {}) {
  const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const payload = await response.json().catch(() => ({ok: false, error: {message: "Invalid controller response."}}));
  if (!response.ok) throw new Error(payload.error?.message || `Request failed (${response.status}).`);
  return payload;
}

async function bootstrap() {
  wireControls();
  initCaps3d();
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    renderSnapshot(await response.json());
  } catch (error) {
    setReadiness([error.message]);
  }
  connectSocket();
  connectCameraPreview();
}

function connectCameraPreview() {
  const image = el("camera-stream");
  const debug = el("camera-debug-stream");
  for (const feed of [image, debug]) {
    feed.addEventListener("error", () => { state.cameraImageFailed = true; });
    feed.addEventListener("load", () => { state.cameraImageFailed = false; });
  }
  el("camera-preview-toggle").addEventListener("change", event => setCameraPreviewEnabled(event.target.checked));
  el("take-photo-button").addEventListener("click", takePhoto);
  for (const id of ["circle-min-distance", "circle-edge-threshold", "circle-canny-threshold"]) {
    el(id).addEventListener("input", event => {
      const value = event.currentTarget.value;
      el(`${id}-value`).textContent = value;
      setCameraCircleParameters();
    });
  }
  el("circle-method").addEventListener("change", () => setCameraCircleParameters());
  window.addEventListener("pagehide", () => {
    if (state.cameraEnabled) {
      fetch("/api/camera/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({enabled: false}),
        keepalive: true,
      }).catch(() => {});
    }
  });
  if (["running", "failed"].includes(state.cameraPickupSession?.state)) restorePickupCameraPreview();
}

async function restorePickupCameraPreview() {
  try {
    const response = await fetch("/api/camera/status", {cache: "no-store"});
    const status = await response.json();
    state.cameraEnabled = !["not_started", "stopped"].includes(status.state);
    el("camera-preview-toggle").checked = state.cameraEnabled;
    if (state.cameraEnabled) {
      refreshCameraStatus();
      startCaps3dPolling();
    }
  } catch (error) {
    setText("camera-details", error.message);
  }
}

async function setCameraPreviewEnabled(enabled) {
  const toggle = el("camera-preview-toggle");
  const image = el("camera-stream");
  const debug = el("camera-debug-stream");
  toggle.disabled = true;
  if (!enabled) {
    state.cameraEnabled = false;
    window.clearTimeout(state.cameraPollTimer);
    stopCaps3dPolling();
    state.caps3d.data = null;
    image.removeAttribute("src");
    image.hidden = true;
    debug.removeAttribute("src");
    debug.hidden = true;
    el("camera-debug-panel").hidden = true;
    state.cameraAvailable = false;
    state.cameraImageFailed = false;
    setChip("camera-chip", "PREVIEW OFF", "neutral");
    setText("camera-details", "Preview is off. Camera capture is stopped.");
    renderCaps3d();
  }
  try {
    const result = await api("/api/camera/preview", {enabled});
    state.cameraEnabled = result.enabled;
    if (result.enabled) {
      setChip("camera-chip", "CAMERA STARTING", "warning");
      setText("camera-details", `Opening ${result.camera.device}…`);
      refreshCameraStatus();
      startCaps3dPolling();
    } else {
      stopCaps3dPolling();
      state.caps3d.data = null;
      renderCaps3d();
    }
  } catch (error) {
    toggle.checked = !enabled;
    announce(error.message);
    if (enabled) {
      setChip("camera-chip", "CAMERA ERROR", "error");
      setText("camera-details", error.message);
    } else {
      state.cameraEnabled = true;
      refreshCameraStatus();
      startCaps3dPolling();
    }
  } finally {
    toggle.disabled = state.cameraLocateBusy || ["running", "failed"].includes(state.cameraPickupSession?.state);
  }
}

async function takePhoto() {
  const button = el("take-photo-button");
  button.disabled = true;
  setText("camera-photo-status", "Capturing a raw frame…");
  try {
    const result = await api("/api/camera/photo");
    setText("camera-photo-status", `Saved ${result.filename} (${result.width} × ${result.height}) to the project root.`);
    announce(`Saved photo ${result.filename}.`);
  } catch (error) {
    setText("camera-photo-status", error.message);
    announce(error.message);
    alert(error.message);
  } finally {
    button.disabled = false;
  }
}

let circleParameterRequest = Promise.resolve();
function setCameraCircleParameters() {
  const minDistance = Number(el("circle-min-distance").value);
  const edgeThreshold = Number(el("circle-edge-threshold").value);
  const cannyThreshold = Number(el("circle-canny-threshold").value);
  const method = el("circle-method").value;
  circleParameterRequest = circleParameterRequest.then(() => api(
    "/api/camera/circle-parameters",
    {
      circle_min_distance: minDistance,
      circle_edge_threshold: edgeThreshold,
      circle_canny_threshold: cannyThreshold,
      circle_method: method,
    },
  )).catch(error => announce(error.message));
}

function setCameraFeed(id, endpoint, visible, retry = false) {
  const image = el(id);
  image.hidden = !visible;
  if (!visible) {
    image.removeAttribute("src");
    delete image.dataset.endpoint;
  } else if (retry || image.dataset.endpoint !== endpoint) {
    image.dataset.endpoint = endpoint;
    image.src = `${endpoint}?retry=${Date.now()}`;
  }
}

async function refreshCameraStatus() {
  if (!state.cameraEnabled) return;
  try {
    const response = await fetch("/api/camera/status", {cache: "no-store"});
    const status = await response.json();
    renderCameraStatus(status);
  } catch (error) {
    setChip("camera-chip", "CAMERA STATUS ERROR", "error");
    setText("camera-details", error.message);
  }
  if (state.cameraEnabled) state.cameraPollTimer = window.setTimeout(refreshCameraStatus, 2000);
}

function renderCameraStatus(status) {
  if (!state.cameraEnabled) return;
  if (Number.isFinite(Number(status.circle_min_distance))) {
    el("circle-min-distance").value = status.circle_min_distance;
    el("circle-min-distance-value").textContent = status.circle_min_distance;
  }
  if (Number.isFinite(Number(status.circle_edge_threshold))) {
    el("circle-edge-threshold").value = status.circle_edge_threshold;
    el("circle-edge-threshold-value").textContent = status.circle_edge_threshold;
  }
  if (Number.isFinite(Number(status.circle_canny_threshold))) {
    el("circle-canny-threshold").value = status.circle_canny_threshold;
    el("circle-canny-threshold-value").textContent = status.circle_canny_threshold;
  }
  if (status.circle_method) el("circle-method").value = status.circle_method;
  const available = Boolean(status.available);
  const starting = ["starting", "connecting"].includes(status.state);
  setChip("camera-chip", available ? "CAMERA LIVE" : starting ? "CAMERA STARTING" : "CAMERA UNAVAILABLE", available ? "ready" : starting ? "warning" : "error");
  const roi = status.roi;
  const detected = status.detected_center ? ` · detected center ${status.detected_center.join(", ")}` : " · center not detected";
  const presence = status.tube_present ? " · tube present" : " · no tube detected";
  setText(
    "camera-details",
    `${status.message} ${status.device} · ${status.width}×${status.height} · ROI center ${roi.center_x}, ${roi.center_y} (${roi.width}×${roi.height})${detected}${presence}`,
  );
  const retryFeeds = available && (!state.cameraAvailable || state.cameraImageFailed);
  setCameraFeed("camera-stream", "/api/camera/stream", available, retryFeeds);
  setCameraFeed("camera-debug-stream", "/api/camera/debug", available, retryFeeds);
  el("camera-debug-panel").hidden = !available;
  if (retryFeeds) state.cameraImageFailed = false;
  state.cameraAvailable = available;
  state.cameraImageFailed = !available;
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/ws`);
  state.socket = socket;
  socket.addEventListener("open", () => { el("stale-banner").hidden = true; state.reconnectDelay = 500; });
  socket.addEventListener("message", event => {
    const message = JSON.parse(event.data);
    // Retained console history intentionally has event sequences older than the
    // freshly generated snapshot envelope and is still valid during bootstrap.
    if (message.sequence && message.sequence <= state.lastSequence && !["status.snapshot", "console.event"].includes(message.type)) return;
    if (message.sequence && state.lastSequence && message.sequence > state.lastSequence + 1) {
      socket.send(JSON.stringify({type: "resync", after_sequence: state.lastSequence}));
    }
    state.lastSequence = Math.max(state.lastSequence, message.sequence || 0);
    if (message.type === "status.snapshot") renderSnapshot(message.payload);
    if (message.type === "console.event") handleConsoleEvent(message.payload);
    if (message.type === "ping") socket.send(JSON.stringify({type: "pong", sequence: message.sequence}));
  });
  socket.addEventListener("close", () => {
    el("stale-banner").hidden = false;
    window.setTimeout(connectSocket, state.reconnectDelay);
    state.reconnectDelay = Math.min(10000, state.reconnectDelay * 2);
  });
  socket.addEventListener("error", () => socket.close());
}

function renderSnapshot(snapshot) {
  state.snapshot = snapshot;
  if (snapshot.camera_pickup && ["running", "failed"].includes(snapshot.camera_pickup.state)) {
    renderCameraPickupSession(snapshot.camera_pickup);
  }
  state.lastSequence = Math.max(state.lastSequence, snapshot.sequence || 0);
  const machine = snapshot.machine;
  const workflow = snapshot.workflow;
  setChip("klipper-chip", `● Klipper ${title(machine.klipper_state)}`, machine.klipper_state === "ready" ? "ready" : machine.connected ? "warning" : "error");
  setChip("controller-chip", `● Controller ${title(workflow.state)}`, workflow.state);
  const p = machine.position_mm;
  setChip("position-chip", p ? `Position X ${number(p.x)} Y ${number(p.y)} Z ${number(p.z)}` : "Position Unknown", "neutral");
  setChip("workflow-chip", workflow.state.toUpperCase(), workflow.state);
  setChip("qr-chip", snapshot.capabilities.qr ? "QR READY" : "QR UNAVAILABLE", snapshot.capabilities.qr ? "ready" : "warning");
  setChip("tooling-chip", snapshot.capabilities.tooling ? "TOOLING READY" : "TOOLING UNAVAILABLE", snapshot.capabilities.tooling ? "ready" : "error");
  const current = workflow.current;
  setChip("coordinate-chip", current?.row ? `R${current.row} C${current.column}` : "NO TUBE", current?.row ? "active" : "neutral");
  setText("current-description", current?.description || workflow.last_error || "Waiting for a scan.");
  setText("step-count", `${workflow.progress.completed_steps} of ${workflow.progress.total_steps} steps`);
  setText("tube-count", `${workflow.progress.completed_tubes} / ${workflow.progress.total_tubes}`);
  setText("percent", `${number(workflow.progress.percent)}%`);
  el("progress-fill").style.width = `${Math.max(0, Math.min(100, workflow.progress.percent))}%`;
  el("progress-fill").parentElement.setAttribute("aria-valuenow", workflow.progress.percent);
  renderStepper(current?.phase, workflow.state);
  renderRack(snapshot.rack);
  renderCaps3d();
  setReadiness(snapshot.readiness.issues.map(issue => issue.message));
  setCapabilities(snapshot.capabilities, workflow.state);
}

function setChip(id, text, status) {
  const node = el(id); node.textContent = text; node.className = `chip ${classToken(status)}`;
}
function title(value) { return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()); }
function number(value) { return Number.isFinite(Number(value)) ? Number(value).toFixed(Number(value) % 1 ? 1 : 0) : "?"; }

function setReadiness(messages) {
  const node = el("readiness");
  node.textContent = messages.length ? `Not ready: ${messages.join(" ")}` : "All scan prerequisites are ready.";
  node.classList.toggle("error", messages.length > 0);
}
function setCapabilities(caps, workflowState) {
  ["home", "preview", "start", "stop"].forEach(name => { el(`${name}-button`).disabled = !caps[name]; });
  const pause = el("pause-button");
  const resume = workflowState === "paused";
  pause.textContent = resume ? "Resume" : "Pause";
  pause.title = resume ? "Continue the paused scan." : "Pause after the active Moonraker command finishes.";
  setText("pause-description", resume ? "Continue scan" : "Hold safely");
  pause.disabled = resume ? !caps.resume : !caps.pause;
  ["tooling-release-button", "tooling-vacuum-button", "tooling-vacuum-off-button", "tooling-zero-button", "rotary-degrees-input"].forEach(id => { el(id).disabled = !caps.tooling; });
  el("rotary-move-button").disabled = !caps.tooling || !isWholeNumber(el("rotary-degrees-input").value);
  const pickupRunning = state.cameraPickupSession?.state === "running";
  const pickupFailed = state.cameraPickupSession?.state === "failed";
  const pickupLocked = pickupRunning || pickupFailed;
  el("stereo-test-button").disabled = !caps.stereo_test || !state.selectedTube || state.cameraLocateBusy || state.stereoTestBusy || pickupLocked;
  el("stereo-test-button").textContent = state.stereoTestBusy ? "Testing Stereo…" : "Test Stereo Position";
  el("locate-selected-tube-button").disabled = !caps.send_gcode || !state.selectedTube || state.cameraLocateBusy || state.stereoTestBusy || pickupRunning;
  el("run-next-pickup-step-button").hidden = !pickupRunning;
  el("run-next-pickup-step-button").disabled = state.cameraLocateBusy || !pickupRunning;
  el("cancel-pickup-steps-button").hidden = !pickupRunning && !pickupFailed;
  el("cancel-pickup-steps-button").disabled = state.cameraLocateBusy || (!pickupRunning && !pickupFailed);
  el("camera-preview-toggle").disabled = state.cameraLocateBusy || pickupLocked;
}

function renderStepper(currentPhase, workflowState) {
  const phases = ["home", "approach", "pickup", "scan", "release"];
  const index = phases.indexOf(currentPhase);
  document.querySelectorAll("#phase-stepper li").forEach((node, i) => {
    node.className = i < index || (!currentPhase && workflowState === "completed") ? "complete" : i === index ? "current" : "";
    node.setAttribute("aria-current", i === index ? "step" : "false");
  });
}

const SVG_NS = "http://www.w3.org/2000/svg";
const RACK_GEOMETRY = {cellWidth: 48, rowDepth: 22, skew: 22, tubeRadius: 8, padX: 52, padY: 84};
const RACK_PLANE = (() => {
  const {cellWidth, skew, rowDepth} = RACK_GEOMETRY;
  const a = cellWidth * cellWidth + skew * skew;
  const b = -skew * rowDepth;
  const d = rowDepth * rowDepth;
  const mid = (a + d) / 2;
  const spread = Math.hypot((a - d) / 2, b);
  const major = Math.sqrt(mid + spread);
  const minor = Math.sqrt(Math.max(0, mid - spread));
  return {tilt: (Math.atan2(2 * b, a - d) * 90) / Math.PI, ratio: minor / major};
})();
const planeTilt = (cx, cy) => `rotate(${RACK_PLANE.tilt.toFixed(2)} ${cx} ${cy})`;
const CAP_COLORS = {
  gray: "#8f8f8f",
  white: "#ededed",
  red: "#c62828",
  purple: "#7b4fb5",
  magenta: "#b5429b",
  pink: "#d16ba5",
  green: "#1d7a3a",
  teal: "#3aa79a",
  "light teal": "#5cc9bd",
  blue: "#1f6fb2",
  yellow: "#d8a51c",
  orange: "#d07338",
};

function svgNode(name, attributes = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  return node;
}

function capColor(color) {
  return CAP_COLORS[String(color || "").toLowerCase()] || "#9a9a9a";
}

function tubeHeightPx(rack, color) {
  const heights = rack.color_heights_mm || {};
  const match = Object.keys(heights).find(name => name.toLowerCase() === String(color || "").toLowerCase());
  const fallback = Object.keys(heights).find(name => name.toLowerCase() === "gray");
  const raw = Number(heights[match ?? fallback] ?? 30);
  const mm = Number.isFinite(raw) && raw > 0 ? raw : 30;
  return Math.max(22, Math.min(72, 14 + mm * 0.95));
}

function rackDefs() {
  const defs = svgNode("defs");
  const glass = svgNode("linearGradient", {id: "tube-glass", x1: "0", x2: "1", y1: "0", y2: "0"});
  [
    ["0%", "rgba(255, 255, 255, .92)"],
    ["14%", "rgba(232, 244, 252, .72)"],
    ["48%", "rgba(184, 214, 235, .46)"],
    ["84%", "rgba(212, 233, 247, .62)"],
    ["100%", "rgba(255, 255, 255, .82)"],
  ].forEach(([offset, color]) => glass.append(svgNode("stop", {offset, "stop-color": color})));
  defs.append(glass);

  const capShade = svgNode("linearGradient", {id: "cap-shade", x1: "0", x2: "0", y1: "0", y2: "1"});
  [
    ["0%", "rgba(0, 0, 0, 0)"],
    ["55%", "rgba(0, 0, 0, .05)"],
    ["100%", "rgba(0, 0, 0, .28)"],
  ].forEach(([offset, color]) => capShade.append(svgNode("stop", {offset, "stop-color": color})));
  defs.append(capShade);
  return defs;
}

function rackTube(cx, cy, tube, rack) {
  const group = svgNode("g", {class: "rack-tube"});
  const radius = RACK_GEOMETRY.tubeRadius;
  const ry = radius * RACK_PLANE.ratio;
  const height = tubeHeightPx(rack, tube.color);
  const topY = cy - height;
  const capHeight = 9;
  const bodyTop = topY + capHeight;
  const color = capColor(tube.color);

  group.append(svgNode("ellipse", {cx, cy: cy + 1.5, rx: radius + 3.5, ry: (radius + 3.5) * RACK_PLANE.ratio, class: "rack-tube-shadow", transform: planeTilt(cx, cy + 1.5)}));
  group.append(svgNode("ellipse", {cx, cy, rx: radius, ry, class: "rack-tube-glass", transform: planeTilt(cx, cy)}));
  group.append(svgNode("rect", {x: cx - radius, y: bodyTop, width: radius * 2, height: Math.max(0, cy - bodyTop), class: "rack-tube-glass"}));
  group.append(svgNode("line", {x1: cx - radius, y1: bodyTop, x2: cx - radius, y2: cy, class: "rack-tube-edge"}));
  group.append(svgNode("line", {x1: cx + radius, y1: bodyTop, x2: cx + radius, y2: cy, class: "rack-tube-edge"}));
  group.append(svgNode("rect", {x: cx - radius * 0.46, y: bodyTop + 2, width: radius * 0.3, height: Math.max(0, cy - bodyTop - 4), class: "rack-tube-glint"}));
  group.append(svgNode("rect", {x: cx - radius, y: topY, width: radius * 2, height: capHeight, fill: color, class: "rack-tube-cap"}));
  group.append(svgNode("rect", {x: cx - radius, y: topY, width: radius * 2, height: capHeight, fill: "url(#cap-shade)", class: "rack-tube-cap-shade"}));
  const capTop = svgNode("g", {transform: planeTilt(cx, topY)});
  capTop.append(svgNode("ellipse", {cx, cy: topY, rx: radius, ry, fill: color, class: "rack-tube-cap"}));
  capTop.append(svgNode("ellipse", {cx: cx - radius * 0.28, cy: topY - ry * 0.15, rx: radius * 0.5, ry: ry * 0.5, class: "rack-tube-cap-glint"}));
  group.append(capTop);
  if (!tube.color) group.classList.add("provisional");
  return group;
}

function renderRack(rack) {
  setText("rack-title", `Rack ${rack.columns} × ${rack.rows}`);
  const grid = el("rack-grid");
  const active = document.activeElement;
  const focusedSlot = active?.dataset?.row ? `${active.dataset.row}:${active.dataset.column}` : null;
  grid.replaceChildren();

  const {cellWidth, rowDepth, skew, tubeRadius, padX, padY} = RACK_GEOMETRY;
  const rows = rack.rows;
  const columns = rack.columns;
  const xFor = (column, rowIndex) => padX + (column - 1) * cellWidth + rowIndex * skew;
  const yFor = rowIndex => padY + (rows - 1 - rowIndex) * rowDepth;
  const minX = xFor(1, 0) - cellWidth / 2 - 48;
  const maxX = xFor(columns, rows - 1) + cellWidth / 2 + skew + 16;
  const minY = padY - rowDepth - 80;
  const maxY = yFor(0) + rowDepth + 32;

  const svg = svgNode("svg", {
    class: "rack-svg",
    viewBox: `${minX} ${minY} ${maxX - minX} ${maxY - minY}`,
    role: "grid",
    "aria-labelledby": "rack-title",
  });
  svg.append(rackDefs());

  const shelf = svgNode("g", {class: "rack-shelf"});
  const nearY = yFor(0);
  const farY = yFor(rows - 1) - rowDepth;
  const nearLeft = xFor(1, 0) - cellWidth / 2;
  const nearRight = xFor(columns, 0) + cellWidth / 2;
  const farRight = xFor(columns, rows - 1) + cellWidth / 2 + skew;
  const thickness = 12;
  shelf.append(svgNode("polygon", {
    points: [
      `${nearRight},${nearY}`,
      `${nearRight},${nearY + thickness}`,
      `${farRight},${farY + thickness}`,
      `${farRight},${farY}`,
    ].join(" "),
    class: "rack-side",
  }));
  shelf.append(svgNode("polygon", {
    points: [
      `${nearLeft},${nearY}`,
      `${nearRight},${nearY}`,
      `${nearRight},${nearY + thickness}`,
      `${nearLeft},${nearY + thickness}`,
    ].join(" "),
    class: "rack-face",
  }));
  svg.append(shelf);

  const labelLayer = svgNode("g", {class: "rack-labels"});
  for (let column = 1; column <= columns; column++) {
    const text = svgNode("text", {x: xFor(column, 0), y: yFor(0) + 30, class: "rack-axis-label", "text-anchor": "middle"});
    text.textContent = `C${column}`;
    labelLayer.append(text);
  }

  for (let row = rows; row >= 1; row--) {
    const rowIndex = row - 1;
    const rowLabel = svgNode("text", {x: xFor(1, rowIndex) - cellWidth / 2 - 22, y: yFor(rowIndex) - rowDepth / 2 + 4, class: "rack-axis-label", "text-anchor": "end"});
    rowLabel.textContent = `R${row}`;
    labelLayer.append(rowLabel);

    for (let column = 1; column <= columns; column++) {
      const tube = rack.tubes.find(item => item.row === row && item.column === column);
      const cx = xFor(column, rowIndex);
      const frontY = yFor(rowIndex);
      const centerX = cx + skew / 2;
      const centerY = frontY - rowDepth / 2;
      const visual = ACTIVE_TUBES.has(tube.status) ? "active" : tube.status;
      const selected = state.selectedTube === `${row}:${column}`;
      const slot = svgNode("g", {class: `rack-slot ${classToken(visual)}${selected ? " selected" : ""}`, role: "gridcell", tabindex: "0"});
      slot.dataset.row = row;
      slot.dataset.column = column;
      slot.setAttribute("aria-label", `Row ${row}, column ${column}, ${title(tube.status)}${tube.color ? `, ${tube.color} cap` : ""}`);
      const slotTitle = svgNode("title");
      slotTitle.textContent = `R${row} C${column} · ${title(tube.status)}${tube.color ? ` · ${tube.color} cap` : ""}`;
      slot.append(slotTitle);
      slot.append(svgNode("polygon", {
        points: [
          `${cx - cellWidth / 2 + 2},${frontY}`,
          `${cx + cellWidth / 2 - 2},${frontY}`,
          `${cx + cellWidth / 2 + skew - 2},${frontY - rowDepth}`,
          `${cx - cellWidth / 2 + skew + 2},${frontY - rowDepth}`,
        ].join(" "),
        class: "rack-slot-top",
      }));
      slot.append(svgNode("ellipse", {cx: centerX, cy: centerY, rx: tubeRadius + 1.5, ry: (tubeRadius + 1.5) * RACK_PLANE.ratio, class: "rack-hole", transform: planeTilt(centerX, centerY)}));
      if (tube.color || selected) slot.append(rackTube(centerX, centerY, tube, rack));
      slot.addEventListener("click", () => selectTube(tube));
      slot.addEventListener("keydown", rackKeydown);
      svg.append(slot);
    }
  }

  svg.append(labelLayer);
  grid.append(svg);
  if (focusedSlot) {
    const [row, column] = focusedSlot.split(":");
    grid.querySelector(`[data-row="${row}"][data-column="${column}"]`)?.focus();
  }
  if (state.selectedTube) {
    const [row, column] = state.selectedTube.split(":").map(Number);
    const selected = rack.tubes.find(tube => tube.row === row && tube.column === column);
    if (selected) showTubeDetails(selected);
  }
}
function selectTube(tube) { state.selectedTube = `${tube.row}:${tube.column}`; renderRack(state.snapshot.rack); showTubeDetails(tube); renderLocateTubePlan(tube); document.querySelector(`[data-row="${tube.row}"][data-column="${tube.column}"]`)?.focus(); if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state); }
function renderLocateTubePlan(tube) {
  const rack = state.snapshot.rack;
  const camera = rack.camera_offset_mm;
  const cameraX = tube.position_mm.x - camera.x;
  const cameraY = tube.position_mm.y - camera.y;
  const stereoHalfBaseline = 5;
  const steps = [
    `Raise to safe Z ${number(rack.safe_z_mm)} mm`,
    `Move camera to left stereo view X ${number(cameraX - stereoHalfBaseline)} Y ${number(cameraY)} mm`,
    "Capture left stereo image and detect tube circle",
    `Move camera to right stereo view X ${number(cameraX + stereoHalfBaseline)} Y ${number(cameraY)} mm`,
    "Capture right stereo image and estimate tube XYZ",
    "Move gripper to corrected tube XY at safe Z",
    "Turn vacuum on",
    `Lower to pickup Z ${number(rack.pickup_height_mm)} mm`,
    `Lift to scan Z ${number(rack.scan_height_mm)} mm; vacuum remains on`,
    "Return to the selected tube slot center, lower, release the tube, turn vacuum off, and lift to scan Z",
  ];
  setText("locate-tube-plan", `R${tube.row} C${tube.column} plan: stereo camera views at X ${number(cameraX - stereoHalfBaseline)} and ${number(cameraX + stereoHalfBaseline)} mm, Y ${number(cameraY)} mm (10 mm baseline); nominal tube XY X ${number(tube.position_mm.x)} Y ${number(tube.position_mm.y)} mm.`);
  const list = document.createElement("ol");
  steps.forEach(step => { const item = document.createElement("li"); item.textContent = step; list.append(item); });
  el("locate-tube-plan-steps").replaceChildren(...list.children);
  setText("locate-tube-status", `Plan ready for R${tube.row} C${tube.column}. Home axes before running.`);
}
function renderCameraPickupSession(session) {
  state.cameraPickupSession = session;
  const list = el("locate-tube-steps");
  list.replaceChildren();
  session.steps.forEach((step, index) => {
    const item = document.createElement("li");
    item.className = `pickup-step ${classToken(step.state)}`;
    item.textContent = `${index + 1}. ${step.label} [${step.state}]${step.command && !["CAMERA_CAPTURE_LEFT", "CAMERA_CAPTURE_RIGHT", "WAITING_FOR_DETECTION"].includes(step.command) ? ` | ${step.command.replaceAll("\n", " ; ")}` : ""}`;
    list.append(item);
  });
  if (session.detected_center_px && session.stereo_position_mm && session.correction_mm) {
    const detail = document.createElement("li");
    detail.className = "pickup-step completed";
    const xyz = session.stereo_position_mm;
    detail.textContent = `Stereo estimate: camera XYZ (${number(xyz.x)}, ${number(xyz.y)}, ${number(xyz.z)}) mm; correction X ${number(session.correction_mm.x)} Y ${number(session.correction_mm.y)} mm; target X ${number(session.target_xy.x)} Y ${number(session.target_xy.y)} mm.`;
    list.append(detail);
  }
  const current = session.steps[session.next_step];
  if (session.state === "running" && current) {
    setText("locate-tube-status", `Paused before step ${session.next_step + 1}: ${current.label}. Press Run Next Step to execute only this step.`);
    el("run-next-pickup-step-button").textContent = `Run Step ${session.next_step + 1} of ${session.steps.length}`;
  } else if (session.state === "completed") {
    setText("locate-tube-status", "Pickup and deposit complete. Tube returned to the selected slot; vacuum is off.");
  } else if (session.state === "cancelled") {
    setText("locate-tube-status", `Step-through cancelled. Vacuum ${session.vacuum_enabled ? "remains on" : "is off"}.`);
  } else if (session.state === "failed") {
    setText("locate-tube-status", "A step failed. Review the error, then cancel to clean up the session.");
  }
}
function showTubeDetails(tube) {
  const p = tube.position_mm;
  const details = [`R${tube.row} C${tube.column}`, title(tube.status), `attempts ${tube.yaw_attempt}/${tube.yaw_attempt_total}`, `X ${number(p.x)} Y ${number(p.y)} Z ${number(p.z)}`];
  if (tube.color) {
    const heightMm = state.snapshot?.rack?.color_heights_mm?.[tube.color];
    details.push(`cap ${tube.color}${Number.isFinite(Number(heightMm)) ? ` · ${number(heightMm)} mm` : ""}`);
  }
  if (tube.decoded_payload) details.push(`payload: ${tube.decoded_payload}`);
  if (tube.confidence != null) details.push(`confidence ${number(tube.confidence)}`);
  if (tube.error) details.push(`error: ${tube.error}`);
  setText("tube-details", details.join(" · "));
}
function rackKeydown(event) {
  const keyMap = {ArrowLeft: [0,-1], ArrowRight:[0,1], ArrowUp:[1,0], ArrowDown:[-1,0]};
  if (!keyMap[event.key]) return;
  event.preventDefault();
  const [dr, dc] = keyMap[event.key];
  const target = document.querySelector(`[data-row="${Number(event.currentTarget.dataset.row)+dr}"][data-column="${Number(event.currentTarget.dataset.column)+dc}"]`);
  target?.focus();
}

function handleConsoleEvent(event) {
  if (event.source !== "camera") return;
  if (event.message.startsWith("Tube debug step ")) {
    const item = document.createElement("li");
    item.textContent = event.message;
    el("locate-tube-steps").append(item);
  }
  if (event.message.startsWith("Tube debug detected center ")) {
    setText("locate-tube-status", event.message);
  }
}

/* --------------------------------------------------------------------------
   Cap 3D view.

   The server places every live cap detection in rack-frame millimetres with
   pixel_to_real plus each colour's cap height. This view draws those caps as
   cylinders over the nominal rack grid, with an orbit camera, so the offset
   between where a tube actually is and where its slot expects it to be is
   visible in 3D.
   -------------------------------------------------------------------------- */

const CAPS3D_FOV = (42 * Math.PI) / 180;
const CAPS3D_DEFAULTS = {yaw: -2.35, pitch: 0.92, distance: 620};
let caps3dFrameHandle = null;

function initCaps3d() {
  const canvas = el("caps3d-canvas");
  if (!canvas) return;
  el("caps3d-reset").addEventListener("click", () => {
    state.caps3d.resetOnNextFrame = true;
    renderCaps3d();
  });
  canvas.addEventListener("pointerdown", event => {
    state.caps3d.dragging = true;
    state.caps3d.lastX = event.clientX;
    state.caps3d.lastY = event.clientY;
    try { canvas.setPointerCapture(event.pointerId); } catch (error) { /* ignored */ }
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", event => {
    if (!state.caps3d.dragging) return;
    const dx = event.clientX - state.caps3d.lastX;
    const dy = event.clientY - state.caps3d.lastY;
    state.caps3d.lastX = event.clientX;
    state.caps3d.lastY = event.clientY;
    state.caps3d.yaw -= dx * 0.008;
    state.caps3d.pitch = caps3dClamp(state.caps3d.pitch + dy * 0.006, 0.12, 1.5);
    renderCaps3d();
  });
  const endDrag = event => {
    if (!state.caps3d.dragging) return;
    state.caps3d.dragging = false;
    canvas.classList.remove("dragging");
    try { if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); } catch (error) { /* ignored */ }
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const zoom = Math.exp(caps3dClamp(event.deltaY * 0.0012, -1, 1));
    state.caps3d.distance = caps3dClamp(state.caps3d.distance * zoom, 90, 5000);
    renderCaps3d();
  }, {passive: false});
  canvas.addEventListener("dblclick", () => {
    state.caps3d.resetOnNextFrame = true;
    renderCaps3d();
  });
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => renderCaps3d()).observe(canvas.parentElement);
  } else {
    window.addEventListener("resize", renderCaps3d);
  }
}

function startCaps3dPolling() {
  if (state.caps3d.polling) return;
  state.caps3d.polling = true;
  const tick = async () => {
    state.caps3d.pollTimer = null;
    if (!state.caps3d.polling) return;
    await refreshCaps3d();
    if (state.caps3d.polling) state.caps3d.pollTimer = window.setTimeout(tick, 350);
  };
  tick();
}

function stopCaps3dPolling() {
  state.caps3d.polling = false;
  window.clearTimeout(state.caps3d.pollTimer);
  state.caps3d.pollTimer = null;
}

async function refreshCaps3d() {
  try {
    const response = await fetch("/api/camera/caps", {cache: "no-store"});
    if (!response.ok) throw new Error(`Cap placement failed (${response.status}).`);
    state.caps3d.data = await response.json();
    state.caps3d.error = null;
  } catch (error) {
    state.caps3d.error = error.message;
  }
  renderCaps3d();
}

function renderCaps3d() {
  if (caps3dFrameHandle !== null) return;
  caps3dFrameHandle = window.requestAnimationFrame(() => {
    caps3dFrameHandle = null;
    drawCaps3d();
  });
}

function drawCaps3d() {
  const canvas = el("caps3d-canvas");
  if (!canvas) return;
  const scene = buildCapsScene(state.caps3d.data);
  updateCaps3dChrome(scene, state.caps3d.data);
  const width = Math.max(1, canvas.clientWidth || canvas.parentElement?.clientWidth || 640);
  const height = Math.max(1, canvas.clientHeight || canvas.parentElement?.clientHeight || 360);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!scene) return;
  if (state.caps3d.resetOnNextFrame || !state.caps3d.target) resetCapsView(scene);
  const view = caps3dProjector(width, height, state.caps3d);
  caps3dDrawRack(ctx, view, scene);
  const tubes = [];
  for (const tube of scene.nominal) tubes.push({...tube, ghost: true});
  for (const tube of scene.detected) tubes.push(tube);
  tubes.sort((a, b) => caps3dDistance2(view.eye, b) - caps3dDistance2(view.eye, a));
  for (const tube of tubes) caps3dDrawTube(ctx, view, tube);
  caps3dDrawCamera(ctx, view, scene);
  if (scene.detected.length && scene.detected.length <= 40) caps3dDrawLabels(ctx, view, scene);
}

function resetCapsView(scene) {
  const size = Math.max(scene.extent, 120);
  state.caps3d.target = {...scene.center};
  state.caps3d.distance = size * 1.45;
  state.caps3d.yaw = CAPS3D_DEFAULTS.yaw;
  state.caps3d.pitch = CAPS3D_DEFAULTS.pitch;
  state.caps3d.resetOnNextFrame = false;
}

function buildCapsScene(data) {
  const snapshot = state.snapshot;
  const rackData = data?.rack;
  const snapshotTubes = snapshot?.rack?.tubes || [];
  let rows = rackData?.rows ?? snapshot?.rack?.rows ?? 0;
  let columns = rackData?.columns ?? snapshot?.rack?.columns ?? 0;
  let origin = rackData?.origin_mm ? {...rackData.origin_mm} : null;
  let pitch = rackData?.pitch_mm ? {...rackData.pitch_mm} : null;
  if ((!origin || !pitch) && snapshotTubes.length) {
    const first = snapshotTubes[0].position_mm;
    const right = snapshotTubes.find(tube => tube.row === 1 && tube.column === 2);
    const down = snapshotTubes.find(tube => tube.row === 2 && tube.column === 1);
    origin = origin || {x: first.x, y: first.y, z: first.z};
    pitch = pitch || {
      x: right ? right.position_mm.x - first.x : 20.2,
      y: down ? down.position_mm.y - first.y : 20.2,
    };
  }
  if (!origin || !pitch || !rows || !columns) return null;

  const heights = data?.cap_heights_mm || snapshot?.rack?.color_heights_mm || {};
  const heightFor = color => {
    const key = Object.keys(heights).find(name => name.toLowerCase() === String(color || "").toLowerCase());
    const value = Number(heights[key]);
    return Number.isFinite(value) ? value : 25;
  };
  const radius = Math.min(Math.abs(pitch.x), Math.abs(pitch.y)) * 0.27;

  const slotCenter = (x, y) => {
    const column = Math.min(columns - 1, Math.max(0, Math.round((x - origin.x) / pitch.x)));
    const row = Math.min(rows - 1, Math.max(0, Math.round((y - origin.y) / pitch.y)));
    return {
      x: origin.x + column * pitch.x,
      y: origin.y + row * pitch.y,
      row: row + 1,
      column: column + 1,
    };
  };

  const detected = [];
  for (const cap of data?.caps || []) {
    if (!cap.placed) continue;
    const x = Number(cap.x_mm);
    const y = Number(cap.y_mm);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const height = Number(cap.cap_height_mm) || heightFor(cap.color);
    const assigned = cap.slot && Number.isFinite(Number(cap.slot.row)) && Number.isFinite(Number(cap.slot.column))
      ? {
        x: origin.x + (Number(cap.slot.column) - 1) * pitch.x,
        y: origin.y + (Number(cap.slot.row) - 1) * pitch.y,
        row: Number(cap.slot.row),
        column: Number(cap.slot.column),
      }
      : null;
    const slot = assigned || slotCenter(x, y);
    detected.push({
      base: {x: slot.x, y: slot.y, z: 0},
      top: {x, y, z: height},
      height,
      color: cap.color || null,
      radius,
      focus: Boolean(cap.focus),
      slot,
    });
  }

  const nominal = [];
  for (const tube of snapshotTubes) {
    if (!tube.color) continue;
    const height = heightFor(tube.color);
    nominal.push({
      base: {x: tube.position_mm.x, y: tube.position_mm.y, z: 0},
      top: {x: tube.position_mm.x, y: tube.position_mm.y, z: height},
      height,
      color: tube.color,
      radius,
    });
  }

  const spanX = Math.abs(pitch.x) * Math.max(1, columns);
  const spanY = Math.abs(pitch.y) * Math.max(1, rows);
  return {
    rows,
    columns,
    origin,
    pitch,
    heights,
    extent: Math.max(spanX, spanY),
    center: {
      x: origin.x + (pitch.x * (columns - 1)) / 2,
      y: origin.y + (pitch.y * (rows - 1)) / 2,
      z: 14,
    },
    detected,
    nominal,
    camera: data?.camera || null,
  };
}

function updateCaps3dChrome(scene, data) {
  const caps = data?.caps || [];
  const placed = data?.placed_count || 0;
  const live = Boolean(data && data.available);
  if (!live) {
    setChip("caps3d-chip", "CAMERA OFF", "neutral");
  } else if (placed > 0) {
    setChip("caps3d-chip", `3D · ${caps.length} CAP${caps.length === 1 ? "" : "S"}`, "ready");
  } else if (caps.length > 0) {
    setChip("caps3d-chip", `3D · ${caps.length} UNPLACED`, "warning");
  } else {
    setChip("caps3d-chip", "3D · NO CAPS", "warning");
  }

  let message = "";
  if (!live) message = "Turn on the camera preview to place detected caps in 3D.";
  else if (data && !data.machine?.known) message = "Gantry position is unknown, so caps cannot be placed yet. Home the machine and keep Klipper connected.";
  else if (caps.length === 0) message = "No caps are visible in the current frame.";
  else if (placed === 0) message = `${caps.length} cap(s) seen, but none has a known colour, so no height can be derived.`;
  const empty = el("caps3d-empty");
  empty.hidden = !message;
  empty.textContent = message;

  const parts = [];
  if (data?.machine?.known) parts.push(`gantry X ${number(data.machine.x)} Y ${number(data.machine.y)} Z ${number(data.machine.z)} mm`);
  if (data?.camera) {
    const height = data.camera.height_mm + data.camera.raise_mm;
    parts.push(`camera ${number(height)} mm above rack · tilt ${number(data.camera.tilt_deg)}°`);
  }
  if (data) parts.push(`${placed} of ${caps.length} caps placed`);
  if (data?.unplaced_count) parts.push(`${data.unplaced_count} without a colour height`);
  if (scene?.detected?.length) {
    const maxTilt = scene.detected.reduce((max, tube) => Math.max(max, caps3dTiltDeg(tube)), 0);
    if (maxTilt >= 1) parts.push(`max lean ${number(maxTilt)}°`);
  }
  setText("caps3d-details", parts.length ? `${parts.join(" · ")}. Drag to orbit, scroll to zoom.` : "Drag to orbit, scroll to zoom.");

  const legend = el("caps3d-legend");
  legend.replaceChildren();
  const colors = new Map();
  for (const cap of caps) {
    if (!cap.placed) continue;
    const key = cap.color || "unknown";
    if (!colors.has(key)) colors.set(key, cap.cap_height_mm);
  }
  for (const [color, height] of colors) {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.className = "caps3d-swatch";
    swatch.style.background = capColor(color);
    item.append(swatch, document.createTextNode(`${color} · ${number(height)} mm`));
    legend.append(item);
  }
}

function caps3dProjector(width, height, cam) {
  const {target, yaw, pitch, distance} = cam;
  const eye = {
    x: target.x + distance * Math.cos(pitch) * Math.cos(yaw),
    y: target.y + distance * Math.cos(pitch) * Math.sin(yaw),
    z: target.z + distance * Math.sin(pitch),
  };
  const forward = caps3dNorm(caps3dSub(target, eye));
  let right = caps3dNorm(caps3dCross(forward, {x: 0, y: 0, z: 1}));
  if (!Number.isFinite(right.x) || Math.hypot(right.x, right.y, right.z) < 1e-6) right = {x: 1, y: 0, z: 0};
  const up = caps3dCross(right, forward);
  const focal = (height / 2) / Math.tan(CAPS3D_FOV / 2);
  const project = point => {
    const delta = caps3dSub(point, eye);
    const depth = caps3dDot(delta, forward);
    if (!Number.isFinite(depth) || depth <= 2) return null;
    return {
      x: width / 2 + (focal * caps3dDot(delta, right)) / depth,
      y: height / 2 - (focal * caps3dDot(delta, up)) / depth,
      depth,
    };
  };
  return {eye, project};
}

function caps3dDrawRack(ctx, view, scene) {
  const {origin, pitch, rows, columns} = scene;
  const pad = 0.55;
  const x0 = origin.x - pitch.x * pad;
  const x1 = origin.x + pitch.x * (columns - 1) + pitch.x * pad;
  const y0 = origin.y - pitch.y * pad;
  const y1 = origin.y + pitch.y * (rows - 1) + pitch.y * pad;
  const corners = [
    {x: x0, y: y0, z: 0},
    {x: x1, y: y0, z: 0},
    {x: x1, y: y1, z: 0},
    {x: x0, y: y1, z: 0},
  ].map(point => view.project(point));
  if (corners.some(point => !point)) return;
  caps3dPoly(ctx, corners, "rgba(128, 140, 152, .10)", "rgba(128, 140, 152, .40)", 1.25);

  ctx.save();
  ctx.strokeStyle = "rgba(128, 140, 152, .28)";
  ctx.lineWidth = 1;
  for (let column = 0; column < columns; column++) {
    const x = origin.x + column * pitch.x;
    const start = view.project({x, y: y0, z: 0});
    const end = view.project({x, y: y1, z: 0});
    if (start && end) caps3dLine(ctx, start, end);
  }
  for (let row = 0; row < rows; row++) {
    const y = origin.y + row * pitch.y;
    const start = view.project({x: x0, y, z: 0});
    const end = view.project({x: x1, y, z: 0});
    if (start && end) caps3dLine(ctx, start, end);
  }
  ctx.restore();

  const holeRadius = Math.min(Math.abs(pitch.x), Math.abs(pitch.y)) * 0.27;
  ctx.save();
  ctx.strokeStyle = "rgba(138, 148, 160, .55)";
  ctx.lineWidth = 1.1;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const center = {x: origin.x + column * pitch.x, y: origin.y + row * pitch.y, z: 0};
      const ring = caps3dRingPoints(center, holeRadius, 0, 14).map(point => view.project(point));
      if (ring.every(point => point)) caps3dPoly(ctx, ring, null, "rgba(138, 148, 160, .55)", 1);
    }
  }
  ctx.restore();
}

function caps3dDrawTube(ctx, view, tube) {
  const segments = 18;
  // The tube leans in its slot: base is the snapped slot, top is the detected cap.
  const axis = caps3dNorm(caps3dSub(tube.top, tube.base));
  const helper = Math.abs(axis.z) < 0.9 ? {x: 0, y: 0, z: 1} : {x: 1, y: 0, z: 0};
  const u = caps3dNorm(caps3dCross(axis, helper));
  const v = caps3dNorm(caps3dCross(axis, u));
  const baseWorld = [];
  const topWorld = [];
  for (let index = 0; index < segments; index++) {
    const angle = (index / segments) * Math.PI * 2;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const offset = {
      x: (u.x * cos + v.x * sin) * tube.radius,
      y: (u.y * cos + v.y * sin) * tube.radius,
      z: (u.z * cos + v.z * sin) * tube.radius,
    };
    baseWorld.push({x: tube.base.x + offset.x, y: tube.base.y + offset.y, z: tube.base.z + offset.z});
    topWorld.push({x: tube.top.x + offset.x, y: tube.top.y + offset.y, z: tube.top.z + offset.z});
  }
  const base = baseWorld.map(point => view.project(point));
  const top = topWorld.map(point => view.project(point));
  if (base.some(point => !point) || top.some(point => !point)) return;

  const color = capColor(tube.color);
  const alpha = tube.ghost ? 0.22 : 0.97;

  if (!tube.ghost) {
    const shadow = caps3dRingPoints({x: tube.base.x, y: tube.base.y}, tube.radius * 1.4, 0, segments)
      .map(point => view.project(point));
    if (shadow.every(point => point)) caps3dPoly(ctx, shadow, "rgba(0, 0, 0, .12)", null, 0);
    if (tube.slot) {
      // Ring on the snapped slot so the lean is readable even when the base is hidden.
      const marker = caps3dRingPoints({x: tube.base.x, y: tube.base.y}, tube.radius * 1.55, 0, segments)
        .map(point => view.project(point));
      if (marker.every(point => point)) caps3dPoly(ctx, marker, null, caps3dRgba(color, .5), 1.2);
    }
  }

  const light = caps3dNorm({x: -0.45, y: -0.5, z: 1});
  for (let index = 0; index < segments; index++) {
    const next = (index + 1) % segments;
    const mid = ((index + 0.5) / segments) * Math.PI * 2;
    const normal = {
      x: u.x * Math.cos(mid) + v.x * Math.sin(mid),
      y: u.y * Math.cos(mid) + v.y * Math.sin(mid),
      z: u.z * Math.cos(mid) + v.z * Math.sin(mid),
    };
    const centerX = (baseWorld[index].x + baseWorld[next].x + topWorld[index].x + topWorld[next].x) / 4;
    const centerY = (baseWorld[index].y + baseWorld[next].y + topWorld[index].y + topWorld[next].y) / 4;
    const centerZ = (baseWorld[index].z + baseWorld[next].z + topWorld[index].z + topWorld[next].z) / 4;
    if (caps3dDot(normal, {x: view.eye.x - centerX, y: view.eye.y - centerY, z: view.eye.z - centerZ}) <= 0) continue;
    const shade = 0.62 + 0.38 * Math.max(0, caps3dDot(normal, light));
    caps3dPoly(ctx, [base[index], base[next], top[next], top[index]], caps3dShade(color, shade, alpha), null, 0);
  }

  caps3dPoly(
    ctx,
    top,
    tube.ghost ? caps3dRgba(color, 0.25) : caps3dTint(color, 0.14),
    tube.focus ? "rgba(255, 255, 255, .95)" : caps3dRgba(color, tube.ghost ? 0.4 : 0.9),
    tube.focus ? 2 : 1,
  );
}

function caps3dDrawCamera(ctx, view, scene) {
  const camera = scene.camera;
  if (!camera) return;
  const apex = view.project({x: camera.x, y: camera.y, z: camera.z});
  if (!apex) return;
  const half = 44;
  const corners = [
    {x: camera.x - half, y: camera.y - half, z: 0},
    {x: camera.x + half, y: camera.y - half, z: 0},
    {x: camera.x + half, y: camera.y + half, z: 0},
    {x: camera.x - half, y: camera.y + half, z: 0},
  ].map(point => view.project(point));
  ctx.save();
  ctx.setLineDash([4, 5]);
  ctx.strokeStyle = "rgba(255, 168, 64, .55)";
  ctx.lineWidth = 1;
  for (const corner of corners) if (corner) caps3dLine(ctx, apex, corner);
  ctx.setLineDash([]);
  if (corners.every(point => point)) caps3dPoly(ctx, corners, null, "rgba(255, 168, 64, .35)", 1);
  ctx.fillStyle = "#ffa840";
  ctx.beginPath();
  ctx.arc(apex.x, apex.y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
  caps3dLabel(ctx, "camera", apex.x + 7, apex.y - 6);
}

function caps3dDrawLabels(ctx, view, scene) {
  for (const tube of scene.detected) {
    const point = view.project({x: tube.top.x, y: tube.top.y, z: tube.top.z + 2});
    if (!point) continue;
    const tilt = caps3dTiltDeg(tube);
    const suffix = tilt >= 1 ? ` · ${Math.round(tilt)}°` : "";
    caps3dLabel(ctx, `${tube.color || "?"} ${Math.round(tube.height)}mm${suffix}`, point.x + 5, point.y - 5);
  }
}

function caps3dLabel(ctx, text, x, y) {
  ctx.save();
  ctx.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
  const width = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(0, 0, 0, .55)";
  ctx.beginPath();
  if (typeof ctx.roundRect === "function") ctx.roundRect(x, y - 12, width + 8, 16, 4);
  else ctx.rect(x, y - 12, width + 8, 16);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, x + 4, y);
  ctx.restore();
}

function caps3dRingPoints(center, radius, z, segments) {
  const points = [];
  for (let index = 0; index < segments; index++) {
    const angle = (index / segments) * Math.PI * 2;
    points.push({x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius, z});
  }
  return points;
}

function caps3dPoly(ctx, points, fill, stroke, width) {
  if (!points.length || points.some(point => !point)) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index++) ctx.lineTo(points[index].x, points[index].y);
  ctx.closePath();
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke && width) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}

function caps3dLine(ctx, start, end) {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.lineTo(end.x, end.y);
  ctx.stroke();
}

function caps3dDistance2(eye, tube) {
  const dx = eye.x - (tube.base.x + tube.top.x) / 2;
  const dy = eye.y - (tube.base.y + tube.top.y) / 2;
  const dz = eye.z - (tube.base.z + tube.top.z) / 2;
  return dx * dx + dy * dy + dz * dz;
}

function caps3dTiltDeg(tube) {
  const horizontal = Math.hypot(tube.top.x - tube.base.x, tube.top.y - tube.base.y);
  const vertical = Math.max(1e-6, tube.top.z - tube.base.z);
  return (Math.atan2(horizontal, vertical) * 180) / Math.PI;
}

function caps3dSub(a, b) { return {x: a.x - b.x, y: a.y - b.y, z: a.z - b.z}; }
function caps3dDot(a, b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
function caps3dCross(a, b) {
  return {x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x};
}
function caps3dNorm(value) {
  const length = Math.hypot(value.x, value.y, value.z) || 1;
  return {x: value.x / length, y: value.y / length, z: value.z / length};
}
function caps3dClamp(value, min, max) { return Math.min(max, Math.max(min, value)); }

function caps3dRgb(hex) {
  const cleaned = String(hex || "").replace("#", "");
  const full = cleaned.length === 3 ? cleaned.split("").map(char => char + char).join("") : cleaned;
  const value = parseInt(full, 16);
  if (!Number.isFinite(value)) return {r: 154, g: 154, b: 154};
  return {r: (value >> 16) & 255, g: (value >> 8) & 255, b: value & 255};
}
function caps3dRgba(hex, alpha) {
  const {r, g, b} = caps3dRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
function caps3dTint(hex, amount) {
  const {r, g, b} = caps3dRgb(hex);
  const mix = channel => Math.round(channel + (255 - channel) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}
function caps3dShade(hex, factor, alpha) {
  const {r, g, b} = caps3dRgb(hex);
  const scale = channel => Math.round(channel * factor);
  return `rgba(${scale(r)}, ${scale(g)}, ${scale(b)}, ${alpha})`;
}

function wireControls() {
  el("home-button").addEventListener("click", () => act("/api/actions/home"));
  el("preview-button").addEventListener("click", preview);
  el("locate-selected-tube-button").addEventListener("click", beginCameraPickup);
  el("stereo-test-button").addEventListener("click", runStereoTest);
  el("run-next-pickup-step-button").addEventListener("click", runNextCameraPickupStep);
  el("cancel-pickup-steps-button").addEventListener("click", cancelCameraPickup);
  el("start-button").addEventListener("click", async () => {
    const degraded = Boolean(state.snapshot?.capabilities.degraded_mode);
    const warning = degraded ? "\n\nDegraded mode will keep pickup and release active, but skip QR decoding and rotary scan moves." : "";
    if (confirm(`Start scanning ${state.snapshot?.workflow.progress.total_tubes || 0} tubes?${warning}`)) await act("/api/workflow/start", {degraded_mode: degraded});
  });
  el("pause-button").addEventListener("click", () => act(state.snapshot?.workflow.state === "paused" ? "/api/workflow/resume" : "/api/workflow/pause"));
  el("stop-button").addEventListener("click", async () => { if (confirm("Stop after the active command? This is not an emergency stop.")) await act("/api/workflow/stop"); });
  el("tooling-release-button").addEventListener("click", () => act("/api/tooling/release"));
  el("tooling-vacuum-button").addEventListener("click", () => act("/api/tooling/vacuum"));
  el("tooling-vacuum-off-button").addEventListener("click", () => act("/api/tooling/vacuum/off"));
  el("tooling-zero-button").addEventListener("click", () => act("/api/tooling/rotary/zero"));
  el("rotary-move-button").addEventListener("click", sendRotaryMove);
  el("rotary-degrees-input").addEventListener("input", event => { el("rotary-move-button").disabled = event.target.disabled || !isWholeNumber(event.target.value); });
}
async function act(path, body = {}) { try { const result = await api(path, body); announce(result.message); } catch (error) { announce(error.message); alert(error.message); } }
async function preview() { try { const result = await api("/api/actions/preview"); const panel = el("preview-result"); panel.hidden = false; panel.textContent = `${result.plan.tube_count} tubes · ${result.plan.step_count} motion steps · ${result.plan.yaw_angles_deg.length} yaw angles. ${result.validation.valid ? "Validation passed." : result.validation.issues.map(issue => issue.message).join(" ")}`; } catch (error) { alert(error.message); } }
function isWholeNumber(value) { return /^-?\d+$/.test(String(value).trim()); }
async function sendRotaryMove() {
  const input = el("rotary-degrees-input");
  if (!isWholeNumber(input.value)) return;
  try {
    const result = await api("/api/tooling/rotary/move", {degrees: Number(input.value)});
    announce(result.message);
  } catch (error) { announce(error.message); alert(error.message); }
}
async function beginCameraPickup() {
  if (!state.selectedTube) return;
  const [row, column] = state.selectedTube.split(":").map(Number);
  state.cameraPickupPreviewWasEnabled = state.cameraEnabled;
  state.cameraLocateBusy = true;
  el("locate-tube-steps").replaceChildren();
  setText("locate-tube-status", `Preparing the camera for R${row} C${column}; no motion will occur until you run a step.`);
  try {
    if (!state.cameraEnabled) await setCameraPreviewEnabled(true);
    if (!state.cameraEnabled) throw new Error("Camera preview could not be enabled.");
    const session = await api("/api/actions/locate-tube", {row, column});
    renderCameraPickupSession(session);
    announce("Pickup step-through ready. No movement has been made yet.");
  } catch (error) {
    setText("locate-tube-status", error.message);
    announce(error.message);
    alert(error.message);
    if (!state.cameraPickupPreviewWasEnabled && state.cameraEnabled) await setCameraPreviewEnabled(false);
  } finally {
    state.cameraLocateBusy = false;
    if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  }
}

async function runStereoTest() {
  if (!state.selectedTube || state.stereoTestBusy) return;
  const [row, column] = state.selectedTube.split(":").map(Number);
  state.stereoTestBusy = true;
  setText("stereo-test-result", `Testing R${row} C${column}: moving through the two stereo positions at safe Z.`);
  if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  try {
    const result = await api("/api/actions/stereo-test", {row, column});
    const position = result.position_relative_to_center_mm;
    setText(
      "stereo-test-result",
      `R${row} C${column} relative to scan midpoint: X ${number(position.x)} mm, Y ${number(position.y)} mm, Z ${number(position.z)} mm (10 mm baseline). Camera returned to midpoint; no pickup performed.`,
    );
  } catch (error) {
    setText("stereo-test-result", `Stereo test failed: ${error.message}`);
    alert(error.message);
  } finally {
    state.stereoTestBusy = false;
    if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  }
}

async function runNextCameraPickupStep() {
  const session = state.cameraPickupSession;
  if (!session || session.state !== "running") return;
  state.cameraLocateBusy = true;
  setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  try {
    const updated = await api("/api/actions/locate-tube/step", {session_id: session.session_id});
    renderCameraPickupSession(updated);
    if (updated.state === "completed") {
      announce("Pickup and deposit complete. Tube returned to the selected slot; vacuum is off.");
      if (!state.cameraPickupPreviewWasEnabled) await setCameraPreviewEnabled(false);
    }
  } catch (error) {
    session.state = "failed";
    if (session.steps[session.next_step]) session.steps[session.next_step].state = "failed";
    renderCameraPickupSession(session);
    setText("locate-tube-status", error.message);
    announce(error.message);
    alert(error.message);
  } finally {
    state.cameraLocateBusy = false;
    if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  }
}

async function cancelCameraPickup() {
  const session = state.cameraPickupSession;
  if (!session || !["running", "failed"].includes(session.state)) return;
  state.cameraLocateBusy = true;
  setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  try {
    const result = await api("/api/actions/locate-tube/cancel", {session_id: session.session_id});
    renderCameraPickupSession(result);
    announce("Pickup step-through cancelled.");
    if (!state.cameraPickupPreviewWasEnabled) await setCameraPreviewEnabled(false);
  } catch (error) {
    setText("locate-tube-status", error.message);
    announce(error.message);
    alert(error.message);
  } finally {
    state.cameraLocateBusy = false;
    if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  }
}
function announce(message) { setText("announcer", message); }

document.addEventListener("DOMContentLoaded", bootstrap);
