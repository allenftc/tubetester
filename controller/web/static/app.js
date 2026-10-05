"use strict";

const LEVELS = ["debug", "info", "warning", "error"];
const SOURCES = ["controller", "workflow", "moonraker", "klipper", "camera", "qr", "user"];
const ACTIVE_TUBES = new Set(["approaching", "picked_up", "scanning"]);
const state = {
  snapshot: null,
  events: new Map(),
  selectedTube: null,
  lastSequence: 0,
  reconnectDelay: 500,
  socket: null,
  cameraEnabled: false,
  cameraAvailable: false,
  cameraImageFailed: false,
  cameraPollTimer: null,
  cameraLocateBusy: false,
  cameraPickupSession: null,
  cameraPickupPreviewWasEnabled: true,
  clearWatermark: sessionStorage.getItem("console-clear-watermark"),
  filters: loadJSON("console-filters", {levels: LEVELS.filter(level => level !== "debug"), sources: SOURCES}),
};
const el = id => document.getElementById(id);

function loadJSON(key, fallback) {
  try { return {...fallback, ...JSON.parse(localStorage.getItem(key) || "null")}; }
  catch (_) { return fallback; }
}
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
  buildFilters();
  applyCollapsedPreference();
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
  image.addEventListener("error", () => { state.cameraImageFailed = true; });
  image.addEventListener("load", () => { state.cameraImageFailed = false; });
  el("camera-preview-toggle").addEventListener("change", event => setCameraPreviewEnabled(event.target.checked));
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
    if (state.cameraEnabled) refreshCameraStatus();
  } catch (error) {
    setText("camera-details", error.message);
  }
}

async function setCameraPreviewEnabled(enabled) {
  const toggle = el("camera-preview-toggle");
  const image = el("camera-stream");
  toggle.disabled = true;
  if (!enabled) {
    state.cameraEnabled = false;
    window.clearTimeout(state.cameraPollTimer);
    image.removeAttribute("src");
    image.hidden = true;
    state.cameraAvailable = false;
    state.cameraImageFailed = false;
    setChip("camera-chip", "PREVIEW OFF", "neutral");
    setText("camera-details", "Preview is off. Camera capture is stopped.");
  }
  try {
    const result = await api("/api/camera/preview", {enabled});
    state.cameraEnabled = result.enabled;
    if (enabled) {
      setChip("camera-chip", "CAMERA STARTING", "warning");
      setText("camera-details", `Opening ${result.camera.device}…`);
      refreshCameraStatus();
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
    }
  } finally {
    toggle.disabled = state.cameraLocateBusy || ["running", "failed"].includes(state.cameraPickupSession?.state);
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
  if (available && (!state.cameraAvailable || state.cameraImageFailed)) {
    el("camera-stream").hidden = false;
    el("camera-stream").src = `/api/camera/stream?retry=${Date.now()}`;
    state.cameraImageFailed = false;
  } else if (!available) {
    el("camera-stream").removeAttribute("src");
    el("camera-stream").hidden = true;
  }
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
    if (message.type === "console.event") upsertEvent(message.payload);
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
  el("gcode-input").disabled = !caps.send_gcode;
  el("send-button").disabled = !caps.send_gcode || !el("gcode-input").value.trim();
  ["tooling-release-button", "tooling-vacuum-button", "tooling-vacuum-off-button", "tooling-zero-button", "rotary-degrees-input"].forEach(id => { el(id).disabled = !caps.tooling; });
  el("rotary-move-button").disabled = !caps.tooling || !isWholeNumber(el("rotary-degrees-input").value);
  const pickupRunning = state.cameraPickupSession?.state === "running";
  const pickupFailed = state.cameraPickupSession?.state === "failed";
  const pickupLocked = pickupRunning || pickupFailed;
  el("locate-selected-tube-button").disabled = !caps.send_gcode || !state.selectedTube || state.cameraLocateBusy || pickupRunning;
  el("run-next-pickup-step-button").hidden = !pickupRunning;
  el("run-next-pickup-step-button").disabled = state.cameraLocateBusy || !pickupRunning;
  el("cancel-pickup-steps-button").hidden = !pickupRunning && !pickupFailed;
  el("cancel-pickup-steps-button").disabled = state.cameraLocateBusy || (!pickupRunning && !pickupFailed);
  el("camera-preview-toggle").disabled = state.cameraLocateBusy || pickupLocked;
  ["macro-calibrate-button", "macro-pickup-button", "macro-deposit-button"].forEach(id => { el(id).disabled = !caps.send_gcode; });
}

function renderStepper(currentPhase, workflowState) {
  const phases = ["home", "approach", "pickup", "scan", "release"];
  const index = phases.indexOf(currentPhase);
  document.querySelectorAll("#phase-stepper li").forEach((node, i) => {
    node.className = i < index || (!currentPhase && workflowState === "completed") ? "complete" : i === index ? "current" : "";
    node.setAttribute("aria-current", i === index ? "step" : "false");
  });
}

function renderRack(rack) {
  setText("rack-title", `Rack ${rack.rows} × ${rack.columns}`);
  const grid = el("rack-grid");
  grid.style.gridTemplateColumns = `42px repeat(${rack.columns}, minmax(48px, 1fr))`;
  grid.replaceChildren();
  const corner = document.createElement("span"); corner.className = "rack-corner"; corner.setAttribute("aria-hidden", "true"); grid.append(corner);
  for (let column = 1; column <= rack.columns; column++) grid.append(label(`C${column}`, "column"));
  for (let row = 1; row <= rack.rows; row++) {
    grid.append(label(`R${row}`, "row"));
    for (let column = 1; column <= rack.columns; column++) {
      const tube = rack.tubes.find(item => item.row === row && item.column === column);
      const button = document.createElement("button");
      const visual = ACTIVE_TUBES.has(tube.status) ? "active" : tube.status;
      button.type = "button";
      button.className = `rack-cell ${classToken(visual)}`;
      if (state.selectedTube === `${row}:${column}`) button.classList.add("selected");
      button.setAttribute("role", "gridcell");
      button.setAttribute("aria-label", `Row ${row}, column ${column}, ${title(tube.status)}`);
      button.dataset.row = row; button.dataset.column = column;
      button.addEventListener("click", () => selectTube(tube));
      button.addEventListener("keydown", rackKeydown);
      grid.append(button);
    }
  }
  if (state.selectedTube) {
    const [row, column] = state.selectedTube.split(":").map(Number);
    const selected = rack.tubes.find(tube => tube.row === row && tube.column === column);
    if (selected) showTubeDetails(selected);
  }
}
function label(text, kind) { const node = document.createElement("span"); node.className = `rack-label ${kind}`; node.textContent = text; node.setAttribute("aria-hidden", "true"); return node; }
function selectTube(tube) { state.selectedTube = `${tube.row}:${tube.column}`; renderRack(state.snapshot.rack); showTubeDetails(tube); renderLocateTubePlan(tube); document.querySelector(`[data-row="${tube.row}"][data-column="${tube.column}"]`)?.focus(); if (state.snapshot) setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state); }
function renderLocateTubePlan(tube) {
  const rack = state.snapshot.rack;
  const camera = rack.camera_offset_mm;
  const cameraX = tube.position_mm.x - camera.x;
  const cameraY = tube.position_mm.y - camera.y;
  const steps = [
    `Raise to safe Z ${number(rack.safe_z_mm)} mm`,
    `Move camera to X ${number(cameraX)} Y ${number(cameraY)} mm`,
    "Scan ROI; detect center and calculate pixel correction",
    "Move gripper to corrected tube XY at safe Z",
    "Turn vacuum on",
    `Lower to pickup Z ${number(rack.pickup_height_mm)} mm`,
    `Lift to safe Z ${number(rack.safe_z_mm)} mm; vacuum remains on`,
  ];
  setText("locate-tube-plan", `R${tube.row} C${tube.column} plan: camera X ${number(cameraX)} Y ${number(cameraY)} mm; nominal tube XY X ${number(tube.position_mm.x)} Y ${number(tube.position_mm.y)} mm; pixel scale ${number(rack.pixel_to_mm_multiplier)} mm/px.`);
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
    item.textContent = `${index + 1}. ${step.label} [${step.state}]${step.command && step.command !== "CAMERA_DETECT" && step.command !== "WAITING_FOR_DETECTION" ? ` | ${step.command.replaceAll("\n", " ; ")}` : ""}`;
    list.append(item);
  });
  if (session.detected_center_px) {
    const detail = document.createElement("li");
    detail.className = "pickup-step completed";
    detail.textContent = `Detection: center px (${session.detected_center_px.join(", ")}); correction X ${number(session.correction_mm.x)} Y ${number(session.correction_mm.y)} mm; target X ${number(session.target_xy.x)} Y ${number(session.target_xy.y)} mm.`;
    list.append(detail);
  }
  const current = session.steps[session.next_step];
  if (session.state === "running" && current) {
    setText("locate-tube-status", `Paused before step ${session.next_step + 1}: ${current.label}. Press Run Next Step to execute only this step.`);
    el("run-next-pickup-step-button").textContent = `Run Step ${session.next_step + 1} of ${session.steps.length}`;
  } else if (session.state === "completed") {
    setText("locate-tube-status", "Pickup sequence complete. Vacuum is still on; turn it off manually when ready.");
  } else if (session.state === "cancelled") {
    setText("locate-tube-status", `Step-through cancelled. Vacuum ${session.vacuum_enabled ? "remains on" : "is off"}.`);
  } else if (session.state === "failed") {
    setText("locate-tube-status", "A step failed. Review the error, then cancel to clean up the session.");
  }
}
function showTubeDetails(tube) {
  const p = tube.position_mm;
  const details = [`R${tube.row} C${tube.column}`, title(tube.status), `attempts ${tube.yaw_attempt}/${tube.yaw_attempt_total}`, `X ${number(p.x)} Y ${number(p.y)} Z ${number(p.z)}`];
  if (tube.decoded_payload) details.push(`payload: ${tube.decoded_payload}`);
  if (tube.confidence != null) details.push(`confidence ${number(tube.confidence)}`);
  if (tube.error) details.push(`error: ${tube.error}`);
  setText("tube-details", details.join(" · "));
}
function rackKeydown(event) {
  const keyMap = {ArrowLeft: [0,-1], ArrowRight:[0,1], ArrowUp:[-1,0], ArrowDown:[1,0]};
  if (!keyMap[event.key]) return;
  event.preventDefault();
  const [dr, dc] = keyMap[event.key];
  const target = document.querySelector(`[data-row="${Number(event.currentTarget.dataset.row)+dr}"][data-column="${Number(event.currentTarget.dataset.column)+dc}"]`);
  target?.focus();
}

function upsertEvent(event) {
  state.events.set(event.id, event);
  if (event.source === "camera" && event.message.startsWith("Tube debug step ")) {
    const item = document.createElement("li");
    item.textContent = event.message;
    el("locate-tube-steps").append(item);
  }
  if (event.source === "camera" && event.message.startsWith("Tube debug detected center ")) {
    setText("locate-tube-status", event.message);
  }
  if (state.events.size > 500) {
    const oldest = [...state.events.values()].sort((a,b) => Date.parse(a.timestamp)-Date.parse(b.timestamp))[0];
    state.events.delete(oldest.id);
  }
  renderConsole();
}
function renderConsole() {
  const log = el("console-log");
  const events = [...state.events.values()]
    .filter(event => state.filters.levels.includes(event.level) && state.filters.sources.includes(event.source))
    .filter(event => !state.clearWatermark || Date.parse(event.timestamp) > Date.parse(state.clearWatermark))
    .sort((a,b) => Date.parse(b.timestamp)-Date.parse(a.timestamp)).slice(0,300);
  log.replaceChildren();
  if (!events.length) { const empty = document.createElement("div"); empty.className = "console-empty"; empty.textContent = "No console messages yet."; log.append(empty); return; }
  events.forEach(event => {
    const row = document.createElement("div"); row.className = `event-row ${classToken(event.level)}`;
    const date = new Date(event.timestamp);
    row.title = `${date.toLocaleString()} · ${event.source}`;
    row.setAttribute("aria-label", `${date.toLocaleString()}, ${event.level}, ${event.source}, ${event.message}, repeated ${event.repeat_count} times`);
    const time = document.createElement("time"); time.className = "event-time"; time.dateTime = event.timestamp; time.textContent = date.toLocaleTimeString([], {hour:"numeric",minute:"2-digit",hour12:true});
    const level = document.createElement("span"); level.className = "event-level"; level.textContent = event.level.toUpperCase();
    const message = document.createElement("span"); message.className = "event-message"; message.textContent = event.message;
    const repeat = document.createElement("span"); repeat.className = "event-repeat"; repeat.textContent = event.repeat_count > 1 ? `${event.repeat_count}×` : "";
    row.append(time, level, message, repeat); log.append(row);
  });
}

function buildFilters() {
  [["level-filters", LEVELS, "levels"], ["source-filters", SOURCES, "sources"]].forEach(([id, values, key]) => {
    values.forEach(value => {
      const label = document.createElement("label"); const input = document.createElement("input");
      input.type = "checkbox"; input.checked = state.filters[key].includes(value); input.value = value;
      input.addEventListener("change", () => { state.filters[key] = [...el(id).querySelectorAll("input:checked")].map(node => node.value); localStorage.setItem("console-filters", JSON.stringify(state.filters)); renderConsole(); });
      label.append(input, document.createTextNode(` ${title(value)}`)); el(id).append(label);
    });
  });
}

function wireControls() {
  el("home-button").addEventListener("click", () => act("/api/actions/home"));
  el("preview-button").addEventListener("click", preview);
  el("locate-selected-tube-button").addEventListener("click", beginCameraPickup);
  el("run-next-pickup-step-button").addEventListener("click", runNextCameraPickupStep);
  el("cancel-pickup-steps-button").addEventListener("click", cancelCameraPickup);
  el("start-button").addEventListener("click", async () => {
    const degraded = Boolean(state.snapshot?.capabilities.degraded_mode);
    const warning = degraded ? "\n\nDegraded mode will keep pickup and release active, but skip QR decoding and rotary scan moves." : "";
    if (confirm(`Start scanning ${state.snapshot?.workflow.progress.total_tubes || 0} tubes?${warning}`)) await act("/api/workflow/start", {degraded_mode: degraded});
  });
  el("pause-button").addEventListener("click", () => act(state.snapshot?.workflow.state === "paused" ? "/api/workflow/resume" : "/api/workflow/pause"));
  el("stop-button").addEventListener("click", async () => { if (confirm("Stop after the active command? This is not an emergency stop.")) await act("/api/workflow/stop"); });
  el("macro-calibrate-button").addEventListener("click", () => act("/api/macros/calibrate"));
  el("macro-pickup-button").addEventListener("click", () => act("/api/macros/pickup"));
  el("macro-deposit-button").addEventListener("click", () => act("/api/macros/deposit"));
  el("tooling-release-button").addEventListener("click", () => act("/api/tooling/release"));
  el("tooling-vacuum-button").addEventListener("click", () => act("/api/tooling/vacuum"));
  el("tooling-vacuum-off-button").addEventListener("click", () => act("/api/tooling/vacuum/off"));
  el("tooling-zero-button").addEventListener("click", () => act("/api/tooling/rotary/zero"));
  el("rotary-move-button").addEventListener("click", sendRotaryMove);
  el("rotary-degrees-input").addEventListener("input", event => { el("rotary-move-button").disabled = event.target.disabled || !isWholeNumber(event.target.value); });
  el("gcode-input").addEventListener("input", event => { el("send-button").disabled = event.target.disabled || !event.target.value.trim(); });
  el("gcode-form").addEventListener("submit", sendGcode);
  el("clear-console").addEventListener("click", () => { state.clearWatermark = new Date().toISOString(); sessionStorage.setItem("console-clear-watermark", state.clearWatermark); renderConsole(); });
  el("show-history").addEventListener("click", () => { state.clearWatermark = null; sessionStorage.removeItem("console-clear-watermark"); renderConsole(); });
  toggleButton("help-console", "help-panel"); toggleButton("settings-console", "settings-panel");
  el("collapse-console").addEventListener("click", () => { const collapsed = el("console-card").classList.toggle("collapsed"); localStorage.setItem("console-collapsed", String(collapsed)); el("collapse-console").setAttribute("aria-expanded", String(!collapsed)); });
}
function toggleButton(buttonId, panelId) { el(buttonId).addEventListener("click", () => { const hidden = !el(panelId).hidden; el(panelId).hidden = hidden; el(buttonId).setAttribute("aria-expanded", String(!hidden)); }); }
function applyCollapsedPreference() { if (localStorage.getItem("console-collapsed") === "true") { el("console-card").classList.add("collapsed"); el("collapse-console").setAttribute("aria-expanded", "false"); } }
async function act(path, body = {}) { try { const result = await api(path, body); announce(result.message); } catch (error) { announce(error.message); alert(error.message); } }
async function preview() { try { const result = await api("/api/actions/preview"); const panel = el("preview-result"); panel.hidden = false; panel.textContent = `${result.plan.tube_count} tubes · ${result.plan.step_count} motion steps · ${result.plan.yaw_angles_deg.length} yaw angles. ${result.validation.valid ? "Validation passed." : result.validation.issues.map(issue => issue.message).join(" ")}`; } catch (error) { alert(error.message); } }
async function sendGcode(event) { event.preventDefault(); const input = el("gcode-input"); const script = input.value.trim(); if (!script) return; try { await api("/api/gcode", {script}); input.value = ""; el("send-button").disabled = true; } catch (error) { alert(error.message); } }
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

async function runNextCameraPickupStep() {
  const session = state.cameraPickupSession;
  if (!session || session.state !== "running") return;
  state.cameraLocateBusy = true;
  setCapabilities(state.snapshot.capabilities, state.snapshot.workflow.state);
  try {
    const updated = await api("/api/actions/locate-tube/step", {session_id: session.session_id});
    renderCameraPickupSession(updated);
    if (updated.state === "completed") {
      announce("Pickup sequence complete. Vacuum remains on.");
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
