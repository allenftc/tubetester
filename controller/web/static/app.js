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
  state.lastSequence = Math.max(state.lastSequence, snapshot.sequence || 0);
  const machine = snapshot.machine;
  const workflow = snapshot.workflow;
  setChip("klipper-chip", `● Klipper ${title(machine.klipper_state)}`, machine.klipper_state === "ready" ? "ready" : machine.connected ? "warning" : "error");
  setChip("controller-chip", `● Controller ${title(workflow.state)}`, workflow.state);
  const p = machine.position_mm;
  setChip("position-chip", p ? `Position X ${number(p.x)} Y ${number(p.y)} Z ${number(p.z)}` : "Position Unknown", "neutral");
  setChip("workflow-chip", workflow.state.toUpperCase(), workflow.state);
  setChip("qr-chip", snapshot.capabilities.qr ? "QR READY" : "QR UNAVAILABLE", snapshot.capabilities.qr ? "ready" : "warning");
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
function selectTube(tube) { state.selectedTube = `${tube.row}:${tube.column}`; renderRack(state.snapshot.rack); showTubeDetails(tube); document.querySelector(`[data-row="${tube.row}"][data-column="${tube.column}"]`)?.focus(); }
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
  el("start-button").addEventListener("click", async () => {
    const degraded = Boolean(state.snapshot?.capabilities.degraded_mode);
    const warning = degraded ? "\n\nDegraded mode will skip pickup, QR/yaw, and release macro commands. Approach motion and homing are still real." : "";
    if (confirm(`Start scanning ${state.snapshot?.workflow.progress.total_tubes || 0} tubes?${warning}`)) await act("/api/workflow/start", {degraded_mode: degraded});
  });
  el("pause-button").addEventListener("click", () => act(state.snapshot?.workflow.state === "paused" ? "/api/workflow/resume" : "/api/workflow/pause"));
  el("stop-button").addEventListener("click", async () => { if (confirm("Stop after the active command? This is not an emergency stop.")) await act("/api/workflow/stop"); });
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
function announce(message) { setText("announcer", message); }

document.addEventListener("DOMContentLoaded", bootstrap);
