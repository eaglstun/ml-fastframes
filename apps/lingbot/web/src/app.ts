/*
 * For licensing see accompanying LICENSE file.
 * Copyright (C) 2026 Apple Inc. All Rights Reserved.
 *
 * Browser client of the LingBot demo. Edit apps/lingbot/web/src/app.ts, then run `npm run build`
 * in apps/lingbot/web: it compiles to apps/lingbot/static/app.js, which is committed so the demo
 * runs without Node (demo.py serves it as is).
 */

// ------------------------------------------------------------------ protocol (demo.py)

type Action = "idle" | "forward" | "backward" | "left" | "right" | "yaw_left" | "yaw_right";
type Access = "owner" | "queued" | "busy" | "available";

interface HelloMessage {
  type: "hello"; gpus: number; ready: boolean;
  sizes?: [number, number][];  // frame sizes a visitor may choose, the server's default first
}
interface QueueStateMessage {
  type: "queue_state"; enabled: boolean; status: Access; position: number | null;
  waiting_count: number; idle_remaining_seconds: number | null; idle_timeout_seconds: number;
}
interface SessionMessage { type: "session"; status: string; frames?: number; reason?: string | null }
interface StatusMessage {
  type: "status"; state: "preparing" | "ready" | "generating" | "stopping" | "stopped" | "complete";
  message?: string;
}
interface ChunkMessage {
  type: "chunk"; request_id: string | null; rgb_frames: number; frame_sizes: number[];
  queue_ms: number; chunk_index: number; model_ms: number; worker_ms: number;
  terminal: boolean; width?: number; height?: number;
}
interface PreviewMessage {
  type: "preview"; request_id: string | null; chunk_index: number; pass: number; passes: number; image: string;
}
interface ErrorMessage { type: "error"; code: string; message?: string }
type RecordingMessage =  // the ended world's video, saved by the server (demo_recording.py)
  | { type: "recording"; status: "saved"; url: string; name: string; path: string; frames: number; seconds: number;
      width: number; height: number }
  | { type: "recording"; status: "failed"; message: string };
type ServerMessage = HelloMessage | QueueStateMessage | SessionMessage | StatusMessage | ChunkMessage
  | PreviewMessage | ErrorMessage | RecordingMessage;

type ClientMessage =
  | { type: "start"; prompt: string; seed: number; image?: string; example?: string; size?: string }
  | { type: "stop" } | { type: "activity" } | { type: "join_queue" } | { type: "leave_queue" }
  | { type: "action"; action: Action; request_id: string }
  | { type: "displayed"; chunk_index: number };

interface Example { id: string; label: string; prompt: string; image_url: string }

// ------------------------------------------------------------------ elements (index.html)

function byId<T extends HTMLElement>(id: string, type: new () => T): T {
  const element = document.getElementById(id);
  if (!(element instanceof type)) throw new Error(`index.html has no <${type.name}> #${id}`);
  return element;
}

const ui = {
  stateLabel: byId("stateLabel", HTMLElement),
  toast: byId("toast", HTMLDivElement),
  startButton: byId("startButton", HTMLButtonElement),
  startButtonLabel: byId("startButtonLabel", HTMLElement),
  stopButton: byId("stopButton", HTMLButtonElement),
  keyboardStatus: byId("keyboardStatus", HTMLElement),
  transportLight: byId("transportLight", HTMLElement),
  takeDot: byId("takeDot", HTMLElement),
  takeLabel: byId("takeLabel", HTMLElement),
  previewLabel: byId("previewLabel", HTMLElement),
  worldFrame: byId("worldFrame", HTMLImageElement),
  stagePlaceholder: byId("stagePlaceholder", HTMLDivElement),
  modelLatency: byId("modelLatency", HTMLElement),
  nonModelLatency: byId("nonModelLatency", HTMLElement),
  totalLatency: byId("totalLatency", HTMLElement),
  latencySample: byId("latencySample", HTMLParagraphElement),
  promptInput: byId("promptInput", HTMLTextAreaElement),
  promptCount: byId("promptCount", HTMLElement),
  seedInput: byId("seedInput", HTMLInputElement),
  randomSeedButton: byId("randomSeedButton", HTMLButtonElement),
  exampleSelect: byId("exampleSelect", HTMLSelectElement),
  frameCount: byId("frameCount", HTMLElement),
  chunkCount: byId("chunkCount", HTMLElement),
  actionReadout: byId("actionReadout", HTMLElement),
  resolutionLabel: byId("resolutionLabel", HTMLElement),
  waitingRoom: byId("waitingRoom", HTMLElement),
  joinQueueButton: byId("joinQueueButton", HTMLButtonElement),
  leaveQueueButton: byId("leaveQueueButton", HTMLButtonElement),
  waitingRoomTitle: byId("waitingRoomTitle", HTMLElement),
  waitingRoomMessage: byId("waitingRoomMessage", HTMLParagraphElement),
  waitingRoomTimer: byId("waitingRoomTimer", HTMLElement),
  gpuLabel: byId("gpuLabel", HTMLElement),
  connectionLabel: byId("connectionLabel", HTMLElement),
  connectionPill: byId("connectionPill", HTMLElement),
  uploadButton: byId("uploadButton", HTMLButtonElement),
  imageInput: byId("imageInput", HTMLInputElement),
  imagePreview: byId("imagePreview", HTMLImageElement),
  imagePlaceholder: byId("imagePlaceholder", HTMLDivElement),
  imageName: byId("imageName", HTMLElement),
  sizeField: byId("sizeField", HTMLDivElement),
  sizeToggle: byId("sizeToggle", HTMLDivElement),
  sizeHint: byId("sizeHint", HTMLParagraphElement),
  recordingLink: byId("recordingLink", HTMLAnchorElement),
};

// ------------------------------------------------------------------ state

const FRAME_MS = 1000 / 50;
const HOLD_MS = 400;
const labels: Record<Action, string> = {
  forward: "Move forward", backward: "Move backward", left: "Move left",
  right: "Move right", yaw_left: "Pan left", yaw_right: "Pan right", idle: "Hold pose",
};
const keys: Partial<Record<string, Action>> = {
  KeyW: "forward", KeyS: "backward", KeyA: "left", KeyD: "right",
  ArrowLeft: "yaw_left", ArrowRight: "yaw_right",
};
const buttons = [...document.querySelectorAll<HTMLButtonElement>("button[data-action]")];

interface Held { action: Action; source: string; since: number }
interface Pending { action: Action; inputAt: number | null }

const state = {
  socket: null as WebSocket | null, connected: false, ready: false, access: "available" as Access,
  active: false, preparing: false, stopping: false, waiting: false, awaitingStart: false,
  playing: false, pending: null as Pending | null, held: null as Held | null,
  holdTimer: undefined as ReturnType<typeof setTimeout> | undefined,
  sequence: 0, epoch: 0, frames: 0, chunks: 0, image: null as string | null,
  examples: [] as Example[], requestTimes: new Map<string, number>(), metadata: [] as ChunkMessage[],
  frameURL: null as string | null, cancelLoad: null as (() => void) | null, lastActivity: -Infinity,
  sizes: [] as string[], size: null as string | null,  // "WxH"; null: the server's default
  sessionSize: null as string | null,  // the size of the world being explored
};
let toastTimer: ReturnType<typeof setTimeout> | undefined;

function buttonAction(button: HTMLButtonElement): Action {
  return button.dataset.action as Action;
}

// ------------------------------------------------------------------ helpers

function send(message: ClientMessage): boolean {
  if (state.socket?.readyState !== WebSocket.OPEN) return false;
  state.socket.send(JSON.stringify(message));
  return true;
}

function status(message: string): void { ui.stateLabel.textContent = message; }

function error(message: string): void {
  ui.toast.textContent = message;
  ui.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { ui.toast.hidden = true; }, 9000);
}

function canControl(): boolean {
  return state.connected && state.active && !state.preparing && !state.stopping && state.access === "owner";
}

function updateControls(): void {
  const allowed = state.access === "owner" || state.access === "available";
  ui.startButton.disabled = !state.connected || !allowed || state.preparing || state.stopping;
  ui.startButtonLabel.textContent = state.preparing ? "Preparing…" : state.active ? "Restart world" : "Start world";
  ui.stopButton.disabled = !state.connected || (!state.active && !state.preparing) || state.stopping;
  buttons.forEach((button) => { button.disabled = !canControl(); });
  ui.keyboardStatus.textContent = canControl() ? "KEYBOARD LIVE" : "STANDBY";
  ui.keyboardStatus.classList.toggle("active", canControl());
  ui.transportLight.classList.toggle("active", state.active);
  ui.takeDot.classList.toggle("active", state.active);
  ui.takeLabel.textContent = state.preparing ? "PREPARING" : state.active ? "LIVE WORLD" : "STANDBY";
  updateSizeHint();
}

function clearHeld(): void {
  state.held = null;
  clearTimeout(state.holdTimer);
  buttons.forEach((button) => button.classList.remove("pressed"));
}

function cancelPlayback(): void {
  state.epoch += 1;
  state.playing = false;
  state.waiting = false;
  state.pending = null;
  state.requestTimes.clear();
  state.cancelLoad?.();
  state.cancelLoad = null;
  ui.previewLabel.hidden = true;
  clearHeld();
}

function showPreview(message: PreviewMessage): void {
  // A rough still of the chunk being generated; the chunk's real frames replace it.
  if (!state.active || state.stopping || state.awaitingStart || state.playing || !state.waiting) return;
  if (typeof message.image !== "string" || !message.image.startsWith("data:image/jpeg;base64,")) return;
  ui.worldFrame.src = message.image;
  ui.worldFrame.hidden = false;
  ui.stagePlaceholder.hidden = true;
  ui.previewLabel.textContent = `PREVIEW ${message.pass}/${message.passes}`;
  ui.previewLabel.hidden = false;
  status(`Generating · rough preview ${message.pass} of ${message.passes}`);
}

function showRecording(message: RecordingMessage): void {
  if (message.status === "failed") { error(message.message); return; }
  if (!message.url.startsWith("/videos/")) return;
  ui.recordingLink.href = message.url;
  ui.recordingLink.download = message.name;
  ui.recordingLink.textContent = `Saved ${message.seconds.toFixed(1)} s · ${message.width} × ${message.height} ↓`;
  ui.recordingLink.title = `${message.frames} frames · saved on the server as ${message.path}`;
  ui.recordingLink.hidden = false;
}

function resetTiming(): void {
  for (const element of [ui.modelLatency, ui.nonModelLatency, ui.totalLatency]) element.textContent = "—";
  ui.latencySample.textContent = "Make a camera move to measure";
}

function activity(): void {
  const now = performance.now();
  if (state.access === "owner" && now - state.lastActivity > 5000) {
    send({ type: "activity" });
    state.lastActivity = now;
  }
}

// ------------------------------------------------------------------ frame size

// Relative speed per chunk measured on an M4 Max (MAC.md); only 832x464 is the trained size.
const SIZE_INFO: Partial<Record<string, { name: string; note: string; hint: string }>> = {
  "832x464": { name: "Full", note: "832 × 464", hint: "The size the model was trained at: best quality, slowest turns." },
  "576x320": { name: "Fast", note: "576 × 320", hint: "~4× faster turns. Softer detail; the world stays coherent." },
  "416x240": { name: "Quarter", note: "416 × 240", hint: "~8× faster turns. Out of the model's range: scenes melt within a few turns." },
};
const SIZE_KEY = "lingbot.size";

function sizeInfo(size: string): { name: string; note: string; hint: string } {
  return SIZE_INFO[size] ?? { name: size.replace("x", " × "), note: "custom size", hint: "" };
}

function storedSize(): string | null {
  try { return window.localStorage.getItem(SIZE_KEY); } catch { return null; }
}

function storeSize(size: string): void {
  try { window.localStorage.setItem(SIZE_KEY, size); } catch { /* private mode: not remembered */ }
}

function updateSizeHint(): void {
  const size = state.size;
  if (!size) { ui.sizeHint.textContent = ""; return; }
  const applies = state.active && state.sessionSize !== null && state.sessionSize !== size;
  ui.sizeHint.textContent = applies ? `${sizeInfo(size).hint} Applies when you restart the world.` : sizeInfo(size).hint;
}

function selectSize(size: string): void {
  state.size = size;
  storeSize(size);
  for (const button of ui.sizeToggle.querySelectorAll<HTMLButtonElement>("button")) {
    button.setAttribute("aria-checked", String(button.dataset.size === size));
  }
  updateSizeHint();
  activity();
}

function renderSizes(sizes: [number, number][] | undefined): void {
  state.sizes = (sizes ?? []).map(([width, height]) => `${width}x${height}`);
  ui.sizeField.hidden = state.sizes.length < 2;
  ui.sizeToggle.replaceChildren(...state.sizes.map((size) => {
    const info = sizeInfo(size);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.size = size;
    button.setAttribute("role", "radio");
    button.title = info.hint;
    const name = document.createElement("strong");
    name.textContent = info.name;
    const note = document.createElement("small");
    note.textContent = info.note;
    button.append(name, note);
    button.addEventListener("click", () => selectSize(size));
    return button;
  }));
  const stored = storedSize();
  const first = state.sizes[0];
  if (stored !== null && state.sizes.includes(stored)) selectSize(stored);
  else if (first !== undefined) selectSize(first);
  else state.size = null;
}

// ------------------------------------------------------------------ session

function start(): void {
  if (ui.startButton.disabled) return;
  const prompt = ui.promptInput.value.trim();
  const seed = Number(ui.seedInput.value);
  if (!prompt) { error("Describe your scene before starting."); ui.promptInput.focus(); return; }
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || ui.seedInput.value === "") {
    error("Choose a whole-number seed between 0 and 4294967295."); ui.seedInput.focus(); return;
  }
  const example = ui.exampleSelect.value;
  if (!state.image && !example) { error("Choose an example or upload a starting image."); return; }
  cancelPlayback();
  resetTiming();
  state.active = true;
  state.preparing = true;
  state.awaitingStart = true;
  state.waiting = true;
  state.frames = 0;
  state.chunks = 0;
  ui.frameCount.textContent = "0000";
  ui.chunkCount.textContent = "0";
  ui.actionReadout.textContent = "Preparing your world";
  ui.toast.hidden = true;
  const size = state.sizes.length > 1 && state.size !== null ? { size: state.size } : {};
  send(state.image ? { type: "start", prompt, seed, image: state.image, ...size }
    : { type: "start", prompt, seed, example, ...size });
  state.sessionSize = state.size;
  updateSizeHint();
  status("Preparing your world…");
  updateControls();
}

function stop(): void {
  if (ui.stopButton.disabled) return;
  cancelPlayback();
  state.active = false;
  state.preparing = false;
  state.stopping = true;
  send({ type: "stop" });
  status("Stopping session…");
  updateControls();
}

// ------------------------------------------------------------------ camera input

function dispatch(action: Action, inputAt: number | null): void {
  if (!canControl()) return;
  const requestId = `camera-${++state.sequence}`;
  if (inputAt !== null) state.requestTimes.set(requestId, inputAt);
  state.waiting = true;
  ui.actionReadout.textContent = labels[action];
  status(`Generating · ${labels[action].toLowerCase()}`);
  if (!send({ type: "action", action, request_id: requestId })) {
    state.requestTimes.delete(requestId);
    state.waiting = false;
  }
}

function requestAction(action: Action, inputAt: number | null): void {
  if (!canControl()) return;
  if (state.waiting || state.playing) {
    // Only the newest unsent input is kept; a fast key sequence cannot build a backlog.
    state.pending = { action, inputAt };
    return;
  }
  dispatch(action, inputAt);
}

function continueHeld(): void {
  clearTimeout(state.holdTimer);
  if (!canControl() || state.waiting || state.playing) return;
  if (state.pending) {
    const pending = state.pending;
    state.pending = null;
    dispatch(pending.action, pending.inputAt);
    return;
  }
  if (!state.held) return;
  const remaining = HOLD_MS - (performance.now() - state.held.since);
  if (remaining > 0) state.holdTimer = setTimeout(continueHeld, remaining);
  else dispatch(state.held.action, null);
}

function press(action: Action, source: string): void {
  if (!canControl() || state.held?.source === source) return;
  clearHeld();
  const inputAt = performance.now();
  state.held = { action, source, since: inputAt };
  buttons.forEach((button) => button.classList.toggle("pressed", buttonAction(button) === action));
  requestAction(action, inputAt);
  state.holdTimer = setTimeout(continueHeld, HOLD_MS);
}

function release(source: string): void {
  if (state.held?.source === source) clearHeld();
}

// ------------------------------------------------------------------ playback

function showTiming(metadata: ChunkMessage, loadedAt: number): void {
  const model = Number.isFinite(metadata.model_ms) ? metadata.model_ms : null;
  const inputAt = metadata.request_id === null ? undefined : state.requestTimes.get(metadata.request_id);
  if (metadata.request_id !== null) state.requestTimes.delete(metadata.request_id);
  ui.modelLatency.textContent = model === null ? "—" : `${Math.round(model)} ms`;
  const total = inputAt === undefined ? null : loadedAt - inputAt;
  ui.totalLatency.textContent = total === null ? "—" : `${Math.round(total)} ms`;
  // No input timestamp means no end-to-end sample. Never use model time as a fallback.
  const overhead = total === null || model === null ? null : total - model;
  ui.nonModelLatency.textContent = overhead === null ? "—" : `${Math.round(Math.max(0, overhead))} ms`;
  ui.latencySample.textContent = total === null ? "Model time · no new camera input" : "Latest camera input · first frame loaded";
}

function loadFrame(blob: Blob, epoch: number): Promise<number | null> {
  return new Promise((resolve, reject) => {
    if (epoch !== state.epoch) { resolve(null); return; }
    const frame = ui.worldFrame;
    const url = URL.createObjectURL(blob);
    const previousURL = state.frameURL;
    state.frameURL = url;
    let finished = false;
    function clean(): void {
      finished = true;
      frame.onload = null;
      frame.onerror = null;
      state.cancelLoad = null;
      if (previousURL) URL.revokeObjectURL(previousURL);
    }
    state.cancelLoad = () => {
      if (finished) return;
      clean();
      resolve(null);
    };
    frame.onload = () => {
      const loadedAt = performance.now();
      clean();
      if (epoch !== state.epoch) { resolve(null); return; }
      frame.hidden = false;
      ui.stagePlaceholder.hidden = true;
      state.frames += 1;
      ui.frameCount.textContent = String(state.frames).padStart(4, "0");
      resolve(loadedAt);
    };
    frame.onerror = () => { clean(); reject(new Error("The browser could not decode a generated frame.")); };
    frame.src = url;
  });
}

async function playChunk(metadata: ChunkMessage, buffer: ArrayBuffer): Promise<void> {
  if (!state.active || state.stopping || state.awaitingStart) return;
  const sizes = metadata.frame_sizes;
  if (!Array.isArray(sizes) || !sizes.length || sizes.some((n) => !Number.isInteger(n) || n <= 0)
      || sizes.reduce((a, b) => a + b, 0) !== buffer.byteLength) {
    error("The server sent an invalid frame block. Restart the session.");
    stop();
    return;
  }
  const epoch = state.epoch;
  state.preparing = false;
  state.playing = true;
  ui.previewLabel.hidden = true;
  if (metadata.request_id === null) ui.actionReadout.textContent = "Hold pose";
  updateControls();
  if (metadata.width && metadata.height) ui.resolutionLabel.textContent = `${metadata.width} × ${metadata.height}`;
  let offset = 0;
  let firstLoadedAt = 0;
  try {
    for (const [index, size] of sizes.entries()) {
      if (epoch !== state.epoch) return;
      if (index) {
        const delay = firstLoadedAt + index * FRAME_MS - performance.now();
        if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
      }
      if (epoch !== state.epoch) return;
      const loadedAt = await loadFrame(new Blob([buffer.slice(offset, offset + size)], { type: "image/jpeg" }), epoch);
      offset += size;
      if (loadedAt === null || epoch !== state.epoch) return;
      if (index === 0) { firstLoadedAt = loadedAt; showTiming(metadata, loadedAt); }
    }
    state.chunks += 1;
    ui.chunkCount.textContent = String(state.chunks);
    state.playing = false;
    state.waiting = false;
    // The next held action is sent only after this block's final frame has loaded.
    send({ type: "displayed", chunk_index: metadata.chunk_index });
    if (metadata.terminal) {
      state.active = false;
      state.pending = null;
      clearHeld();
      status("World complete · start another scene");
      updateControls();
    } else {
      status("Frame held · move to continue");
      continueHeld();
    }
  } catch (failure) {
    if (epoch !== state.epoch) return;
    error(failure instanceof Error ? failure.message : String(failure));
    stop();
  }
}

// ------------------------------------------------------------------ server messages

function queueState(message: QueueStateMessage): void {
  const previous = state.access;
  state.access = message.status;
  if (previous === "owner" && state.access !== "owner") {
    cancelPlayback();
    state.active = false;
    state.preparing = false;
    state.stopping = false;
    status("Session released · start when the server is available");
  }
  const queued = message.status === "queued";
  const busy = message.status === "busy";
  const waiting = Number(message.waiting_count) || 0;
  ui.waitingRoom.hidden = !(queued || busy || (message.status === "owner" && waiting > 0));
  ui.joinQueueButton.hidden = !busy;
  ui.leaveQueueButton.hidden = !queued;
  ui.waitingRoomTitle.textContent = queued ? `Waiting room · position ${message.position}` : busy ? "Server in use" : `${waiting} ${waiting === 1 ? "person" : "people"} waiting`;
  ui.waitingRoomMessage.textContent = queued ? "Your controls will unlock when it is your turn." : busy ? "One person can explore at a time. Join the queue to take a turn." : "Keep exploring, or stop your session to let the next person play.";
  const remaining = message.idle_remaining_seconds;
  ui.waitingRoomTimer.textContent = message.status === "owner" && typeof remaining === "number" && Number.isFinite(remaining) ? `Idle timeout: ${Math.ceil(remaining)}s` : "";
  if (previous === "queued" && message.status === "owner") status("Your turn · start a world to explore");
  updateControls();
}

function handleStatus(message: Pick<StatusMessage, "state" | "message">): void {
  // Frames already in transit from the old world must not undo a restart.
  if (state.awaitingStart && message.state !== "preparing" && message.state !== "stopped") return;
  if (message.message) status(message.message);
  if (message.state === "preparing") { state.preparing = true; state.awaitingStart = false; }
  if (message.state === "ready") { state.ready = true; state.preparing = false; }
  if (message.state === "stopping") { state.stopping = true; clearHeld(); }
  if (message.state === "stopped") {
    cancelPlayback();
    state.active = false;
    state.preparing = false;
    state.stopping = false;
    state.awaitingStart = false;
  }
  // A terminal block is acknowledged after playback; a completion status may
  // arrive before its binary frames, so it must not cancel that playback.
  if (message.state === "complete" && !state.waiting && !state.playing) {
    state.active = false;
    clearHeld();
  }
  updateControls();
}

function handleMessage(message: ServerMessage): void {
  switch (message.type) {
    case "hello":
      state.ready = Boolean(message.ready);
      ui.gpuLabel.textContent = `${message.gpus} GPU${message.gpus === 1 ? "" : "s"}`;
      renderSizes(message.sizes);
      status(state.ready ? "Ready · set the scene and start your world" : "Ready · first start warms up the model");
      updateControls();
      break;
    case "queue_state": queueState(message); break;
    case "session":
      if (message.status === "stopped") {
        handleStatus({ state: "stopped", message: message.reason === "idle_timeout" ? "Session released after inactivity" : "Session stopped · start a new world" });
      }
      break;
    case "status": handleStatus(message); break;
    case "chunk": state.metadata.push(message); break;
    case "preview": showPreview(message); break;
    case "recording": showRecording(message); break;
    case "error":
      error(message.message || "The server could not complete the request.");
      cancelPlayback();
      state.preparing = false;
      state.stopping = false;
      state.active = false;
      state.awaitingStart = false;
      status("Session interrupted · restart to try again");
      updateControls();
      break;
    default: break;
  }
}

function connect(): void {
  const url = new URL("ws", window.location.href);
  url.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(url);
  socket.binaryType = "arraybuffer";
  state.socket = socket;
  socket.onopen = () => {
    state.connected = true;
    ui.connectionLabel.textContent = "Connected";
    ui.connectionPill.classList.add("online");
    updateControls();
  };
  socket.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
    if (typeof event.data !== "string") {
      const metadata = state.metadata.shift();
      if (metadata) void playChunk(metadata, event.data);
      else error("Received frames without timing metadata. Restart the session.");
      return;
    }
    let message: ServerMessage;
    try { message = JSON.parse(event.data) as ServerMessage; }
    catch { error("The server sent an unreadable message."); return; }
    handleMessage(message);
  };
  socket.onclose = () => {
    cancelPlayback();
    state.connected = false;
    state.ready = false;
    state.active = false;
    state.preparing = false;
    state.stopping = false;
    state.awaitingStart = false;
    state.metadata = [];
    state.access = "available";
    ui.waitingRoom.hidden = true;
    ui.connectionLabel.textContent = "Reconnecting";
    ui.connectionPill.classList.remove("online");
    status("Connection lost · reconnecting to your server…");
    updateControls();
    setTimeout(connect, 2000);
  };
  socket.onerror = () => { /* onclose handles reconnecting without duplicate alerts. */ };
}

// ------------------------------------------------------------------ scene setup

function updatePrompt(): void {
  ui.promptCount.textContent = `${ui.promptInput.value.length} / 1600`;
  activity();
}

function selectExample(): void {
  const example = state.examples.find((item) => String(item.id) === ui.exampleSelect.value);
  if (!example) return;
  state.image = null;
  ui.imageInput.value = "";
  ui.imagePreview.src = example.image_url;
  ui.imagePreview.hidden = false;
  ui.imagePlaceholder.hidden = true;
  ui.imageName.textContent = example.label;
  ui.imageName.hidden = false;
  ui.promptInput.value = example.prompt || "";
  updatePrompt();
}

async function loadExamples(): Promise<void> {
  try {
    const response = await fetch(new URL("api/examples", window.location.href));
    if (!response.ok) throw new Error("Examples unavailable");
    const examples: unknown = await response.json();
    if (!Array.isArray(examples)) throw new Error("Invalid examples response");
    state.examples = examples as Example[];
    ui.exampleSelect.replaceChildren();
    const custom = new Option(examples.length ? "Choose an example…" : "Upload an image to begin", "");
    ui.exampleSelect.add(custom);
    state.examples.forEach((example) => ui.exampleSelect.add(new Option(example.label, String(example.id))));
    const first = state.examples[0];
    if (first) { ui.exampleSelect.value = String(first.id); selectExample(); }
  } catch {
    ui.exampleSelect.replaceChildren(new Option("Examples unavailable · upload an image", ""));
  }
}

// ------------------------------------------------------------------ wiring

ui.startButton.addEventListener("click", start);
ui.stopButton.addEventListener("click", stop);
ui.joinQueueButton.addEventListener("click", () => send({ type: "join_queue" }));
ui.leaveQueueButton.addEventListener("click", () => send({ type: "leave_queue" }));
ui.promptInput.addEventListener("input", updatePrompt);
ui.seedInput.addEventListener("input", activity);
ui.exampleSelect.addEventListener("change", selectExample);
ui.randomSeedButton.addEventListener("click", () => {
  ui.seedInput.value = String(crypto.getRandomValues(new Uint32Array(1))[0]);
  activity();
});
ui.uploadButton.addEventListener("click", () => ui.imageInput.click());
ui.imageInput.addEventListener("change", () => {
  const file = ui.imageInput.files?.[0];
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) { error("Choose a JPEG, PNG, or WebP image."); return; }
  if (file.size > 8 * 1024 * 1024) { error("Choose an image smaller than 8 MiB."); return; }
  const reader = new FileReader();
  reader.onload = () => {
    if (typeof reader.result !== "string") return;
    state.image = reader.result;
    ui.imagePreview.src = reader.result;
    ui.imagePreview.hidden = false;
    ui.imagePlaceholder.hidden = true;
    ui.imageName.textContent = file.name;
    ui.imageName.hidden = false;
    ui.exampleSelect.value = "";
    activity();
  };
  reader.onerror = () => error("Your browser could not read that image. Try another file.");
  reader.readAsDataURL(file);
});
ui.imagePreview.addEventListener("error", () => {
  ui.imagePreview.hidden = true;
  ui.imagePlaceholder.hidden = false;
  ui.imagePlaceholder.textContent = "Image preview unavailable";
});

buttons.forEach((button) => {
  button.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || button.disabled) return;
    event.preventDefault();
    button.setPointerCapture(event.pointerId);
    press(buttonAction(button), `pointer-${event.pointerId}`);
  });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
    button.addEventListener(name, (event) => release(`pointer-${event.pointerId}`));
  }
  // Enter/Space and assistive-technology activation have no pointer event.
  button.addEventListener("click", (event) => {
    if (event.detail === 0) requestAction(buttonAction(button), performance.now());
  });
});

window.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); start(); return; }
  if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return;
  if (event.target instanceof Element && event.target.closest("input, textarea, select, [contenteditable='true']")) return;
  const action = keys[event.code];
  if (!action || !canControl()) return;
  event.preventDefault();
  press(action, event.code);
});
window.addEventListener("keyup", (event) => release(event.code));
window.addEventListener("blur", () => { clearHeld(); state.pending = null; });
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { clearHeld(); state.pending = null; }
});

void loadExamples();
connect();

// A module keeps these names (status, stop, ...) out of window scope; index.html loads it with type="module".
export {};
