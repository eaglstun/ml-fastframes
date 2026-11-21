// End-to-end check of the LingBot demo page in headless Chrome over the DevTools protocol (no
// dependencies; Node >= 22 for the built-in WebSocket). Needs a running demo server: it loads
// the page, starts a world, presses W (keyboard) and the on-screen D button (pointer), stops,
// and fails on any page exception or console error. Generates real chunks, so it takes about a
// minute at 832x464 and uses the GPU like a normal session.
// Usage: npm run e2e -- [url (default http://127.0.0.1:8000/)] [screenshot dir (default .e2e)]
// CHROME=/path/to/chrome overrides the browser.
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url = "http://127.0.0.1:8000/", out = ".e2e"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const PORT = 9333;
const profile = mkdtempSync(join(tmpdir(), "lingbot-e2e-"));
const chrome = spawn(process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--no-first-run", "--no-default-browser-check", "--window-size=1440,1000", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const t0 = Date.now();
const log = (...a) => console.log(`[${((Date.now() - t0) / 1000).toFixed(1).padStart(6)}s]`, ...a);

let ws, nextId = 0;
const pending = new Map(), problems = [];
async function cdp(method, params = {}) {
  const id = ++nextId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}
async function evaluate(expression) {
  const r = await cdp("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (r.exceptionDetails) throw new Error(`eval failed: ${expression}: ${r.exceptionDetails.text}`);
  return r.result.value;
}
async function until(what, expression, timeoutMs, onPoll) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const value = await evaluate(expression);
    if (onPoll) await onPoll();
    if (value) { log(`ok: ${what}`); return value; }
    await sleep(150);
  }
  throw new Error(`timed out waiting for: ${what}`);
}
const ui = `({state: stateLabel.textContent, frames: frameCount.textContent, chunks: chunkCount.textContent,
  preview: previewLabel.hidden ? null : previewLabel.textContent, keyboard: keyboardStatus.textContent,
  model: modelLatency.textContent, total: totalLatency.textContent, other: nonModelLatency.textContent,
  resolution: resolutionLabel.textContent})`.replace(/(\w+)\.(textContent|hidden)/g, 'document.getElementById("$1").$2');
const previews = new Set();
const watchPreview = async () => { const p = await evaluate(ui); if (p.preview) previews.add(`chunk ${p.chunks}: ${p.preview}`); };
async function shot(name) {
  const { data } = await cdp("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, name), Buffer.from(data, "base64"));
  log(`screenshot ${join(out, name)}`);
}

try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    try { target = await (await fetch(`http://127.0.0.1:${PORT}/json/new?about:blank`, { method: "PUT" })).json(); }
    catch { await sleep(200); }
  }
  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r) => ws.addEventListener("open", r));
  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id); pending.delete(msg.id);
      msg.error ? p.reject(new Error(JSON.stringify(msg.error))) : p.resolve(msg.result);
    } else if (msg.method === "Runtime.exceptionThrown") {
      problems.push(`exception: ${msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text}`);
    } else if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(msg.params.type)) {
      problems.push(`console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
    } else if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
      problems.push(`log: ${msg.params.entry.text} ${msg.params.entry.url ?? ""}`);
    }
  });
  await cdp("Runtime.enable"); await cdp("Log.enable"); await cdp("Page.enable");
  await cdp("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp("Page.navigate", { url });

  await until("module loaded and socket connected", `document.getElementById("connectionLabel")?.textContent === "Connected"`, 15000);
  await until("examples loaded", `document.getElementById("exampleSelect").options.length > 1 && document.getElementById("promptInput").value.length > 0`, 10000);
  log("app.js is a module (window.status untouched):", await evaluate(`typeof window.stop === "function" && typeof window.status === "string"`));
  await until("start enabled", `!document.getElementById("startButton").disabled`, 10000);
  await shot("1_loaded.png");

  await evaluate(`document.getElementById("startButton").click()`);
  log("clicked Start world");
  await until("first chunk played, keyboard live", `document.getElementById("keyboardStatus").textContent === "KEYBOARD LIVE" && Number(document.getElementById("chunkCount").textContent) >= 1`, 180000, watchPreview);
  log("state:", JSON.stringify(await evaluate(ui)));
  await shot("2_first_chunk.png");

  // Real keyboard input: W through the window keydown handler.
  const key = { code: "KeyW", key: "w", windowsVirtualKeyCode: 87, nativeVirtualKeyCode: 87 };
  await cdp("Input.dispatchKeyEvent", { type: "keyDown", ...key, text: "w" });
  await sleep(120);
  await cdp("Input.dispatchKeyEvent", { type: "keyUp", ...key });
  log("pressed W");
  await until("W chunk played", `Number(document.getElementById("chunkCount").textContent) >= 2 && document.getElementById("keyboardStatus").textContent === "KEYBOARD LIVE" && !document.getElementById("stateLabel").textContent.startsWith("Generating")`, 120000, watchPreview);
  const afterW = await evaluate(ui);
  log("state:", JSON.stringify(afterW));

  // Real pointer input on the on-screen D button.
  const box = await evaluate(`(() => { const r = document.querySelector('button[data-action="right"]').getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2}; })()`);
  await cdp("Input.dispatchMouseEvent", { type: "mousePressed", x: box.x, y: box.y, button: "left", clickCount: 1, buttons: 1 });
  await sleep(120);
  await cdp("Input.dispatchMouseEvent", { type: "mouseReleased", x: box.x, y: box.y, button: "left", clickCount: 1, buttons: 0 });
  log("pressed the on-screen D button");
  let sawPreviewShot = false;
  await until("D chunk played", `Number(document.getElementById("chunkCount").textContent) >= 3 && !document.getElementById("stateLabel").textContent.startsWith("Generating")`, 120000, async () => {
    await watchPreview();
    if (!sawPreviewShot && !(await evaluate(`document.getElementById("previewLabel").hidden`))) { sawPreviewShot = true; await shot("3_preview.png"); }
  });
  log("state:", JSON.stringify(await evaluate(ui)));
  await shot("4_after_d.png");

  // Frame-size toggle (offered on a Mac): restart at each other size and play turns there.
  const sizes = await evaluate(`[...document.querySelectorAll("#sizeToggle button")].map((b) => b.dataset.size)`);
  log("sizes offered:", sizes.join(", ") || "one (toggle hidden)");
  const pressW = async () => {
    const key = { code: "KeyW", key: "w", windowsVirtualKeyCode: 87, nativeVirtualKeyCode: 87 };
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", ...key, text: "w" });
    await sleep(60);
    await cdp("Input.dispatchKeyEvent", { type: "keyUp", ...key });
  };
  const chunks = async () => Number(await evaluate(`document.getElementById("chunkCount").textContent`));
  const idle = `document.getElementById("keyboardStatus").textContent === "KEYBOARD LIVE" && !document.getElementById("stateLabel").textContent.startsWith("Generating")`;
  for (const [size, turns] of [["576x320", 1], ["416x240", 10]]) {
    if (!sizes.includes(size)) continue;
    await evaluate(`document.querySelector('#sizeToggle button[data-size="${size}"]').click()`);
    log(`selected ${size}; hint: ${await evaluate(`document.getElementById("sizeHint").textContent`)}`);
    await evaluate(`document.getElementById("startButton").click()`);
    log("clicked Restart world");
    await until(`first chunk at ${size}`, `document.getElementById("chunkCount").textContent === "1" && ${idle}`, 180000, watchPreview);
    const label = await evaluate(`document.getElementById("resolutionLabel").textContent`);
    if (label !== size.replace("x", " × ")) throw new Error(`resolution label ${label} at ${size}`);
    for (let turn = 1; turn <= turns; turn++) {
      const before = await chunks();
      await pressW();  // immediately, no think time: measures any wait for between-turn work
      await until(`${size} turn ${turn}`, `Number(document.getElementById("chunkCount").textContent) > ${before} && ${idle}`, 120000, watchPreview);
      const s = await evaluate(ui);
      log(`  ${size} turn ${turn}: input -> first frame ${s.total}, model ${s.model}, non-model ${s.other}`);
    }
    await shot(`5_${size}.png`);
  }

  await evaluate(`document.getElementById("stopButton").click()`);
  await until("session stopped", `!document.getElementById("startButton").disabled && document.getElementById("startButtonLabel").textContent === "Start world"`, 30000);
  log("final:", JSON.stringify(await evaluate(ui)));
  // The server saves the ended world's video and the page links to it.
  // Restarts save the earlier worlds too; wait for the link to the world that was just stopped.
  const lastSize = (await evaluate(`document.getElementById("resolutionLabel").textContent`)).replace(" × ", "x");
  const link = await until(`recording link for the ${lastSize} world`, `(() => { const a = document.getElementById("recordingLink");
    return a && !a.hidden && a.href.includes("-${lastSize}") ? {text: a.textContent, href: a.href, title: a.title} : null; })()`,
    120000);
  log("recording:", link.text, "·", link.title);
  const video = await evaluate(`fetch(${JSON.stringify(link.href)}).then(async (r) => ({status: r.status,
    type: r.headers.get("content-type"), bytes: (await r.arrayBuffer()).byteLength}))`);
  log("downloaded:", JSON.stringify(video));
  if (video.status !== 200 || video.type !== "video/mp4" || video.bytes < 10000) throw new Error("recording download failed");
  await shot("6_saved.png");
  log("previews seen:", [...previews].join(" | ") || "none");
  log(problems.length ? `PROBLEMS:\n  ${problems.join("\n  ")}` : "no page exceptions or console errors");
  process.exitCode = problems.length ? 1 : 0;
} catch (e) {
  log("FAILED:", e.message);
  if (ws) { try { log("state:", JSON.stringify(await evaluate(ui))); await shot("failure.png"); } catch {} }
  log(problems.length ? `PROBLEMS:\n  ${problems.join("\n  ")}` : "no page exceptions or console errors");
  process.exitCode = 1;
} finally {
  ws?.close();
  chrome.kill();
}
