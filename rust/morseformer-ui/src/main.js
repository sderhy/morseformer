const wavPath = document.querySelector("#wavPath");
const onnxDir = document.querySelector("#onnxDir");
const freq = document.querySelector("#freq");
const bandwidth = document.querySelector("#bandwidth");
const sampleRate = document.querySelector("#sampleRate");
const windowSeconds = document.querySelector("#windowSeconds");
const hopSeconds = document.querySelector("#hopSeconds");
const noWindowing = document.querySelector("#noWindowing");
const browseBtn = document.querySelector("#browseBtn");
const resetModelBtn = document.querySelector("#resetModelBtn");
const decodeBtn = document.querySelector("#decodeBtn");
const clearBtn = document.querySelector("#clearBtn");
const clearAllBtn = document.querySelector("#clearAllBtn");
const copyBtn = document.querySelector("#copyBtn");
const statusEl = document.querySelector("#status");
const summary = document.querySelector("#summary");
const transcript = document.querySelector("#transcript");
const diagnosticsOutput = document.querySelector("#diagnosticsOutput");
const rawOutput = document.querySelector("#rawOutput");
const fileTabBtn = document.querySelector("#fileTabBtn");
const liveTabBtn = document.querySelector("#liveTabBtn");
const filePanel = document.querySelector("#filePanel");
const livePanel = document.querySelector("#livePanel");
const modeTitle = document.querySelector("#modeTitle");
const themeBtn = document.querySelector("#themeBtn");
const inputDevice = document.querySelector("#inputDevice");
const refreshDevicesBtn = document.querySelector("#refreshDevicesBtn");
const startLiveBtn = document.querySelector("#startLiveBtn");
const stopLiveBtn = document.querySelector("#stopLiveBtn");
const clearLiveBtn = document.querySelector("#clearLiveBtn");
const liveFreq = document.querySelector("#liveFreq");
const liveBandwidth = document.querySelector("#liveBandwidth");
const liveSampleRate = document.querySelector("#liveSampleRate");
const fileBandReadout = document.querySelector("#fileBandReadout");
const liveBandReadout = document.querySelector("#liveBandReadout");
const autoCenterToggle = document.querySelector("#autoCenterToggle");
const vuStatus = document.querySelector("#vuStatus");
const vuLevelFill = document.querySelector("#vuLevelFill");
const levelDb = document.querySelector("#levelDb");
const levelPeak = document.querySelector("#levelPeak");
const inputMeta = document.querySelector("#inputMeta");
const spectrumCanvas = document.querySelector("#spectrumCanvas");
const usePeakBtn = document.querySelector("#usePeakBtn");
const prefsBtn = document.querySelector("#prefsBtn");
const prefsPanel = document.querySelector("#prefsPanel");
const caseSelect = document.querySelector("#caseSelect");
const fontSelect = document.querySelector("#fontSelect");
const fontSizeInput = document.querySelector("#fontSizeInput");
const densitySelect = document.querySelector("#densitySelect");
const breakK = document.querySelector("#breakK");
const breakKN = document.querySelector("#breakKN");
const breakSK = document.querySelector("#breakSK");
const breakPlus = document.querySelector("#breakPlus");
const breakEquals = document.querySelector("#breakEquals");
const callsignToggle = document.querySelector("#callsignToggle");
const qrzLoginBtn = document.querySelector("#qrzLoginBtn");

const DEFAULT_ONNX_DIR = "build/onnx/rnnt_phase11b";
const CALLSIGN_RE = /\b(?:[A-Z]{1,3}\d{1,2}[A-Z]{1,4}|\d[A-Z]{1,2}\d[A-Z]{1,4})\b/gi;
const PREFS_KEY = "morseformer-transcript-prefs";

let lastText = "";
let lastRawOutput = "";
let liveMonitorTimer = null;
let liveSpectrumTimer = null;
let liveDecodeTimer = null;
let liveStatusRunning = false;
let liveSpectrumRunning = false;
let liveDecodeRunning = false;
let lastPeakHz = null;

function tauriApi() {
  return window.__TAURI__ || {};
}

function setStatus(value) {
  statusEl.textContent = value;
}

function setSummary(value) {
  summary.textContent = value;
}

function setDiagnostics(lines) {
  diagnosticsOutput.textContent = Array.isArray(lines) ? lines.join("\n") : String(lines || "");
}

function renderRuntimeStatus(status) {
  setDiagnostics([
    `runtime: ${status.runtimeMode || "unknown"} ${status.runtimeExists ? "OK" : "missing"}`,
    status.runtimeBin || "",
    `onnx: ${status.onnxExists ? "OK" : "missing"}`,
    status.onnxDir || onnxDir.value.trim(),
  ]);
}

function renderDecodeDiagnostics(result, mode) {
  setDiagnostics([
    `mode: ${mode}`,
    `runtime: ${result.runtimeMode || "unknown"}`,
    result.runtimeBin || "",
    `onnx: ${result.onnxDir || onnxDir.value.trim()}`,
    `wav: ${result.wavPath || wavPath.value.trim() || "live buffer"}`,
    `freq: ${Math.round(Number(result.freq || 0))} Hz`,
    `bandwidth: ${Math.round(Number(result.bandwidth || 0))} Hz`,
    `sample rate: ${result.targetSampleRate || "?"} Hz`,
    `window/hop: ${result.windowSeconds || "?"} / ${result.hopSeconds || "?"} s`,
  ]);
}

async function refreshRuntimeStatus() {
  const invoke = tauriApi().core?.invoke;
  if (!invoke) {
    setDiagnostics("Tauri runtime unavailable in static preview.");
    return;
  }
  try {
    const status = await invoke("runtime_status", { onnxDir: onnxDir.value.trim() });
    renderRuntimeStatus(status);
  } catch (error) {
    setDiagnostics(String(error));
  }
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function currentPrefs() {
  return {
    caseMode: caseSelect.value,
    fontFamily: fontSelect.value,
    fontSize: Number(fontSizeInput.value),
    density: densitySelect.value,
    breakK: breakK.checked,
    breakKN: breakKN.checked,
    breakSK: breakSK.checked,
    breakPlus: breakPlus.checked,
    breakEquals: breakEquals.checked,
    qrzLinks: callsignToggle.checked,
  };
}

function savePrefs() {
  localStorage.setItem(PREFS_KEY, JSON.stringify(currentPrefs()));
}

function loadPrefs() {
  const defaults = {
    caseMode: "original",
    fontFamily: "mono",
    fontSize: 18,
    density: "comfortable",
    breakK: true,
    breakKN: true,
    breakSK: true,
    breakPlus: true,
    breakEquals: true,
    qrzLinks: true,
  };
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(PREFS_KEY) || "{}") };
  } catch {
    return defaults;
  }
}

function applyPrefsToControls() {
  const prefs = loadPrefs();
  caseSelect.value = prefs.caseMode;
  fontSelect.value = prefs.fontFamily;
  fontSizeInput.value = String(clampNumber(Number(prefs.fontSize), 14, 28, 18));
  densitySelect.value = prefs.density === "compact" ? "compact" : "comfortable";
  breakK.checked = Boolean(prefs.breakK);
  breakKN.checked = Boolean(prefs.breakKN);
  breakSK.checked = Boolean(prefs.breakSK);
  breakPlus.checked = Boolean(prefs.breakPlus);
  breakEquals.checked = Boolean(prefs.breakEquals);
  callsignToggle.checked = Boolean(prefs.qrzLinks);
  applyTranscriptPrefs();
}

function transcriptFontFamily(value) {
  if (value === "system") {
    return 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  }
  if (value === "serif") {
    return 'Georgia, "Times New Roman", serif';
  }
  return '"SFMono-Regular", Consolas, monospace';
}

function applyTranscriptPrefs() {
  const size = clampNumber(Number(fontSizeInput.value), 14, 28, 18);
  fontSizeInput.value = String(size);
  transcript.style.fontFamily = transcriptFontFamily(fontSelect.value);
  transcript.style.fontSize = `${size}px`;
  transcript.classList.toggle("compact-density", densitySelect.value === "compact");
}

function breakAround(value, token) {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return value.replace(new RegExp(`\\s*${escaped}\\s*`, "g"), `\n${token}\n`);
}

function breakWord(value, word) {
  return value.replace(new RegExp(`\\s+\\b${word}\\b\\s+`, "gi"), `\n${word}\n`);
}

function applyLineBreaks(value) {
  let out = value;
  if (breakEquals.checked) {
    out = breakAround(out, "=");
  }
  if (breakPlus.checked) {
    out = breakAround(out, "+");
  }
  if (breakKN.checked) {
    out = breakWord(out, "KN");
  }
  if (breakSK.checked) {
    out = breakWord(out, "SK");
  }
  if (breakK.checked) {
    out = breakWord(out, "K");
  }
  return out.replace(/\n{3,}/g, "\n\n").trim();
}

function linkCallsigns(value) {
  return value.replace(CALLSIGN_RE, (call) => {
    const encoded = encodeURIComponent(call.toUpperCase());
    return `<a href="https://www.qrz.com/db/${encoded}" target="_blank" rel="noreferrer">${call}</a>`;
  });
}

function displayText() {
  let value = lastText || "";
  value = applyLineBreaks(value);
  if (caseSelect.value === "lower") {
    value = value.toLowerCase();
  } else if (caseSelect.value === "upper") {
    value = value.toUpperCase();
  }
  return value;
}

function renderTranscript() {
  const value = displayText();
  if (!value) {
    transcript.innerHTML = '<span class="empty-state">No transcript yet</span>';
    copyBtn.disabled = true;
    return;
  }

  let html = escapeHtml(value);
  if (callsignToggle.checked) {
    html = linkCallsigns(html);
  }
  transcript.innerHTML = html;
  copyBtn.disabled = false;
}

function setOutput(text, raw = "") {
  lastText = text || "";
  lastRawOutput = raw || "";
  rawOutput.textContent = lastRawOutput;
  renderTranscript();
}

function clearOutput(status = "idle") {
  setOutput("", "");
  setStatus(status);
  setSummary("No decode yet");
}

function setBusy(isBusy) {
  decodeBtn.disabled = isBusy;
  browseBtn.disabled = isBusy;
  clearBtn.disabled = isBusy;
  clearAllBtn.disabled = isBusy;
}

function setLiveLevel(level) {
  const rms = Number(level?.rms || 0);
  const peak = Number(level?.peak || 0);
  const db = Number(level?.db ?? -120);
  const width = Math.min(100, Math.max(0, Math.sqrt(Math.max(rms, peak)) * 100));
  vuLevelFill.style.width = `${width.toFixed(1)}%`;
  vuLevelFill.classList.toggle("active", Boolean(level?.signal));
  vuStatus.textContent = level?.samples ? (level.signal ? "signal detected" : "silence") : "no samples";
  levelDb.textContent = `${db.toFixed(1)} dBFS`;
  levelPeak.textContent = `peak ${peak.toFixed(3)}`;
  inputMeta.textContent = level?.samples
    ? `${level.sampleRate} Hz | ${level.channels} ch | ${level.samples} samples`
    : "no audio buffers";
}

function resetLiveLevel() {
  setLiveLevel({ rms: 0, peak: 0, db: -120, samples: 0, sampleRate: 0, channels: 0, signal: false });
  vuStatus.textContent = "not monitoring";
  inputMeta.textContent = "waiting";
}

function resetSpectrum() {
  const ctx = spectrumCanvas.getContext("2d");
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--surface-3").trim() || "#edf1f5";
  ctx.fillRect(0, 0, spectrumCanvas.width, spectrumCanvas.height);
  lastPeakHz = null;
  usePeakBtn.textContent = "peak -- Hz";
  usePeakBtn.disabled = true;
}

function spectrumColor(value) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  const hue = 210 - v * 170;
  const light = 18 + v * 48;
  return `hsl(${hue} 85% ${light}%)`;
}

function drawSpectrumRow(spectrum) {
  const ctx = spectrumCanvas.getContext("2d");
  const { width, height } = spectrumCanvas;
  const image = ctx.getImageData(0, 0, width, height - 1);
  ctx.putImageData(image, 0, 1);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--surface-3").trim() || "#edf1f5";
  ctx.fillRect(0, 0, width, 1);

  const bins = Array.isArray(spectrum?.bins) ? spectrum.bins : [];
  if (!bins.length) {
    return;
  }
  if (!spectrum.signal) {
    drawBandOverlay(ctx, spectrum, width);
    lastPeakHz = null;
    usePeakBtn.textContent = "below threshold";
    usePeakBtn.disabled = true;
    return;
  }
  const bandWidth = width / bins.length;
  bins.forEach((value, idx) => {
    const x = Math.floor(idx * bandWidth);
    const w = Math.max(1, Math.ceil(bandWidth));
    ctx.fillStyle = spectrumColor(value);
    ctx.fillRect(x, 0, w, 1);
  });
  drawBandOverlay(ctx, spectrum, width);
  lastPeakHz = Math.round(spectrum.peakHz);
  if (autoCenterToggle.checked) {
    applyPeakToLive(lastPeakHz, false, true);
  }
  usePeakBtn.textContent = `use ${lastPeakHz} Hz`;
  usePeakBtn.disabled = false;
}

function drawBandOverlay(ctx, spectrum, width) {
  const minHz = Number(spectrum?.minHz ?? 400);
  const maxHz = Number(spectrum?.maxHz ?? 800);
  const span = Math.max(1, maxHz - minHz);
  const center = Number(liveFreq.value);
  const bw = clampBandwidth(Number(liveBandwidth.value));
  const low = center - bw / 2;
  const high = center + bw / 2;
  const x1 = Math.max(0, Math.min(width, ((low - minHz) / span) * width));
  const x2 = Math.max(0, Math.min(width, ((high - minHz) / span) * width));
  if (x2 <= 0 || x1 >= width || x2 <= x1) {
    return;
  }
  ctx.fillStyle = "rgba(23, 105, 224, 0.22)";
  ctx.fillRect(Math.floor(x1), 0, Math.max(1, Math.ceil(x2 - x1)), 1);
}

function setTab(mode) {
  const live = mode === "live";
  fileTabBtn.classList.toggle("active", !live);
  liveTabBtn.classList.toggle("active", live);
  filePanel.classList.toggle("active", !live);
  livePanel.classList.toggle("active", live);
  modeTitle.textContent = live ? "Live input decode" : "WAV file decode";
  setStatus(live ? "live idle" : "idle");
}

function commandPayload() {
  return {
    wavPath: wavPath.value.trim(),
    onnxDir: onnxDir.value.trim(),
    freq: Number(freq.value),
    bandwidth: clampBandwidth(Number(bandwidth.value)),
    targetSampleRate: Number(sampleRate.value),
    windowSeconds: Number(windowSeconds.value),
    hopSeconds: Number(hopSeconds.value),
    noWindowing: noWindowing.checked,
  };
}

function clampNumber(value, min, max, fallback) {
  const numeric = Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, numeric));
}

function clampBandwidth(value) {
  const numeric = Number.isFinite(value) ? value : 100;
  return Math.min(400, Math.max(100, numeric));
}

function normalizeBandwidthInput(input) {
  input.value = String(clampBandwidth(Number(input.value)));
}

function updateTuneReadouts() {
  const fileFreq = clampNumber(Number(freq.value), 100, 3000, 600);
  const fileBw = clampBandwidth(Number(bandwidth.value));
  const liveCenter = clampNumber(Number(liveFreq.value), 100, 3000, 600);
  const liveBw = clampBandwidth(Number(liveBandwidth.value));
  fileBandReadout.textContent = `${Math.round(fileFreq)} Hz | BW ${Math.round(fileBw)} Hz (${Math.round(fileFreq - fileBw / 2)}-${Math.round(fileFreq + fileBw / 2)})`;
  liveBandReadout.textContent = `${Math.round(liveCenter)} Hz | BW ${Math.round(liveBw)} Hz (${Math.round(liveCenter - liveBw / 2)}-${Math.round(liveCenter + liveBw / 2)})`;
}

function stepNumberInput(targetId, step) {
  const input = document.querySelector(`#${targetId}`);
  if (!input) {
    return;
  }
  const min = Number(input.min || -Infinity);
  const max = Number(input.max || Infinity);
  const current = Number(input.value || 0);
  input.value = String(clampNumber(current + step, min, max, current));
  if (targetId === "bandwidth" || targetId === "liveBandwidth") {
    normalizeBandwidthInput(input);
  }
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function setBandwidth(targetId, value) {
  const input = document.querySelector(`#${targetId}`);
  if (!input) {
    return;
  }
  input.value = String(clampBandwidth(Number(value)));
  input.dispatchEvent(new Event("change", { bubbles: true }));
}

function applyPeakToLive(value, mirrorFile, silent = false) {
  if (!value) {
    return;
  }
  liveFreq.value = String(value);
  if (mirrorFile) {
    freq.value = String(value);
  }
  updateTuneReadouts();
  if (!silent) {
    setStatus(`center set to ${value} Hz`);
  }
}

function applyTheme(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  const label = next === "dark" ? "Switch light mode" : "Switch dark mode";
  themeBtn.title = label;
  themeBtn.setAttribute("aria-label", label);
  localStorage.setItem("morseformer-theme", next);
}

browseBtn.addEventListener("click", async () => {
  const dialog = tauriApi().dialog;
  if (!dialog?.open) {
    setStatus("file picker unavailable; paste a path");
    wavPath.focus();
    return;
  }
  try {
    const selected = await dialog.open({
      multiple: false,
      filters: [{ name: "WAV audio", extensions: ["wav"] }],
    });
    if (typeof selected === "string") {
      wavPath.value = selected;
      setStatus("file selected");
    }
  } catch (error) {
    setOutput(String(error), String(error));
    setStatus("file picker error");
  }
});

resetModelBtn.addEventListener("click", () => {
  onnxDir.value = DEFAULT_ONNX_DIR;
  setStatus("model path reset");
  refreshRuntimeStatus();
});

onnxDir.addEventListener("change", refreshRuntimeStatus);

clearBtn.addEventListener("click", () => clearOutput("idle"));
clearAllBtn.addEventListener("click", () => clearOutput("idle"));

copyBtn.addEventListener("click", async () => {
  const value = transcript.innerText.trim();
  if (!value) {
    return;
  }
  await navigator.clipboard.writeText(value);
  setStatus("copied");
});

fileTabBtn.addEventListener("click", () => setTab("file"));
liveTabBtn.addEventListener("click", () => setTab("live"));

themeBtn.addEventListener("click", () => {
  const current = document.documentElement.dataset.theme;
  applyTheme(current === "dark" ? "light" : "dark");
});

prefsBtn.addEventListener("click", () => {
  prefsPanel.hidden = !prefsPanel.hidden;
});

document.addEventListener("click", (event) => {
  if (prefsPanel.hidden) {
    return;
  }
  if (prefsPanel.contains(event.target) || prefsBtn.contains(event.target)) {
    return;
  }
  prefsPanel.hidden = true;
});

for (const control of [caseSelect, fontSelect, fontSizeInput, densitySelect, breakK, breakKN, breakSK, breakPlus, breakEquals, callsignToggle]) {
  control.addEventListener("change", () => {
    applyTranscriptPrefs();
    savePrefs();
    renderTranscript();
  });
}

for (const control of [freq, bandwidth, liveFreq, liveBandwidth]) {
  control.addEventListener("change", () => {
    if (control === bandwidth || control === liveBandwidth) {
      normalizeBandwidthInput(control);
    }
    updateTuneReadouts();
  });
}

document.querySelectorAll(".step-btn").forEach((button) => {
  button.addEventListener("click", () => {
    stepNumberInput(button.dataset.target, Number(button.dataset.step || 0));
  });
});

document.querySelectorAll(".preset-btn").forEach((button) => {
  button.addEventListener("click", () => {
    setBandwidth(button.dataset.bandwidthTarget, Number(button.dataset.bandwidth));
  });
});

qrzLoginBtn.addEventListener("click", async () => {
  const invoke = tauriApi().core?.invoke;
  const url = "https://www.qrz.com/login";
  if (!invoke) {
    window.open(url, "_blank", "noopener,noreferrer");
    return;
  }
  try {
    await invoke("open_external_url", { url });
    setStatus("opened QRZ login");
  } catch (error) {
    setOutput(String(error), lastRawOutput);
    setStatus("QRZ login error");
  }
});

usePeakBtn.addEventListener("click", () => {
  applyPeakToLive(lastPeakHz, true);
});

transcript.addEventListener("click", async (event) => {
  const link = event.target.closest("a[href]");
  if (!link) {
    return;
  }
  event.preventDefault();
  const invoke = tauriApi().core?.invoke;
  if (!invoke) {
    window.open(link.href, "_blank", "noopener,noreferrer");
    return;
  }
  try {
    await invoke("open_external_url", { url: link.href });
    setStatus("opened link");
  } catch (error) {
    setOutput(String(error), lastRawOutput);
    setStatus("link error");
  }
});

async function refreshInputDevices() {
  const invoke = tauriApi().core?.invoke;
  if (!invoke) {
    setStatus("Tauri API unavailable");
    return;
  }
  refreshDevicesBtn.disabled = true;
  inputDevice.innerHTML = "";
  setStatus("refreshing devices");
  try {
    const devices = await invoke("list_input_devices");
    if (!devices.length) {
      inputDevice.append(new Option("No input devices found", ""));
      startLiveBtn.disabled = true;
      setStatus("no input devices");
      return;
    }
    for (const device of devices) {
      const label = device.default ? `${device.name} (default)` : device.name;
      const option = new Option(label, device.id);
      option.selected = Boolean(device.default);
      inputDevice.append(option);
    }
    startLiveBtn.disabled = false;
    setStatus(`${devices.length} input device${devices.length > 1 ? "s" : ""}`);
  } catch (error) {
    inputDevice.append(new Option("Device listing failed", ""));
    startLiveBtn.disabled = true;
    setOutput(String(error), String(error));
    setStatus("device error");
  } finally {
    refreshDevicesBtn.disabled = false;
  }
}

refreshDevicesBtn.addEventListener("click", refreshInputDevices);
inputDevice.addEventListener("change", () => {
  stopLiveMonitor("live idle");
  resetLiveLevel();
});

clearLiveBtn.addEventListener("click", () => {
  clearOutput("live idle");
  resetLiveLevel();
});

async function pollLiveStatus() {
  const invoke = tauriApi().core?.invoke;
  if (!invoke || liveStatusRunning) {
    return;
  }
  liveStatusRunning = true;
  try {
    const level = await invoke("live_input_status");
    setLiveLevel(level);
    setStatus(level.signal ? "live signal" : "live silence");
  } catch (error) {
    stopLiveMonitor("input monitor error");
    setOutput(String(error), String(error));
    setSummary("Input monitor failed");
  } finally {
    liveStatusRunning = false;
  }
}

async function pollLiveSpectrum() {
  const invoke = tauriApi().core?.invoke;
  if (!invoke || liveSpectrumRunning) {
    return;
  }
  liveSpectrumRunning = true;
  try {
    const spectrum = await invoke("live_spectrum");
    drawSpectrumRow(spectrum);
  } catch {
    // The status poll already reports capture errors; keep the display stable here.
  } finally {
    liveSpectrumRunning = false;
  }
}

function appendLiveText(text) {
  const next = [lastText, text].filter(Boolean).join(" ").replace(/\s+/g, " ").trim();
  lastText = next;
  renderTranscript();
}

async function decodeLiveWindow() {
  const invoke = tauriApi().core?.invoke;
  if (!invoke || liveDecodeRunning) {
    return;
  }
  liveDecodeRunning = true;
  try {
    const result = await invoke("decode_live_window", {
      onnxDir: onnxDir.value.trim(),
      freq: Number(liveFreq.value),
      bandwidth: clampBandwidth(Number(liveBandwidth.value)),
      targetSampleRate: Number(liveSampleRate.value),
      windowSeconds: 6,
    });
    rawOutput.textContent = result.rawOutput || "";
    renderDecodeDiagnostics(result, "live");
    if (result.text) {
      appendLiveText(result.text);
      setSummary(`Live decode | ${result.frames} frames`);
    } else {
      setSummary("Live decode running, no symbols yet");
    }
  } catch (error) {
    const message = String(error);
    if (message.includes("not enough live audio")) {
      setSummary("Buffering live audio");
    } else {
      stopLiveMonitor("live decode error");
      setOutput(message, message);
      setSummary("Live decode failed");
    }
  } finally {
    liveDecodeRunning = false;
  }
}

async function startLiveMonitor() {
  const invoke = tauriApi().core?.invoke;
  const selected = inputDevice.selectedOptions[0]?.textContent || "";
  if (!invoke) {
    setStatus("Tauri API unavailable");
    return;
  }
  if (!inputDevice.value) {
    setStatus("missing input device");
    return;
  }
  if (liveMonitorTimer) {
    return;
  }
  if (!onnxDir.value.trim()) {
    setStatus("missing ONNX directory");
    onnxDir.focus();
    return;
  }
  try {
    const info = await invoke("start_live_capture", { deviceId: inputDevice.value });
    setOutput("", `selected input: ${selected}\n${JSON.stringify(info, null, 2)}`);
    setStatus("monitoring input");
    setSummary("Live capture buffering");
    startLiveBtn.disabled = true;
    stopLiveBtn.disabled = false;
    refreshDevicesBtn.disabled = true;
    inputDevice.disabled = true;
    pollLiveStatus();
    pollLiveSpectrum();
    liveMonitorTimer = window.setInterval(pollLiveStatus, 450);
    liveSpectrumTimer = window.setInterval(pollLiveSpectrum, 160);
    liveDecodeTimer = window.setInterval(decodeLiveWindow, 6200);
    window.setTimeout(decodeLiveWindow, 6500);
  } catch (error) {
    setOutput(String(error), String(error));
    setStatus("live capture error");
    setSummary("Live capture failed");
  }
}

async function stopLiveMonitor(status = "live stopped") {
  const invoke = tauriApi().core?.invoke;
  if (liveMonitorTimer) {
    window.clearInterval(liveMonitorTimer);
    liveMonitorTimer = null;
  }
  if (liveSpectrumTimer) {
    window.clearInterval(liveSpectrumTimer);
    liveSpectrumTimer = null;
  }
  if (liveDecodeTimer) {
    window.clearInterval(liveDecodeTimer);
    liveDecodeTimer = null;
  }
  if (invoke) {
    try {
      await invoke("stop_live_capture");
    } catch {
      // Best effort cleanup; the UI state below still needs to recover.
    }
  }
  liveStatusRunning = false;
  liveSpectrumRunning = false;
  liveDecodeRunning = false;
  startLiveBtn.disabled = !inputDevice.value;
  stopLiveBtn.disabled = true;
  refreshDevicesBtn.disabled = false;
  inputDevice.disabled = false;
  setStatus(status);
}

startLiveBtn.addEventListener("click", startLiveMonitor);
stopLiveBtn.addEventListener("click", () => stopLiveMonitor());

decodeBtn.addEventListener("click", async () => {
  const payload = commandPayload();
  if (!payload.wavPath) {
    setStatus("missing WAV path");
    wavPath.focus();
    return;
  }
  if (!payload.onnxDir) {
    setStatus("missing ONNX directory");
    onnxDir.focus();
    return;
  }

  setBusy(true);
  setOutput("", "");
  setStatus("decoding");
  setSummary("Runtime running");
  try {
    const invoke = tauriApi().core?.invoke;
    if (!invoke) {
      throw new Error("Tauri invoke API unavailable; launch the app with `npm run dev`");
    }
    const result = await invoke("decode_wav", payload);
    setOutput(result.text || "", result.rawOutput || "");
    renderDecodeDiagnostics(result, "file");
    setStatus(`${result.frames} frames | ${result.chunks} chunks`);
    setSummary(result.text ? "Decode complete" : "Decode complete, empty transcript");
  } catch (error) {
    setOutput(String(error), String(error));
    setStatus("error");
    setSummary("Decode failed");
  } finally {
    setBusy(false);
  }
});

const savedTheme = localStorage.getItem("morseformer-theme");
const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)").matches;
applyTheme(savedTheme || (prefersDark ? "dark" : "light"));
applyPrefsToControls();
updateTuneReadouts();
renderTranscript();
resetLiveLevel();
resetSpectrum();
refreshInputDevices();
refreshRuntimeStatus();
