const wavPath = document.querySelector("#wavPath");
const onnxDir = document.querySelector("#onnxDir");
const freq = document.querySelector("#freq");
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
const rawOutput = document.querySelector("#rawOutput");
const fileTabBtn = document.querySelector("#fileTabBtn");
const liveTabBtn = document.querySelector("#liveTabBtn");
const filePanel = document.querySelector("#filePanel");
const livePanel = document.querySelector("#livePanel");
const modeTitle = document.querySelector("#modeTitle");
const themeBtn = document.querySelector("#themeBtn");
const themeLabel = document.querySelector("#themeLabel");
const inputDevice = document.querySelector("#inputDevice");
const refreshDevicesBtn = document.querySelector("#refreshDevicesBtn");
const startLiveBtn = document.querySelector("#startLiveBtn");
const stopLiveBtn = document.querySelector("#stopLiveBtn");
const clearLiveBtn = document.querySelector("#clearLiveBtn");
const liveFreq = document.querySelector("#liveFreq");
const liveSampleRate = document.querySelector("#liveSampleRate");
const vuStatus = document.querySelector("#vuStatus");
const vuLevelFill = document.querySelector("#vuLevelFill");
const levelDb = document.querySelector("#levelDb");
const levelPeak = document.querySelector("#levelPeak");
const inputMeta = document.querySelector("#inputMeta");
const spectrumCanvas = document.querySelector("#spectrumCanvas");
const spectrumPeak = document.querySelector("#spectrumPeak");
const lowercaseToggle = document.querySelector("#lowercaseToggle");
const lineBreakToggle = document.querySelector("#lineBreakToggle");
const callsignToggle = document.querySelector("#callsignToggle");

const DEFAULT_ONNX_DIR = "build/onnx/rnnt_phase11b";
const CALLSIGN_RE = /\b(?:[A-Z]{1,3}\d{1,2}[A-Z]{1,4}|\d[A-Z]{1,2}\d[A-Z]{1,4})\b/gi;

let lastText = "";
let lastRawOutput = "";
let liveMonitorTimer = null;
let liveSpectrumTimer = null;
let liveDecodeTimer = null;
let liveStatusRunning = false;
let liveSpectrumRunning = false;
let liveDecodeRunning = false;

function tauriApi() {
  return window.__TAURI__ || {};
}

function setStatus(value) {
  statusEl.textContent = value;
}

function setSummary(value) {
  summary.textContent = value;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function applyLineBreaks(value) {
  return value
    .replace(/\s*([=+])\s*/g, "\n$1\n")
    .replace(/\s+\b(KN|SK|K)\b\s+/gi, "\n$1\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function linkCallsigns(value) {
  return value.replace(CALLSIGN_RE, (call) => {
    const encoded = encodeURIComponent(call.toUpperCase());
    return `<a href="https://www.qrz.com/db/${encoded}" target="_blank" rel="noreferrer">${call}</a>`;
  });
}

function displayText() {
  let value = lastText || "";
  if (lineBreakToggle.checked) {
    value = applyLineBreaks(value);
  }
  if (lowercaseToggle.checked) {
    value = value.toLowerCase();
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
    ? `${level.sampleRate} Hz · ${level.channels} ch · ${level.samples} samples`
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
  spectrumPeak.textContent = "peak -- Hz";
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
  const bandWidth = width / bins.length;
  bins.forEach((value, idx) => {
    const x = Math.floor(idx * bandWidth);
    const w = Math.max(1, Math.ceil(bandWidth));
    ctx.fillStyle = spectrumColor(value);
    ctx.fillRect(x, 0, w, 2);
  });
  spectrumPeak.textContent = `peak ${Math.round(spectrum.peakHz)} Hz`;
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
    targetSampleRate: Number(sampleRate.value),
    windowSeconds: Number(windowSeconds.value),
    hopSeconds: Number(hopSeconds.value),
    noWindowing: noWindowing.checked,
  };
}

function applyTheme(theme) {
  const next = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  themeLabel.textContent = next === "dark" ? "Light mode" : "Dark mode";
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
});

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

for (const control of [lowercaseToggle, lineBreakToggle, callsignToggle]) {
  control.addEventListener("change", renderTranscript);
}

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
      targetSampleRate: Number(liveSampleRate.value),
      windowSeconds: 6,
    });
    rawOutput.textContent = result.rawOutput || "";
    if (result.text) {
      appendLiveText(result.text);
      setSummary(`Live decode · ${result.frames} frames`);
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
    setStatus(`${result.frames} frames · ${result.chunks} chunks`);
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
renderTranscript();
resetLiveLevel();
resetSpectrum();
refreshInputDevices();
