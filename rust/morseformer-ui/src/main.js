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
const statusEl = document.querySelector("#status");
const transcript = document.querySelector("#transcript");
const rawOutput = document.querySelector("#rawOutput");
const fileTabBtn = document.querySelector("#fileTabBtn");
const liveTabBtn = document.querySelector("#liveTabBtn");
const filePanel = document.querySelector("#filePanel");
const livePanel = document.querySelector("#livePanel");
const inputDevice = document.querySelector("#inputDevice");
const refreshDevicesBtn = document.querySelector("#refreshDevicesBtn");
const startLiveBtn = document.querySelector("#startLiveBtn");
const stopLiveBtn = document.querySelector("#stopLiveBtn");
const clearLiveBtn = document.querySelector("#clearLiveBtn");

const DEFAULT_ONNX_DIR = "build/onnx/rnnt_phase11b";

function tauriApi() {
  return window.__TAURI__ || {};
}

function setStatus(value) {
  statusEl.textContent = value;
}

function setBusy(isBusy) {
  decodeBtn.disabled = isBusy;
  browseBtn.disabled = isBusy;
  clearBtn.disabled = isBusy;
}

function setTab(mode) {
  const live = mode === "live";
  fileTabBtn.classList.toggle("active", !live);
  liveTabBtn.classList.toggle("active", live);
  filePanel.classList.toggle("active", !live);
  livePanel.classList.toggle("active", live);
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
    transcript.textContent = String(error);
    rawOutput.textContent = String(error);
    setStatus("file picker error");
  }
});

resetModelBtn.addEventListener("click", () => {
  onnxDir.value = DEFAULT_ONNX_DIR;
  setStatus("model path reset");
});

clearBtn.addEventListener("click", () => {
  transcript.textContent = "";
  rawOutput.textContent = "";
  setStatus("idle");
});

fileTabBtn.addEventListener("click", () => setTab("file"));
liveTabBtn.addEventListener("click", () => setTab("live"));

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
    transcript.textContent = String(error);
    rawOutput.textContent = String(error);
    setStatus("device error");
  } finally {
    refreshDevicesBtn.disabled = false;
  }
}

refreshDevicesBtn.addEventListener("click", refreshInputDevices);
clearLiveBtn.addEventListener("click", () => {
  transcript.textContent = "";
  rawOutput.textContent = "";
  setStatus("live idle");
});

startLiveBtn.addEventListener("click", () => {
  transcript.textContent = "Live decoding is not wired yet. Device selection is ready.";
  rawOutput.textContent = `selected input: ${inputDevice.selectedOptions[0]?.textContent || ""}`;
  setStatus("live prototype");
});

stopLiveBtn.addEventListener("click", () => {
  setStatus("live stopped");
});

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
  transcript.textContent = "";
  rawOutput.textContent = "";
  setStatus("decoding");
  try {
    const invoke = tauriApi().core?.invoke;
    if (!invoke) {
      throw new Error("Tauri invoke API unavailable; launch the app with `npm run dev`");
    }
    const result = await invoke("decode_wav", payload);
    transcript.textContent = result.text || "";
    rawOutput.textContent = result.rawOutput || "";
    setStatus(`${result.frames} frames · ${result.chunks} chunks`);
  } catch (error) {
    transcript.textContent = String(error);
    rawOutput.textContent = String(error);
    setStatus("error");
  } finally {
    setBusy(false);
  }
});

refreshInputDevices();
