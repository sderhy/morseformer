# Morseformer Rust UI Handoff

Date: 2026-05-27
Branch: `feature/onnx-rust-runtime`

## Goal

Move Morseformer toward a desktop application with a Rust runtime, Tauri UI, ONNX model execution, WAV decoding, and live audio capture.

The current direction is:

1. Python remains useful for training/export/experiments.
2. Rust owns desktop runtime, audio I/O, WAV frontend, ONNX inference, and installers.
3. Tauri provides the GUI shell for macOS/Windows.

## Current State

### Rust runtime

Runtime crate:

```text
rust/morseformer-rt
```

Implemented commands include:

```bash
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b info
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b smoke
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b decode-wav /path/to/file.wav
```

`decode-wav` currently performs:

- WAV loading through Rust.
- Mono conversion.
- simple CW quadrature frontend around a configurable tone frequency.
- resampling to target sample rate.
- feature extraction.
- ONNX RNN-T greedy decode.
- optional windowed decode for longer files.

### Tauri UI

UI app:

```text
rust/morseformer-ui
```

Run in dev:

```bash
cd rust/morseformer-ui
npm install
npm run dev
```

Build:

```bash
cd rust/morseformer-ui
npm run build
```

Build a Windows installer:

```powershell
cd rust\morseformer-ui
npm run build:installer:windows
```

The installer script:

- builds `rust\morseformer-rt` in release mode.
- copies `morseformer-rt.exe` into the Tauri resource bundle.
- copies any `onnxruntime*.dll` found beside the release build.
- copies the default ONNX export from `build\onnx\rnnt_phase11b`.
- runs `tauri build --bundles nsis`.

Use a non-default model export with:

```powershell
powershell -ExecutionPolicy Bypass -File ..\scripts\package-windows.ps1 -OnnxDir C:\path\to\rnnt_phase11b
```

The app currently has:

- File decode tab.
- WAV file picker.
- ONNX directory field.
- advanced decode settings.
- transcript with copy/clear/lowercase/line breaks/QRZ links.
- light/dark mode.
- Live tab with input device picker.
- live VU meter.
- live audio buffer using `cpal`.
- live decode attempt on rolling 6 second windows.
- compact waterfall spectrum, 400-1000 Hz on X axis, time flowing downward.

## Important Paths

Default ONNX directory expected by UI:

```text
build/onnx/rnnt_phase11b
```

Runtime binary expected by UI in dev mode:

```text
rust/morseformer-rt/target/debug/morseformer-rt
```

Windows equivalent:

```text
rust\morseformer-rt\target\debug\morseformer-rt.exe
```

Build it before using the UI:

```bash
cd rust/morseformer-rt
cargo build
```

## Windows Bring-Up

Install on Windows:

- Git
- Rust via `rustup`
- Node.js LTS
- Visual Studio Build Tools with `Desktop development with C++`
- WebView2 Runtime if missing

Verify:

```powershell
git --version
cargo --version
node --version
npm --version
```

Clone and switch branch:

```powershell
git clone <repo-url>
cd morseformer
git checkout feature/onnx-rust-runtime
```

Copy or regenerate ONNX export:

```text
build\onnx\rnnt_phase11b\manifest.json
build\onnx\rnnt_phase11b\rnnt_encoder.onnx
build\onnx\rnnt_phase11b\rnnt_predictor_step.onnx
build\onnx\rnnt_phase11b\rnnt_joint.onnx
```

Build runtime:

```powershell
cd rust\morseformer-rt
cargo build
```

Run UI:

```powershell
cd ..\morseformer-ui
npm install
npm run dev
```

## Live Audio Notes

The Live tab now has two separate pieces:

- Audio capture and visualization.
- Periodic live decode.

The VU meter and spectrum use the captured audio buffer directly. If VU/spectrum move, the app is receiving audio.

Live decode currently writes a temporary WAV from the rolling live buffer and invokes `morseformer-rt decode-wav`. This is intentionally simple and not yet the final low-latency architecture.

Known limitations:

- The first live decode can only happen after enough audio is buffered.
- ONNX runtime startup cost may be noticeable because each live window shells out to the runtime binary.
- Duplicate/unstable text is likely until streaming state and overlap reconciliation are improved.
- Tone frequency still matters. The spectrogram should be used to set `Tone Hz` close to the visible CW line.

## Next Engineering Steps

High priority:

1. Test Windows dev mode end to end.
2. Confirm ONNX Runtime native dependencies on Windows.
3. Verify the Windows NSIS installer output end to end on a machine with Rust, Node.js, Visual Studio Build Tools, and the ONNX export available.
4. Replace shelling out for live windows with an in-process Rust runtime or long-lived worker process.
5. Add a stable live decode state machine with overlap, deduplication, and partial transcript handling.

Medium priority:

1. Add a real settings file for model path, tone frequency, and audio device.
2. Add installer target verification for Windows.
3. Add UI error messages for missing model/runtime binary.
4. Add a diagnostic panel: selected input, sample rate, channel count, VU, spectrum peak, decode latency.

## 100% Rust Decision

Recommendation: do not rewrite absolutely everything immediately, but make the shipped application 100% Rust/Tauri.

Practical split:

- Python remains for training, dataset tools, model research, and ONNX export.
- Rust becomes the product runtime: GUI, audio, WAV frontend, inference, packaging.

This gives most of the benefit of "100% Rust" for users without throwing away the Python training/export stack too early.

The next major technical step toward this is to move ONNX runtime loading into the Tauri backend or a persistent Rust worker, instead of shelling out to `morseformer-rt` for every live window.

## Verification Performed

Recently verified on macOS:

```bash
cd rust/morseformer-ui/src-tauri
cargo fmt --check
cargo check
```

```bash
cd rust/morseformer-ui
npm run build
```

The macOS `.app` bundle builds successfully.
