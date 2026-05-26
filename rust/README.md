# Rust runtime prototype

This directory contains the experimental Rust runtime for the ONNX export
pipeline.

Current goal:

1. load the exported `manifest.json`
2. load the three ONNX Runtime sessions
3. expose a minimal CLI smoke check
4. later add feature extraction, greedy RNN-T decoding, audio input, then Tauri

The first usable command will be:

```bash
cd rust/morseformer-rt
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b info
```

Decode precomputed features:

```bash
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b \
  decode-features ../../build/rust_features/cq.npy
```

Decode an 8 kHz WAV directly with the Rust frontend:

```bash
cargo run -- --onnx-dir ../../build/onnx/rnnt_phase11b \
  decode-wav ../../build/rust_features/cq.wav
```

`decode-wav` resamples input audio to 8 kHz by default and uses the
same 6 s / 2 s sliding-window shape as the Python streaming decoder for
long files. Pass `--no-windowing` for a single full-file decode.

Rust is not required for the Python package.
