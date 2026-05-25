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

Rust is not required for the Python package.
