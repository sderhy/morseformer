# morseformer

> Open-source transformer-based Morse / CW decoder with a built-in ham-specialised language model. Real-time. Fully local. Apache 2.0.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)](#)

**Status: pre-alpha.** Active development. No checkpoints released yet. Star / watch if you want to follow along.

## Why

Existing open-source CW decoders (`fldigi`, `cwdecoder`, `MRP40`) rely on hand-tuned DSP and seuil-based segmentation; they struggle in weak-signal conditions, QRM, QSB, and with non-ideal operator timing. The commercial reference, `CW Skimmer` (VE3NEA), is closed-source and built on ~2009-era Kalman filtering. **As of April 2026, there is no published transformer-based CW decoder, and no open-source CW decoder with an integrated language model.** `morseformer` fills that gap.

## Design

A five-stage pipeline, fully local, CPU-real-time at inference:

```
IC-7300 audio ──▶ [1] DSP front-end (AFC/PLL + complex BPF)
              ──▶ [2] WPM + fist encoder (tiny CNN, FiLM conditioning)
              ──▶ [3] Acoustic model (compact Transformer, CTC + RNN-T)
              ──▶ [4] Shallow fusion with a local nanoGPT-style language model
              ──▶ [5] Async rescorer (same LM, wider context, QSO state)
              ──▶ text
```

The language model is trained from scratch on amateur-radio text: RBN spot archives, CQWW/ARRL contest exchanges, QSO transcripts, Q-codes, and a synthetic callsign corpus weighted by real ITU prefix-activity distributions. **No external APIs in the hot path.**

## Goals

- **Real-time streaming** (< 200 ms latency) for live QSO copy
- **State-of-the-art weak-signal performance** (target: CER < 10 % at −5 dB SNR, < 25 % at −15 dB)
- **Callsign-aware decoding** via regex-constrained beam search and ITU prefix priors
- **CPU-real-time inference** — no GPU required to *use* the decoder
- **Fully reproducible training pipeline**, including an open synthetic-data generator with HF-channel simulation (AWGN, QRN, QRM, QSB, multipath, drift)

## Current state

- [ ] Phase 0 — evaluation harness, baseline reproduction
- [ ] Phase 1 — DSP front-end + `morse_synth` channel simulator
- [ ] Phase 2 — acoustic model (clean + moderate-noise training)
- [ ] Phase 3 — weak-signal training + RNN-T head + FiLM conditioning
- [ ] Phase 4 — language-model pretraining + shallow fusion
- [ ] Phase 5 — real-time WSL CLI on live IC-7300 audio
- [ ] Phase 6 — GitHub / HuggingFace release, benchmark, paper

## Quick start

```bash
# coming soon — pre-alpha, not yet installable
```

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgements

- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model
- The amateur-radio community — decades of publicly available CW recordings and transcripts

---

*73 de morseformer*
