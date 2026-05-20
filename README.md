# morseformer

> Open-source transformer-based Morse / CW decoder. Fully local. Apache 2.0.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](#)
[![Release: v0.6.3](https://img.shields.io/badge/release-v0.6.3-brightgreen.svg)](CHANGELOG.md#release-v063)
[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Hub-sderhy/morseformer-yellow)](https://huggingface.co/sderhy/morseformer)

Conformer + RNN-T Morse decoder with a real-time streaming CLI, trained on a reproducible synthetic-HF pipeline. The current release is **v0.6.3** — a packaging refresh on top of the v0.6.2 acoustic revert from `rnnt_phase5_8.pt` back to `rnnt_phase5_5.pt` (−24 % relative mean CER on the LCWO bench). See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## Why

Existing open-source CW decoders (`fldigi`, `cwdecoder`, `MRP40`) rely on hand-tuned DSP and threshold-based segmentation; they struggle in weak-signal conditions, QRM, QSB, and with non-ideal operator timing. The commercial reference, `CW Skimmer` (VE3NEA), is closed-source and built on ~2009-era Kalman filtering. **As of April 2026, there is no published transformer-based CW decoder, and no open-source CW decoder with an integrated language model.** `morseformer` fills that gap.

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10-3.13).
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
python -m pip install --upgrade pip

# 2a. Decode an audio file (offline). First run fetches the model from HuggingFace.
pip install morseformer
morseformer decode my_recording.wav

# 2b. Or decode a live receiver in real time (PulseAudio input).
pip install "morseformer[live]"
morseformer live
```

That's it. **No GPU needed** — the decoder runs on CPU by default.

> **No Nvidia card? Save ~4 GB.** PyTorch ships a 2-3 GB GPU build by default. On a CPU-only machine, install the lightweight CPU build *before* `morseformer`:
> ```bash
> pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
> pip install morseformer            # or "morseformer[live]"
> ```

### More options

```bash
# LM shallow fusion (λ=0.7) for prose / ragchew audio.
morseformer decode my_recording.wav --preset prose

# Live presets — looser for fast exchanges, tighter for very noisy bands.
morseformer live --preset contest
morseformer live --preset conservative

# Inspect / download checkpoints.
morseformer models list
morseformer models list --advanced
morseformer models download rnnt_phase5_7
```

The four shipped presets — `live` (default), `prose`, `contest`, `conservative` — bundle the model + thresholds + optional LM behind one flag. `morseformer --help` lists every subcommand.

## Development setup

The quick start above is for users. For development, use Python 3.12 locally;
CI covers Python 3.10-3.13. Some systems expose `python3` as Python 3.14 or
newer; that can create a dev virtualenv outside the supported PyTorch /
torchaudio range. The recommended local setup is:

```bash
git clone git@github.com:sderhy/morseformer.git
cd morseformer

uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,live,gui,demo]"

ruff check .
pytest -q

# CLI works against local release/ and checkpoints/ trees, no Hub fetch.
morseformer decode my_recording.wav

# Or call the underlying scripts directly for fine-grained control.
python -m scripts.decode_audio my_recording.wav \
    --ckpt    release/rnnt_phase5_7.pt \
    --lm-ckpt release/lm_phase5_2.pt \
    --fusion-weight 0.7 \
    --confidence-threshold 0.6 \
    --digit-threshold 0.90
```

Example output on a clean synthetic `CQ DE F4HYY K` @ 20 WPM / +20 dB SNR:

```
CTC  : 'CQ DE F4HYY K'
RNN-T: 'CQ DE F4HYY K'
```

## Changelog

Per-release notes (model deltas, benches, limitations, recommended decode
recipes) live in [CHANGELOG.md](CHANGELOG.md). The latest entry is
[v0.6.3](CHANGELOG.md#release-v063); older entries go all the way back to
v0.1.0. The changelog is the project's history file; there is no separate
`HISTORY.md`.

## Benchmarks

The current release is gated by `eval/bench_lcwo.py`, a small reproducible
bench that mixes LCWO prose/oratory, one real websdr FAV22-style clip, and a
synthetic contest guard. The v0.6.2 decision reverted the recommended
acoustic from `rnnt_phase5_8.pt` back to `rnnt_phase5_5.pt`:

| Acoustic | mean CER |
|---|---:|
| **`rnnt_phase5_5`** | **2.85 %** |
| `rnnt_phase5_7` | 3.81 % |
| `rnnt_phase5_8` | 3.93 % |
| `rnnt_phase5_9` | 3.93 % |

`rnnt_phase5_5` wins 5 / 6 clips and beats the v0.6.0/v0.6.1 acoustic by
24 % relative mean CER. See [MODEL_CARD.md](MODEL_CARD.md) for model-card
metrics and [CHANGELOG.md](CHANGELOG.md) for the full release-by-release
bench history.

### Shipping a release

Every shippable acoustic must pass `eval/release_gate.py`, which runs the
LCWO clips, a synthetic silence false-positive guard, an inflated word-gap
guard, and a streaming-latency check against versioned thresholds calibrated
on `rnnt_phase5_5` (the v0.6.2 baseline, see
`eval/release_gate_v1.json`):

```bash
python -m eval.release_gate                          # gate the baseline
python -m eval.release_gate --acoustic rnnt_phase5_X # gate a candidate
python -m eval.release_gate --acoustic rnnt_phase5_X --lm lm_phase5_2
```

A JSON report lands in `reports/release_gate_<acoustic>_<date>.json` and the
process exits 0 if every category is within its non-regression margin
(default +0.5 pp absolute), 1 otherwise. The gate is the single
ship-decision criterion: any new candidate must clear it before its
checkpoint is promoted in the registry. See
[reports/technical_debt_2026-05-18.html](reports/technical_debt_2026-05-18.html)
for the P0 that motivated this gate.

## Architecture

A compact 5-stage pipeline, fully local, CPU-real-time at inference:

```
audio ──▶ [1] DSP front-end (complex BPF at carrier)
      ──▶ [2] Conformer encoder (d=144, L=8, RoPE, 4× subsample)
      ──▶ [3] Dual heads: CTC (framewise) + RNN-T (prediction + joint)
      ──▶ [4] Optional offline LM shallow fusion
      ──▶ text
```

- **Encoder**: 8-layer Conformer with RoPE attention, depth-wise conv module with LayerNorm, 4× time sub-sampling. ~3.9 M params. Shared between the CTC and RNN-T heads.
- **CTC head**: single linear on encoder output → per-frame vocab logits.
- **RNN-T head**: 128-dim LSTM prediction network + 256-dim joint network, blank at index 0. ~0.2 M params.
- **LM**: decoder-only GPT (RMSNorm + SwiGLU + RoPE + tied embeddings + causal SDPA), d=256, L=6. ~4.8 M params. Used by the offline `prose` preset; live decoding is acoustic-only.

Vocabulary: 49 tokens (blank + 26 letters + 10 digits + 9 punctuation / Morse prosigns + `É`, `À`, apostrophe).

## Training data

The release models are trained primarily from the synthetic HF pipeline, with
a small real-audio fine-tune mixed in for the v0.5+ line:

- **Text**: callsigns, Q-codes, QSO templates, numerics, English words,
  random characters, multilingual prose, and French prose with `É`, `À`, and
  apostrophe preserved.
- **Waveform renderer**: parametric operator model, Morse keying, and HF
  channel simulation in `morse_synth/`. Usage is documented in
  [morse_synth/README.md](morse_synth/README.md).
- **Channel**: AWGN, QSB, QRN, carrier jitter, carrier drift, receiver
  bandpass, and QRM.
- **Operator timing**: widened element/gap jitter, dash:dot ratio variation,
  gap inflation, and long inter-word silence inflation.
- **Real audio**: a small set of hand-keyed, aligned 6 s chunks mixed into
  Phase 5.4; broader real-audio coverage remains the main data gap.

## Project history

The phase-by-phase training history is intentionally kept in
[CHANGELOG.md](CHANGELOG.md), which is the single source for release notes,
model promotions/demotions, benchmark tables, and known regressions. The
current model-card summary lives in [MODEL_CARD.md](MODEL_CARD.md), and the
current debt snapshot lives in
[reports/technical_debt_2026-05-18.html](reports/technical_debt_2026-05-18.html).

## License

Apache 2.0 — see [LICENSE](LICENSE). The released model weights are distributed under the same license.

## Acknowledgements

- **Sébastien Derhy** — design, engineering, and on-air validation of morseformer
- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model
- **Project Gutenberg** — public-domain literary texts in English, French, German, and Spanish used to build the Phase 3.3 multilingual prose corpus
- The amateur-radio community — decades of publicly available CW recordings and transcripts

---

*73 de morseformer*
