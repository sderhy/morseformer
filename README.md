# morseformer

> Open-source transformer-based Morse / CW decoder. Fully local. Apache 2.0.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10-3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](#)
[![Release: v0.6.4](https://img.shields.io/badge/release-v0.6.4-brightgreen.svg)](CHANGELOG.md#release-v064)
[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Hub-sderhy/morseformer-yellow)](https://huggingface.co/sderhy/morseformer)

Conformer + RNN-T Morse decoder with a real-time streaming CLI, trained on a reproducible synthetic-HF pipeline plus a forced-alignment-aware real-audio fine-tune. The current release is **v0.6.4** — promotes **`rnnt_phase11b.pt`** as the recommended acoustic (`-34 %` relative mean CER and `-37 %` mean WER on a real-OTA bench vs the v0.6.3 baseline) and ships a new amateur-idiom char n-gram LM (`lm_amateur_3gram.pkl`, 482 KB) used by the dictionary splitter. See [CHANGELOG.md](CHANGELOG.md) for the full version history.

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
# LM shallow fusion (λ=0.7) + word splitter for prose / ragchew audio.
# The splitter re-segments run-on amateur idioms (DROMCHRIS → DR OM CHRIS)
# using a built-in amateur + English dictionary.
morseformer decode my_recording.wav --preset prose
# Force the splitter on or off independently of the preset:
morseformer decode my_recording.wav --post-segment
morseformer decode my_recording.wav --preset prose --no-post-segment

# Live presets — looser for fast exchanges, tighter for very noisy bands.
morseformer live --preset contest
morseformer live --preset conservative

# Inspect / download checkpoints.
morseformer models list
morseformer models list --advanced
morseformer models download rnnt_phase11b
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
    --ckpt    release/rnnt_phase11b.pt \
    --confidence-threshold 0.6 \
    --digit-threshold 0.90 \
    --post-segment \
    --post-segment-lm release/lm_amateur_3gram.pkl
```

Example output on a clean synthetic `CQ DE F4HYY K` @ 20 WPM / +20 dB SNR:

```
CTC  : 'CQ DE F4HYY K'
RNN-T: 'CQ DE F4HYY K'
```

## Changelog

Per-release notes (model deltas, benches, limitations, recommended decode
recipes) live in [CHANGELOG.md](CHANGELOG.md). The latest entry is
[v0.6.4](CHANGELOG.md#release-v064); older entries go all the way back to
v0.1.0. The changelog is the project's history file; there is no separate
`HISTORY.md`.

## Benchmarks

The v0.6.4 release passes `eval/release_gate.py` against
`eval/release_gate_v2.json` (10/10 categories: 4 LCWO prose/oratory
clips, 1 callsign clip, 1 websdr FAV22-style clip, 1 synthetic
contest guard, plus silence-FP / word-gap / latency stress tests).
Promotion was driven by a real-OTA audit of 26 hand-keyed ragchew
clips (g3ses + g6pz, 31 min total) decoded with the `prose` preset:

| Metric           | v0.6.3 (`rnnt_phase5_5`) | **v0.6.4 (`rnnt_phase11b`)** | Δ rel. |
|------------------|--------------------------|------------------------------|--------|
| ALL CER          | 26.98 %                  | **17.75 %**                  | **-34 %** |
| ALL WER          | 70.34 %                  | **44.31 %**                  | **-37 %** |
| g3ses CER        | 20.56 %                  | 8.45 %                       | -59 %  |
| g6pz CER (held-out) | 34.46 %               | 28.60 %                      | -17 %  |

g6pz is held out of the training real-audio mix. v0.6.4 closes a
silent-truncation bug in the real-audio augmentation that had blocked
five consecutive retrains (Phase 8 / 8a / 9 / 10 / 11) — see
[CHANGELOG.md](CHANGELOG.md#release-v064) for the diagnosis.

### Shipping a release

Every shippable acoustic must pass `eval/release_gate.py`, which runs the
LCWO + callsign clips, a synthetic silence false-positive guard, an
inflated word-gap guard, and a streaming-latency check against versioned
thresholds (`release_gate_v1.json` calibrated on `rnnt_phase5_5`;
`release_gate_v2.json` re-calibrated for v0.6.4):

```bash
python -m eval.release_gate --manifest eval/release_gate_v2.json
python -m eval.release_gate --acoustic rnnt_phase11b   # gate by name
python -m eval.release_gate --ckpt-path checkpoints/<X>/best_rnnt.pt
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
- **Splitter LM** (v0.6.4): char 3-gram with stupid-backoff smoothing, trained on 100k synthetic amateur samples from the Phase 9 mix. 482 KB on disk. Rescores candidate splits from `morseformer.decoding.word_splitter` in the `prose` preset.
- **Neural LM** (legacy, off by default): decoder-only GPT (RMSNorm + SwiGLU + RoPE + tied embeddings + causal SDPA), d=256, L=6. ~4.8 M params. Available via `--lm lm_phase5_2` for research; dropped from the default `prose` preset at v0.6.3 because it hurt amateur jargon on literary prose.

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
- **Real audio**: forced-alignment-aware fine-tune (Phase 11b, v0.6.4)
  on hand-keyed ragchew chunks. Per-token timestamps from
  `torchaudio.functional.forced_align` drive a word-gap augmentation
  that inserts silence in the true inter-word gap *and* trims the
  label when the inflated audio overflows the target window — closing
  a silent-truncation bug that had blocked Phases 8 / 8a / 9 / 10 /
  11. Broader real-audio coverage (multi-operator, W1AW transcripts)
  remains the main data gap.

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
