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

- [x] **Phase 0** — evaluation harness, rule-based baseline decoder
- [x] **Phase 1** — parametric operator model + HF-channel simulator (AWGN, QRN, QSB, carrier drift, RX filter); SNR-ladder benchmark
- [x] **Phase 2.0** — acoustic model (Conformer + RoPE, 3.9 M params, CTC head) trained on clean audio; 1.17 % CER on a balanced 200-sample clean validation set (0 % on 4 of 5 WPM bins at 40 samples each)
- [ ] Phase 2.1 — moderate-noise training (AWGN, SNR ∈ [0, 30] dB, operator jitter); target: beat the rule-based baseline across the SNR ladder
- [ ] Phase 3 — weak-signal training + RNN-T head + FiLM conditioning
- [ ] Phase 4 — language-model pretraining + shallow fusion
- [ ] Phase 5 — real-time WSL CLI on live IC-7300 audio
- [ ] Phase 6 — GitHub / HuggingFace release, benchmark, paper

## Baseline (rule-based DSP)

SNR-ladder benchmark, 40 samples per bin, WPM uniform in [18, 28], AWGN only.
This is the number the neural pipeline must beat.

```
  SNR (dB) |  n  |    CER   |    WER   | Callsign F1
  ---------+-----+----------+----------+-------------
      +inf |  40 |   0.0000 |   0.0000 |      1.0000
     +20.0 |  40 |   0.0000 |   0.0000 |      1.0000
     +15.0 |  40 |   0.0000 |   0.0000 |      1.0000
     +10.0 |  40 |   0.0000 |   0.0000 |      1.0000
      +5.0 |  40 |   0.0058 |   0.0204 |      1.0000
       0.0 |  40 |   2.5355 |   3.0799 |      0.5750
      −5.0 |  40 |   5.8327 |   9.6283 |      0.6500
     −10.0 |  40 |  19.4111 |  23.7179 |      0.5750
     −15.0 |  40 |  25.1948 |  29.2015 |      0.6250
```

CER values above 1.0 indicate that the decoder hallucinates characters
from noise impulses — a well-known failure mode of threshold-based CW
decoders below 0 dB SNR. The Phase-2 transformer should collapse this
cliff via probabilistic element scoring and shallow-fusion language
priors.

## Phase 2.0 result (clean-audio acoustic model)

Conformer encoder (d_model = 144, 8 layers, 4 heads, 4× time sub-
sampling, RoPE attention, LayerNorm-based conv module), ~3.9 M
trainable parameters. Trained for 14 k steps on the synthetic stream
(WPM ∈ U(16, 28), no channel, no operator jitter, fixed 6 s utterances
at 8 kHz). AdamW, 2 k warmup + cosine decay, EMA 0.9999, bf16 autocast,
CTC loss with fp32 log-softmax head. Single RTX 3060 (6 GB), ≈ 3 hours
wall time.

Clean validation (200 samples, 5 WPM bins × 40 samples, same text mix
as training):

```
  WPM  |  n  |   CER   |   WER
  -----+-----+---------+--------
   16  |  40 |  0.0000 |  0.0000
   20  |  40 |  0.0000 |  0.0000
   22  |  40 |  0.0000 |  0.0000
   25  |  40 |  0.0000 |  0.0000
   28  |  40 |  0.0583 |  0.0750
  -----+-----+---------+--------
  all  | 200 |  0.0117 |  0.0150
```

The 28 WPM residual is concentrated on 1–3 character Q-codes where the
6 s window is mostly silence and the model occasionally fires spurious
characters on the silent lead-in. Noisy-condition training (Phase 2.1)
will expose the same edge case to ambient-noise data and should
regularise it away.

## Quick start

```bash
git clone git@github.com:sderhy/morseformer.git
cd morseformer
pip install -e ".[dev,audio]"
pytest -q                                            # 57 unit tests, <2 s
python -m eval.cli --decoder rule_based --dataset sanity
python -m eval.cli --decoder rule_based --dataset snr_ladder \
    --snrs="+20,+10,+5,0,-5,-10,-15" --n-per-snr 20
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
