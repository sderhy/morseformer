# morseformer

> Open-source transformer-based Morse / CW decoder. Fully local. Apache 2.0.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![Release: v0.1.0](https://img.shields.io/badge/release-v0.1.0-brightgreen.svg)](#release-v010)
[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Hub-sderhy/morseformer-yellow)](https://huggingface.co/sderhy/morseformer)

Conformer + RNN-T Morse decoder with an optional character-level language model, trained on a reproducible synthetic-HF pipeline. The **v0.1.0 release** ships a 4.1 M-parameter acoustic model that hits **0 % CER at SNR ≥ 0 dB** on the official in-distribution SNR-ladder benchmark, a 4.8 M-parameter LM, and LM-fusion decoders (shallow + ILME) for research.

## Why

Existing open-source CW decoders (`fldigi`, `cwdecoder`, `MRP40`) rely on hand-tuned DSP and threshold-based segmentation; they struggle in weak-signal conditions, QRM, QSB, and with non-ideal operator timing. The commercial reference, `CW Skimmer` (VE3NEA), is closed-source and built on ~2009-era Kalman filtering. **As of April 2026, there is no published transformer-based CW decoder, and no open-source CW decoder with an integrated language model.** `morseformer` fills that gap.

## Quick start

```bash
git clone git@github.com:sderhy/morseformer.git
cd morseformer
pip install -e ".[dev,audio]"
pytest -q

# download the release checkpoint (4.1 M params, ~16 MB)
huggingface-cli download sderhy/morseformer rnnt_phase3_0.pt \
    --local-dir checkpoints/phase3_0

# decode a .wav file (any length — audio is chunked into 6 s windows,
# the length the model was trained on)
python -m scripts.decode_audio my_recording.wav \
    --ckpt checkpoints/phase3_0/rnnt_phase3_0.pt
```

Example output on a clean synthetic `CQ DE F4HYY K` @ 20 WPM / +20 dB SNR:

```
CTC  : 'CQ DE F4HYY K'
RNN-T: 'CQ DE F4HYY K'
```

## Release v0.1.0

The release consists of three artifacts, all on [🤗 sderhy/morseformer](https://huggingface.co/sderhy/morseformer):

| file                        | params | purpose                                             |
|-----------------------------|--------|-----------------------------------------------------|
| `rnnt_phase3_0.pt`          | 4.13 M | **main acoustic model** — Conformer + RNN-T/CTC     |
| `lm_phase4_0.pt`            | 4.76 M | character-level language model (optional, research) |
| `MODEL_CARD.md`             | —      | architecture, training data, benchmarks, limits     |

The acoustic model is the primary release. The LM and fusion decoder are bundled for reproducibility — on the current synthetic distribution they do **not** improve CER (see [Phase 4.1 result](#phase-41--lm-fusion-does-not-help-on-synthetic-data) below), but shipping them makes the code auditable and leaves the door open for future work on real-text LM corpora.

## Benchmarks

**Phase 3.0 SNR-ladder** (1200 in-distribution validation samples: 40 per WPM × 5 WPM bins × 6 SNR; synthetic AWGN + 500 Hz RX filter; same operator-jitter distribution as training):

```
  SNR (dB) |  CER    |  WER    |  notes
  ---------+---------+---------+---------------------------------
     +20.0 |  0.0000 |  0.0000 |
     +10.0 |  0.0000 |  0.0000 |
      +5.0 |  0.0000 |  0.0000 |
       0.0 |  0.0000 |  0.0000 |
      −5.0 |  0.0280 |  0.0737 |
     −10.0 |  0.7620 |  0.9460 |  ← single bin dominates aggregate CER
  ---------+---------+---------+
  overall  |  0.1317 |  0.1718 |
```

RNN-T head vs CTC head on the same encoder: RNN-T wins by ~4 pp at −10 dB (the only bin where the two disagree). On SNR ≥ 0 dB the heads are tied at 0 %. See [`scripts/eval_fusion.py`](scripts/eval_fusion.py) to reproduce.

**Speed-up vs the rule-based baseline** (`python -m eval.cli --decoder rule_based`, same seed, same bench):

```
  SNR (dB) |  baseline CER  |  Phase 3.0 CER  |  speed-up
  ---------+----------------+-----------------+----------
       0.0 |      2.5355    |      0.0000     |    ∞
      −5.0 |      5.8327    |      0.0280     |  208 ×
     −10.0 |     19.4111    |      0.7620     |   25 ×
```

## Architecture

A compact 5-stage pipeline, fully local, CPU-real-time at inference:

```
audio ──▶ [1] DSP front-end (complex BPF at carrier)
      ──▶ [2] Conformer encoder (d=144, L=8, RoPE, 4× subsample)
      ──▶ [3] Dual heads: CTC (framewise) + RNN-T (prediction + joint)
      ──▶ [4] Optional LM fusion (shallow or ILME density-ratio)
      ──▶ text
```

- **Encoder**: 8-layer Conformer with RoPE attention, depth-wise conv module with LayerNorm, 4× time sub-sampling. ~3.9 M params. Shared between the CTC and RNN-T heads.
- **CTC head**: single linear on encoder output → per-frame vocab logits.
- **RNN-T head**: 128-dim LSTM prediction network + 256-dim joint network, blank at index 0. ~0.2 M params.
- **LM** (v0.1 release includes, but fusion gain is null — see Phase 4.1): decoder-only GPT (RMSNorm + SwiGLU + RoPE + tied embeddings + causal SDPA), d=256, L=6. ~4.8 M params. Trained to PPL 3.75 on the synthetic morseformer text distribution.

Vocabulary: 46 tokens (blank + 26 letters + 10 digits + 9 punctuation / Morse prosigns).

## Training data

All training audio in v0.1 is synthetic:

- **Text corpus**: a mix of random-callsign strings weighted by real ITU prefix-activity distributions (RBN-derived), contest QSO exchanges (CQ WW / ARRL-DX / SS formats), common Q-codes and abbreviations, RST reports, and ragchew fragments. See `morseformer/data/text.py`.
- **Waveform renderer**: parametric operator model (WPM, element / gap jitter), Morse-code keying with sine carrier, HF-channel simulator (AWGN, RX bandpass) — see `morse_synth/`.
- **Curriculum**: Phase 2.1 conditions — AWGN SNR ∈ U(0, 30) dB, 500 Hz RX filter at 600 Hz carrier, element jitter ∈ U(0, 0.05), gap jitter ∈ U(0, 0.10), WPM ∈ U(16, 28), fixed 6 s utterances at 8 kHz. Phase 3.0 trained 80 k steps on this.

No real-audio data is used in v0.1. Real-audio finetuning is on the Phase 3.1+ roadmap.

## Training history

<details>
<summary><b>Phase 0</b> — evaluation harness + rule-based DSP baseline</summary>

CER ranges from 0 at high SNR to ≈ 25 at −15 dB (threshold decoders hallucinate on noise). Used as the scoreboard the neural pipeline must beat. See `eval/` for the bench.
</details>

<details>
<summary><b>Phase 1</b> — HF-channel simulator + operator model</summary>

AWGN, QRN, QSB, carrier drift, RX filter. Feeds every subsequent phase. See `morse_synth/`.
</details>

<details>
<summary><b>Phase 2.0</b> — clean-audio acoustic baseline (CTC only)</summary>

Conformer + RoPE, 3.9 M params. 14 k steps on clean 6 s clips. **1.17 % CER** on balanced clean validation.
</details>

<details>
<summary><b>Phase 2.1</b> — noisy curriculum (CTC only)</summary>

Same arch, 50 k steps on AWGN ∈ U(0, 30) dB + RX filter + mild jitter. Beat the rule-based baseline by 33× at 0 dB, 7× at −5 dB, 16× at −10 dB. Encoder weights bootstrap Phase 3.
</details>

<details>
<summary><b>Phase 2.2</b> — wider-jitter ablation (archived)</summary>

Traded high-SNR precision for less hallucination at −10 / −15 dB. Kept as ablation; does not supersede 2.1.
</details>

### Phase 3.0 — RNN-T head + multi-task CTC / RNN-T training (v0.1 release)

Added a PredictionNetwork (single-layer LSTM, d_pred = 128) and a JointNetwork (d_joint = 256) on top of the Phase 2.1 Conformer encoder, trained multi-task with `ctc_weight = 0.3` and `rnnt_weight = 1.0` for 80 k steps. Encoder weights bootstrapped from `phase2_1/best_cer.pt`; CTC head re-used. Same curriculum as Phase 2.1, same AMP (bf16) config, EMA 0.9999. ~10 h wall time on an RTX 3060.

Result: **13.15 % overall CER**, with the RNN-T head beating the CTC head by 4 pp at the single −10 dB bin (where the RNN-T's sequence-level prior actually matters). At SNR ≥ 0 dB the two heads are tied at 0 %.

A d=144×L=16 scaling attempt (Phase 3-scale, 8 M params) tied with Phase 3.0 to within noise (13.16 % vs 13.15 %) — **capacity is not the bottleneck, the −10 dB AWGN bin is intrinsic to the synthetic distribution**.

### Phase 4.0 — character-level LM

A 4.76 M-param decoder-only transformer (RMSNorm, SwiGLU, RoPE, tied embeddings, causal SDPA), trained on the same synthetic text distribution as the acoustic model. Plateaued at val PPL 3.75 (step ~2.5 k); killed at step 8.5 k once it was clear the synthetic corpus was saturated. Released for research; not required for decoding.

### Phase 4.1 — LM fusion does not help on synthetic data

Both shallow fusion (Gulati et al. 2020) and ILME / density-ratio fusion (Meng et al. 2021, arXiv:2011.01991) were implemented on top of the RNN-T greedy decoder and swept over the SNR ladder:

```
  λ_lm  |  λ_ilm  |  overall CER  |  CER at −10 dB
  ------+---------+---------------+----------------
  0.00  |  0.00   |    13.17 %    |    76.20 %     (baseline, n = 1200)
  0.20  |  0.00   |    13.26 %    |    76.70 %     (shallow)
  0.30  |  0.00   |    13.09 %    |    75.63 %     (shallow, best λ)
  0.20  |  0.20   |    13.65 %    |    78.37 %     (ILME, n = 150)
  0.50  |  0.50   |    21.19 %    |   116.52 %     (ILME, aggressive)
```

The λ curve is flat inside the ±0.3 pp noise floor at n = 1200 — **fusion gives no measurable gain** on this distribution. ILME is actively harmful at −10 dB.

Root cause: the LM and the RNN-T prediction network see **exactly the same synthetic text corpus**. Standard ASR results where external-LM fusion wins rely on a domain shift between training text and eval text, which we do not have here. The fusion code ships as a documented capability; expect it to start helping once v0.2 adds a ham-realistic LM corpus (QSO logs, contest exchanges, RBN spots) distinct from the acoustic model's training text.

## Roadmap

- [x] **Phase 0** — evaluation harness + rule-based baseline
- [x] **Phase 1** — operator model + HF-channel simulator
- [x] **Phase 2.0** — clean-audio CTC baseline (1.17 % CER)
- [x] **Phase 2.1** — noisy CTC curriculum (beats rule-based 7-33× on SNR ladder)
- [x] **Phase 2.2** — wider-jitter ablation (archived)
- [x] **Phase 3.0** — RNN-T head, multi-task training — **v0.1 release**
- [x] **Phase 3-scale** — d×L ablation; capacity is not the bottleneck
- [x] **Phase 4.0** — character-level LM (PPL 3.75)
- [x] **Phase 4.1** — LM fusion (shallow + ILME); neutral on synthetic data
- [ ] **Phase 3.1** — realistic synthetic channel (QRM, QRN, selective fading, real RX filter) — this is where the next real CER gain is expected
- [ ] **Phase 4.2** — ham-realistic LM corpus (RBN spots, contest logs, real QSO transcripts) — prerequisite for fusion gains
- [ ] **Phase 5** — real-audio finetuning (W1AW aligned transcripts, RBN/SDR pairings)
- [ ] **Phase 6** — real-time streaming CLI on live IC-7300 audio
- [ ] **Phase 7** — callsign-aware beam search with ITU prefix priors

## License

Apache 2.0 — see [LICENSE](LICENSE). The released model weights are distributed under the same license.

## Acknowledgements

- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model
- The amateur-radio community — decades of publicly available CW recordings and transcripts

---

*73 de morseformer*
