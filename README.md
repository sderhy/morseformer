# morseformer

> Open-source transformer-based Morse / CW decoder. Fully local. Apache 2.0.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![Release: v0.6.3](https://img.shields.io/badge/release-v0.6.3-brightgreen.svg)](CHANGELOG.md#release-v063)
[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Hub-sderhy/morseformer-yellow)](https://huggingface.co/sderhy/morseformer)

Conformer + RNN-T Morse decoder with a real-time streaming CLI, trained on a reproducible synthetic-HF pipeline. The current release is **v0.6.3** — a packaging refresh on top of the v0.6.2 acoustic revert from `rnnt_phase5_8.pt` back to `rnnt_phase5_5.pt` (−24 % relative mean CER on the LCWO bench). See [CHANGELOG.md](CHANGELOG.md) for the full version history.

## Why

Existing open-source CW decoders (`fldigi`, `cwdecoder`, `MRP40`) rely on hand-tuned DSP and threshold-based segmentation; they struggle in weak-signal conditions, QRM, QSB, and with non-ideal operator timing. The commercial reference, `CW Skimmer` (VE3NEA), is closed-source and built on ~2009-era Kalman filtering. **As of April 2026, there is no published transformer-based CW decoder, and no open-source CW decoder with an integrated language model.** `morseformer` fills that gap.

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.10+).
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

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

Developing from a checkout? Same CLI, plus the existing scripts:

```bash
git clone git@github.com:sderhy/morseformer.git
cd morseformer
pip install -e ".[dev,live]"
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
v0.1.0.

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

### Phase 3.1 — realistic-channel fine-tune (intermediate, not released)

Phase 3.0 fine-tuned for 40 k steps on the full realistic synthetic channel: carrier-frequency jitter (±50 Hz), QSB (slow fading 0.05–1 Hz), QRN (atmospheric impulses), carrier drift, 25 % chance of a secondary CW signal at ±50–300 Hz offset (QRM), and a 5 % branch of empty-audio samples labelled with the empty string. Closed the realistic-channel CER from 59.64 % to 52.85 % (−7 pp), zero regression on the AWGN guard.

Live-tested on a real IC-7300 receiver: synthetic gain only partly transferred. The model still produced "letter-soup" (long runs of `E I S T` short-symbol sequences) on quiet bands and weak signals — **the actual release-blocker**. Phase 3.1 was not published.

### Phase 3.2 — anti-hallucination curriculum (v0.2 release)

Same architecture, fine-tuned 80 k steps from Phase 3.1 on a curriculum that targets the live failure mode directly: **30 % random A-Z / 0-9 / punctuation sequences** (no linguistic prior — breaks the model's tendency to fall back on plausible-English letter combinations on weak signal) and **20 % "no decodable signal" samples in three sub-modes** (pure AWGN, AWGN + atmospheric impulses, distant weak CW labelled empty — teaches "real signal but below the floor is still no signal"). Channel impairments unchanged from Phase 3.1.

Result: realistic-channel CER **52.85 % → 8.76 %** (−51 pp). Letter-soup hallucination on pure noise: **98.7 % → 0.0 %**. Live-validated on a real receiver across English / French poetry and contest fragments.

Trade-off: AWGN-only at −5 / −10 dB regresses (4 % → 37 %, 80 % → 91 %) because the anti-hallucination prior is now strong enough that the model emits blank when a signal is real but very faint. This is desirable for real-band audio (which always has propagation impairments — the realistic bench is meaningful) but visible on synthetic AWGN-only ladders. The v0.1 checkpoint stays available on HF for applications where the AWGN-only profile is preferred.

A real-time streaming decoder (`scripts/decode_live.py` v1) ships with v0.2: 6 s sliding window, 2 s hop, central-zone commit, ~4 s end-to-end latency. Replaces the v0.1 chunked decoder which stuttered at chunk boundaries (`CCCCQ`) and cut callsigns mid-word (`F4HY|Y`).

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
- [x] **Phase 3.1** — realistic synthetic channel (QRM, QRN, selective fading) — fine-tune; live-validated, not released
- [x] **Phase 3.2** — anti-hallucination curriculum (random text + 3-mode empty audio) — **v0.2 release**
- [x] **Phase 6** — real-time streaming CLI (`scripts/decode_live.py` v1, sliding window + central-zone commit) — shipped with v0.2
- [x] **Phase 3.3** — multilingual prose corpus (FR / DE / ES / EN, ASCII-normalised) — **v0.3 release**; closes the English-prior bias on French live audio
- [ ] **Phase 3.4** — extend the 46-token vocabulary to 49 tokens with the French-specific Morse codes for `é` (`..-..`), `à` (`.--.-`), and apostrophe (`.----.`); retrain from scratch. Addresses the live-test failure where `L'AUTOMNE` is decoded as `L1AUTOMNE`.
- [ ] **Phase 4.2** — ham-realistic LM corpus (RBN spots, contest logs, real QSO transcripts) — prerequisite for fusion gains
- [ ] **Phase 5** — real-audio finetuning at scale (W1AW aligned transcripts + RBN/SDR pairings + user IC-7300 captures)
- [ ] **Phase 7** — callsign-aware beam search with ITU prefix priors

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
