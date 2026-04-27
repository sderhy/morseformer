---
license: apache-2.0
library_name: pytorch
pipeline_tag: automatic-speech-recognition
tags:
  - morse
  - cw
  - amateur-radio
  - ham-radio
  - speech-recognition
  - conformer
  - rnn-transducer
  - audio
  - multilingual
language:
  - en
  - fr
  - de
  - es
base_model: []
model-index:
  - name: morseformer-phase3-3-rnnt
    results:
      - task:
          type: automatic-speech-recognition
          name: Morse / CW decoding
        dataset:
          type: synthetic
          name: morseformer realistic-channel ladder (Phase 3.1 channel, 40/wpm × 5 wpm × 6 snr = 1200 samples)
        metrics:
          - type: cer
            value: 0.0749
            name: Realistic-channel CER (overall)
          - type: cer
            value: 0.0000
            name: CER at +20 dB realistic
          - type: cer
            value: 0.0030
            name: CER at 0 dB realistic
          - type: cer
            value: 0.0502
            name: CER at −5 dB realistic
          - type: cer
            value: 0.3965
            name: CER at −10 dB realistic
          - type: false_positive_chars_per_sample
            value: 1.01
            name: Mean characters emitted on noise-only audio (target = 0)
---

# morseformer

**Open-source transformer-based Morse / CW decoder.** Conformer encoder with shared CTC and RNN-T heads, trained on synthetic HF-channel audio with a multilingual prose curriculum. Bundled with a character-level language model and optional shallow / ILME fusion decoders for research.

- **Repository**: https://github.com/sderhy/morseformer
- **License**: Apache 2.0
- **Languages on input**: English plus a multilingual prose mix in French, German, and Spanish (ASCII-normalised — diacritics stripped, German umlauts transliterated to AE/OE/UE/SS).
- **Output vocabulary**: 46 tokens (A–Z, 0–9, space, `. , ? ! / = + -`).

This repository hosts the **v0.3.0 release** of morseformer. The Phase 3.3 acoustic model (`rnnt_phase3_3.pt`) is the new recommended checkpoint: it cuts the realistic-channel CER from 8.38 % (Phase 3.2) to **7.49 %**, narrows the AWGN guard regression at low SNR, preserves the anti-hallucination behaviour shipped in v0.2, and adds a multilingual prose mix that closes the English-prior bias observed on real French QSOs in v0.2 (e.g. `automne` decoded as `LTOM`/`RUTOM`).

## What's new in v0.3.0

- **`rnnt_phase3_3.pt`** — fine-tuned 50 k steps from Phase 3.2 with a Phase 3.3 curriculum that adds a 12 % slice of multilingual prose (FR/DE/ES/EN) drawn from a Project Gutenberg corpus, normalised to the 46-token ASCII vocabulary. The random-clump anti-hallucination weight is reduced from 30 % to 20 % to make room; everything else (channel, empty-audio modes) is identical to Phase 3.2.
- **Streaming-decoder space-bug fix.** `tokenizer.decode()` previously called `.strip()` on every output, which silently dropped any inter-word space landing at a window boundary in the streaming decoder, producing collisions like `LE PONT MIRABEAUCOULELASEINE`. Fixed in v0.3 — all checkpoints (v0.1, v0.2, v0.3) benefit if you rebuild from source. Two regression tests were added.
- **Better numerics on the realistic bench.** Across the SNR ladder the new model is at parity or better than v0.2 at every operating point, with the largest gain at −10 dB realistic (44.4 % → 39.6 %, −4.77 pp absolute). On the AWGN guard ladder the worst-SNR regression v0.2 traded for anti-hallucination is partly recovered (e.g. −5 dB: 33.5 % → 29.4 %).

## Model artifacts

| file | params | description | recommended |
|---|---|---|---|
| `rnnt_phase3_3.pt` | 4.13 M | **v0.3 acoustic model** — Phase 3.2 channel + multilingual prose curriculum. | ✅ |
| `rnnt_phase3_2.pt` | 4.13 M | v0.2 acoustic model — anti-hallucination curriculum, no multilingual data. Kept for comparison / reproducibility. | |
| `rnnt_phase3_0.pt` | 4.13 M | v0.1 acoustic model — AWGN-only training. | |
| `lm_phase4_0.pt`   | 4.76 M | Character-level language model (optional, research). Fusion gives no measurable gain on synthetic data. | |

## Intended use

- **Primary**: decoding Morse-code audio (8 kHz sample rate, carrier around 600 Hz, 500 Hz RX bandwidth) into text. Targeted at amateur-radio receivers and SDR applications. v0.3 has been live-validated on a real IC-7300 on a mixed-language stream (CQ/macros + Verlaine + Poe + Apollinaire) and decodes recognisable French / English prose word-by-word with proper spacing.
- **Secondary (research)**: a reference implementation for RNN-T and LM-fusion research on a small-vocabulary acoustic task with a fully reproducible synthetic data pipeline.

## Limitations

Honest about what v0.3 does and does not do:

1. **Still trained on synthetic audio only.** The model has never seen a real amateur-radio recording during training. v0.3's channel covers QSB / QRN / QRM / carrier jitter / drift; a real-audio fine-tune (W1AW, contest captures, or the user's own IC-7300 captures) is the natural next step.
2. **Diacritics and apostrophes are mapped to ASCII.** The 46-token vocabulary has no tokens for `é` (Morse `..-..`), `à` (`.--.-`), or apostrophe (`.----.`). On live French audio the model maps the apostrophe code to the digit `1` (because `1` is `.----`), so `L'AUTOMNE` comes out as `L1AUTOMNE`. The training corpus was ASCII-normalised so this does not hurt accuracy on the rest of the word — only the punctuation token is wrong. A future Phase 3.4 will extend the vocabulary to 49 tokens and retrain from scratch.
3. **AWGN low-SNR residual.** v0.3 partially recovers the AWGN-only regression v0.2 introduced (see *Evaluation* below) but at −10 dB pure-AWGN the model still emits less than v0.1. On the realistic-channel bench v0.3 is dramatically better than v0.1 at every SNR. Pick `rnnt_phase3_0.pt` if your application needs aggressive output on quiet-but-faint AWGN signals with no other impairment.
4. **Failure modes seen in live testing.** On heavy operator jitter the model occasionally confuses `M` (`--`) with `Q` (`--.-`) and produces letter duplications (`COULE` → `COUULE`); on weak signal it can emit short letter-soup patches (`QIDEN` for `MAIDEN`, `NAG` for `NAME`). These are documented in [project_live_observations_phase3_3.md](https://github.com/sderhy/morseformer/blob/main/.claude/projects/-home-serge-morseformer/memory/project_live_observations_phase3_3.md) and are the priority for Phase 3.4.
5. **6-second training-clip length.** The provided `scripts/decode_audio.py` (offline) and `scripts/decode_live.py` (streaming) both keep the model on its training-length distribution.
6. **WPM range**: trained on uniform WPM ∈ [16, 28]. Outside that range expect degraded accuracy.
7. **No callsign / Q-code-aware beam search yet** (Phase 7 on the roadmap).

## How to use

### Offline decode of a `.wav` file

```bash
pip install torch torchaudio scipy numpy
git clone https://github.com/sderhy/morseformer
cd morseformer
pip install -e ".[audio]"

# Download the v0.3 checkpoint
pip install huggingface_hub
hf download sderhy/morseformer rnnt_phase3_3.pt \
    --local-dir checkpoints/phase3_3

# Decode (any length — chunked into 6 s windows automatically)
python -m scripts.decode_audio my_recording.wav \
    --ckpt checkpoints/phase3_3/rnnt_phase3_3.pt \
    --freq 600 --sample-rate 8000
```

### Real-time streaming decode

```bash
python -m scripts.decode_live --ckpt checkpoints/phase3_3/rnnt_phase3_3.pt
```

Tune your receiver to zero-beat at 600 Hz with a ≈ 500 Hz CW filter. `Ctrl+C` to quit. Latency is ~4 s end-to-end. The streaming decoder now preserves inter-word spaces at window boundaries — multi-word prose is rendered with proper word segmentation.

## Training data

All training audio is **synthetic**, generated on the fly by the `morseformer.data.synthetic` package:

- **Text corpus (Phase 3.3 mix)**: callsigns 12 % (ITU-prefix-weighted), Q-codes 14 %, QSO templates 25 %, numerics 13 %, English-word stream 4 %, **random A-Z / 0-9 / punctuation 20 %**, **multilingual prose 12 %** (FR/DE/ES/EN, drawn from 12 Project Gutenberg books, ASCII-normalised — see `data/corpus/fetch.py`), plus a 20 % branch of empty-label audio (3 modes: AWGN, AWGN+QRN bursts, distant weak CW). Defined in `morseformer/data/text.py:PHASE_3_3_MIX` and `morseformer/data/synthetic.py:DatasetConfig.phase_3_3()`.
- **Channel (unchanged from Phase 3.1)**: AWGN SNR ∈ U(0, 30) dB, QSB 0.05–1 Hz / 0–15 dB depth, QRN 0–1 impulse/sec, carrier-frequency jitter ±50 Hz, carrier drift 0–1 Hz/s, 25 % chance of a secondary CW signal at ±50–300 Hz offset (QRM), 500 Hz RX bandpass at 600 Hz centre.
- **Curriculum**: 50 k training steps fp32 fine-tuned from `checkpoints/phase3_2/last.pt`, peak LR 5e-5 (half of Phase 3.2's 1e-4 — gentler for fine-tune), 500-step warm-up, cosine decay, batch size 12, EMA 0.9999. Wall time ~6.6 h on a single RTX 3060 Laptop.

The multilingual prose corpus (`data/corpus/prose.txt`, ~8 MB raw) is gitignored but reproducible by running `python data/corpus/fetch.py` — it downloads the same 12 Project Gutenberg books used during training.

**No real amateur-radio recordings were used for training v0.3.**

## Evaluation

### Realistic-channel SNR ladder (Phase 3.1 channel — the meaningful real-world bench)

1200 samples (40 / WPM × 5 WPM × 6 SNR), full Phase 3.1 channel stack:

| SNR (dB) | v0.1 (Phase 3.0) CER | v0.2 (Phase 3.2 last) CER | **v0.3 (Phase 3.3 best) CER** | Δ vs v0.2 |
|---|---|---|---|---|
| +20 | 0.2421 | 0.0017 | **0.0000** | −0.17 pp |
| +10 | 0.3816 | 0.0000 | **0.0000** | 0.00 pp |
| +5  | 0.3361 | 0.0000 | **0.0000** | 0.00 pp |
| 0   | 0.5000 | 0.0030 | **0.0030** | 0.00 pp |
| −5  | 0.7871 | 0.0542 | **0.0502** | −0.40 pp |
| −10 | 1.3316 | 0.4442 | **0.3965** | **−4.77 pp** |
| **overall** | **0.5964** | **0.0838** | **0.0749** | **−0.89 pp** |

### AWGN guard ladder (no channel — pure AWGN regression check)

1200 samples, same WPM × SNR grid, AWGN only:

| SNR (dB) | v0.1 CER | v0.2 (Phase 3.2 last) CER | **v0.3 (Phase 3.3 best) CER** | Δ vs v0.2 |
|---|---|---|---|---|
| +20 | 0.0000 | 0.0000 | **0.0000** | 0.00 pp |
| +10 | 0.0000 | 0.0000 | **0.0000** | 0.00 pp |
| +5  | 0.0000 | 0.0000 | **0.0000** | 0.00 pp |
| 0   | 0.0000 | 0.0229 | **0.0099** | −1.30 pp |
| −5  | 0.0407 | 0.3348 | **0.2943** | −4.05 pp |
| −10 | 0.8019 | 0.8919 | **0.8826** | −0.92 pp |
| **overall** | — | **0.2083** | **0.1978** | −1.05 pp |

The −5 / −10 dB AWGN regression v0.2 introduced (in exchange for the anti-hallucination prior) is partly recovered in v0.3 — the multilingual prose appears to make the prior less brittle on noise-shaped inputs.

### False-positive bench (noise-only audio, 150 samples × 3 modes — empty-label target)

| metric | v0.1 (Phase 3.1 best) | v0.2 (Phase 3.2 last) | **v0.3 (Phase 3.3 best)** |
|---|---|---|---|
| Mean characters emitted | 11.17 | 1.05 | **1.01** |
| Median | 11.0 | 1.0 | 1.0 |
| Max | 21 | 2 | 2 |
| % "letter-soup" (>5 chars) | 98.7 % | 0.0 % | **0.0 %** |

### Live validation

A live-radio test on 2026-04-27 with an IC-7300 driving the streaming decoder, mixed-language stream of CQ macros + Verlaine *Chanson d'automne* + Poe *Annabel Lee* + Apollinaire *Le Pont Mirabeau*. v0.3 (`rnnt_phase3_3.pt`) decoded the operator's own callsign macros cleanly (`F4HYY F4HYY`), preserved word boundaries in the prose passages (`LE PONT MIRABEAU COULE LA SEINE ET O NOI NOS ET AMOURS`), and rendered French verses with recognisable accuracy where v0.2 had previously collapsed `automne` into English-prior fragments.

## Environmental impact

Training budget for v0.3 (everything, Phase 0 → Phase 3.3): roughly 72 h of single-RTX-3060 wall-clock time (~0.08 kWh/h at load → ~6 kWh total). Carbon impact at EU-grid median: ~1.7 kgCO₂.

## Technical specifications

**Acoustic model** (`rnnt_phase3_3.pt`, 4.13 M params, identical architecture to v0.1 / v0.2):
- Encoder: Conformer d=144, L=8, H=4, ff_expansion=4, conv_kernel=31, RoPE, LayerNorm conv, 4× subsample
- CTC head: Linear(144 → 46)
- PredictionNetwork: Embedding(46, 128) + LSTM(128, 128, 1 layer)
- JointNetwork: Linear(144 → 256) + Linear(128 → 256) + tanh + Linear(256 → 46)

**Language model** (`lm_phase4_0.pt`, 4.76 M params): unchanged from v0.1.

**Vocabulary**: 46 tokens — blank (index 0), 26 uppercase letters A–Z, 10 digits 0–9, 9 punctuation / Morse prosigns ('.', ',', '?', '!', '/', '-', '=', '+', ' ').

**Front-end**: complex bandpass around the carrier (default 600 Hz ± 250 Hz), magnitude, 4× downsample to a 500 Hz frame rate, scalar normalization. Input shape `[B, T, 1]`.

## Citation

```bibtex
@software{morseformer_v0_3_2026,
  author       = {Derhy, Serge},
  title        = {morseformer: open-source transformer-based Morse / CW decoder},
  year         = 2026,
  version      = {v0.3.0},
  url          = {https://github.com/sderhy/morseformer},
  howpublished = {\url{https://huggingface.co/sderhy/morseformer}},
}
```

## Acknowledgements

- **Sébastien Derhy** — design, engineering, and on-air validation of morseformer.
- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work.
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference.
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model.
- **Project Gutenberg** — public-domain literary texts in English, French, German, and Spanish used to build the Phase 3.3 multilingual prose corpus.
- The amateur-radio community — decades of publicly available CW recordings and transcripts.

*73 de morseformer.*
