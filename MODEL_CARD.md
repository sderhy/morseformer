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
language:
  - en
base_model: []
model-index:
  - name: morseformer-phase3-2-rnnt
    results:
      - task:
          type: automatic-speech-recognition
          name: Morse / CW decoding
        dataset:
          type: synthetic
          name: morseformer realistic-channel ladder (Phase 3.1 channel, 50/SNR × 5 WPM × 6 SNR)
        metrics:
          - type: cer
            value: 0.0876
            name: Realistic-channel CER (overall)
          - type: cer
            value: 0.0067
            name: CER at +20 dB realistic
          - type: cer
            value: 0.0040
            name: CER at 0 dB realistic
          - type: cer
            value: 0.0446
            name: CER at −5 dB realistic
          - type: cer
            value: 0.4703
            name: CER at −10 dB realistic
          - type: false_positive_chars_per_sample
            value: 1.05
            name: Mean characters emitted on noise-only audio (target = 0)
---

# morseformer

**Open-source transformer-based Morse / CW decoder.** Conformer encoder with shared CTC and RNN-T heads, trained on synthetic HF-channel audio. Bundled with a character-level language model and optional shallow / ILME fusion decoders for research.

- **Repository**: https://github.com/sderhy/morseformer
- **License**: Apache 2.0
- **Language**: Morse-code text (English letters, digits, punctuation, Morse prosigns — 46-token vocabulary)

This repository hosts the **v0.2.0 release** of morseformer. The Phase 3.2 acoustic model (`rnnt_phase3_2.pt`) is the new recommended checkpoint: it cuts the realistic-channel CER from 52.85 % to 8.76 % vs Phase 3.1 and reduces letter-soup hallucination on noise-only audio from ~11 chars/sample to ~1 char/sample. The Phase 3.0 weights (`rnnt_phase3_0.pt`) are kept for reference / reproducibility.

## What's new in v0.2.0

- **`rnnt_phase3_2.pt`** — new acoustic model trained on a curriculum that adds:
  - 30 % random A-Z / 0-9 / punctuation sequences (no linguistic prior) — breaks the hallucination tendency to fall back on plausible-English letter combinations on weak signal.
  - 20 % "no decodable signal" samples in three modes (pure AWGN, AWGN + atmospheric impulses, distant weak CW labelled empty) — teaches "real signal but below the floor is still no signal".
- **Real-time streaming decoder** (`scripts/decode_live.py`) — sliding 6 s window with central-zone commit. Latency ~4 s. Eliminates the chunk-boundary stutter (`CCCCQ`) and word-cut (`F4HY|Y`) artefacts of the v0.1 chunked decoder.
- **False-positive bench** (`scripts/eval_false_positive.py`) — measures characters emitted on noise-only audio. Phase 3.0/3.1: 11+ chars/sample, ~99 % "letter-soup". Phase 3.2: ~1 char/sample, 0 % "letter-soup".

## Model artifacts

| file | params | description | recommended |
|---|---|---|---|
| `rnnt_phase3_2.pt` | 4.13 M | **v0.2 acoustic model** — Phase 3.1 channel + anti-hallucination curriculum. | ✅ |
| `rnnt_phase3_0.pt` | 4.13 M | v0.1 acoustic model — AWGN-only training, kept for reference. | |
| `lm_phase4_0.pt`   | 4.76 M | Character-level language model (optional, research). Fusion gives no measurable gain on synthetic data. | |

## Intended use

- **Primary**: decoding Morse-code audio (8 kHz sample rate, carrier around 600 Hz, 500 Hz RX bandwidth) into text. Targeted at amateur-radio receivers and SDR applications. v0.2 is now usable on real-band audio with reasonable quality on poetry / QSO / ragchew text in good conditions.
- **Secondary (research)**: a reference implementation for RNN-T and LM-fusion research on a small-vocabulary (46-token) acoustic task with a fully reproducible synthetic data pipeline.

## Limitations

Honest about what v0.2 does and does not do:

1. **Still trained on synthetic audio only.** The model has never seen a real amateur-radio recording during training. v0.2's Phase 3.1 channel covers QSB / QRN / QRM / carrier jitter / drift; v0.3 will add real W1AW recordings and a multilingual prose corpus.
2. **English-language prior bias.** A live test on French poetry decoded `automne` as `LTOM` / `RUTOM` — the model has learned that the letter trigram `TOM` is frequent (it appears in QSO-template names and many English bigrams) and confidently extracts it from foreign words at the expense of surrounding context. Phase 3.3 will add multilingual prose to neutralise this.
3. **AWGN low-SNR regression.** On synthetic AWGN-only at −5 / −10 dB the new model emits less content than v0.1 (CER 37 % / 91 % vs 4 % / 80 % for v0.1). This is a side-effect of the strong "blank-on-noise" prior we trained: the model now refuses to hallucinate. On the realistic-channel bench (with QRM / QSB / drift, closer to real-world conditions), v0.2 is dramatically better at every SNR. Pick `rnnt_phase3_0.pt` if your application needs aggressive output on quiet-but-faint AWGN signals.
4. **6-second training-clip length.** The provided `scripts/decode_audio.py` (offline) and `scripts/decode_live.py` (streaming) both keep the model on its training-length distribution.
5. **WPM range**: trained on uniform WPM ∈ [16, 28]. Outside that range expect degraded accuracy.
6. **No callsign / Q-code-aware beam search yet** (Phase 7 on the roadmap).

## How to use

### Offline decode of a `.wav` file

```bash
pip install torch torchaudio scipy numpy
git clone https://github.com/sderhy/morseformer
cd morseformer
pip install -e ".[audio]"

# Download the v0.2 checkpoint
pip install huggingface_hub
hf download sderhy/morseformer rnnt_phase3_2.pt \
    --local-dir checkpoints/phase3_2

# Decode (any length — chunked into 6 s windows automatically)
python -m scripts.decode_audio my_recording.wav \
    --ckpt checkpoints/phase3_2/rnnt_phase3_2.pt \
    --freq 600 --sample-rate 8000
```

### Real-time streaming decode

```bash
python -m scripts.decode_live --ckpt checkpoints/phase3_2/rnnt_phase3_2.pt
```

Tune your receiver to zero-beat at 600 Hz with a ≈ 500 Hz CW filter. `Ctrl+C` to quit. Latency is ~4 s end-to-end.

## Training data

All training audio is **synthetic**, generated on the fly by the `morse_synth` package:

- **Text corpus (Phase 3.2 mix)**: callsigns 12 % (ITU-prefix-weighted), Q-codes 14 %, QSO templates 22 %, numerics 13 %, English-word stream 6 %, **random A-Z / 0-9 / punctuation 30 %**, plus a 20 % branch of empty-label audio (3 modes: AWGN, AWGN+QRN bursts, distant weak CW). See `morseformer/data/text.py` and `morseformer/data/synthetic.py`.
- **Channel (Phase 3.1, unchanged)**: AWGN SNR ∈ U(0, 30) dB, QSB 0.05–1 Hz / 0–15 dB depth, QRN 0–1 impulse/sec, carrier-frequency jitter ±50 Hz, carrier drift 0–1 Hz/s, 25 % chance of a secondary CW signal at ±50–300 Hz offset (QRM), 500 Hz RX bandpass at 600 Hz centre.
- **Curriculum**: 80 k training steps fp32 fine-tuned from Phase 3.1, peak LR 1e-4, 1 k warm-up, cosine decay, batch size 12, EMA 0.9999.

**No real amateur-radio recordings were used for training v0.2.**

## Evaluation

### Realistic-channel SNR ladder (Phase 3.1 channel — the meaningful real-world bench)

300 samples (10 / WPM × 5 WPM × 6 SNR), full Phase 3.1 channel stack:

| SNR (dB) | v0.1 (Phase 3.0) CER | Phase 3.1 fine-tune CER | **v0.2 (Phase 3.2) CER** | Δ vs v0.1 |
|---|---|---|---|---|
| +20 | 0.2421 | 0.1623 | **0.0067** | −23.54 pp |
| +10 | 0.3816 | 0.3183 | **0.0000** | −38.16 pp |
| +5  | 0.3361 | 0.4077 | **0.0000** | −33.61 pp |
| 0   | 0.5000 | 0.3200 | **0.0040** | −49.60 pp |
| −5  | 0.7871 | 0.7902 | **0.0446** | −74.25 pp |
| −10 | 1.3316 | 1.1726 | **0.4703** | −86.13 pp |
| **overall** | **0.5964** | **0.5285** | **0.0876** | **−51 pp** |

### AWGN guard ladder (no channel — pure AWGN regression check)

| SNR (dB) | v0.1 CER | **v0.2 CER** | Δ |
|---|---|---|---|
| +20 / +10 / +5 / 0 | 0.0000 each | 0.0000 / 0.0000 / 0.0000 / 0.0227 | flat or +2 pp |
| −5  | 0.0407 | 0.3709 | +33 pp (regression) |
| −10 | 0.8019 | 0.9060 | +10 pp (regression) |

The AWGN regression at −5 / −10 dB is real and discussed in *Limitations* §3.

### False-positive bench (noise-only audio, 150 samples × 3 modes — empty-label target)

| metric | v0.1 (Phase 3.1 best) | **v0.2 (Phase 3.2)** |
|---|---|---|
| Mean characters emitted | 11.17 | **1.05** |
| Median | 11.0 | 1.0 |
| Max | 21 | 2 |
| % "letter-soup" (>5 chars) | 98.7 % | **0.0 %** |

## Environmental impact

Training budget for v0.2 (everything, Phase 0 → Phase 3.2): roughly 65 h of single-RTX-3060 wall-clock time (~0.08 kWh/h at load → ~5 kWh total). Carbon impact at EU-grid median: ~1.5 kgCO₂.

## Technical specifications

**Acoustic model** (`rnnt_phase3_2.pt`, 4.13 M params, identical architecture to v0.1):
- Encoder: Conformer d=144, L=8, H=4, ff_expansion=4, conv_kernel=31, RoPE, LayerNorm conv, 4× subsample
- CTC head: Linear(144 → 46)
- PredictionNetwork: Embedding(46, 128) + LSTM(128, 128, 1 layer)
- JointNetwork: Linear(144 → 256) + Linear(128 → 256) + tanh + Linear(256 → 46)

**Language model** (`lm_phase4_0.pt`, 4.76 M params): unchanged from v0.1.

**Vocabulary**: 46 tokens — blank (index 0), 26 uppercase letters A–Z, 10 digits 0–9, 9 punctuation / Morse prosigns ('.', ',', '?', '!', '/', '-', '=', '+', ' ').

**Front-end**: complex bandpass around the carrier (default 600 Hz ± 250 Hz), magnitude, 4× downsample to a 500 Hz frame rate, scalar normalization. Input shape `[B, T, 1]`.

## Citation

```bibtex
@software{morseformer_v0_2_2026,
  author       = {Derhy, Serge},
  title        = {morseformer: open-source transformer-based Morse / CW decoder},
  year         = 2026,
  version      = {v0.2.0},
  url          = {https://github.com/sderhy/morseformer},
  howpublished = {\url{https://huggingface.co/sderhy/morseformer}},
}
```

## Acknowledgements

- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work.
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference.
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model.
- The amateur-radio community — decades of publicly available CW recordings and transcripts.

*73 de morseformer.*
