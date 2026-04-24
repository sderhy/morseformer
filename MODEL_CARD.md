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
  - name: morseformer-phase3-0-rnnt
    results:
      - task:
          type: automatic-speech-recognition
          name: Morse / CW decoding
        dataset:
          type: synthetic
          name: morseformer SNR-ladder (AWGN, 40/WPM × 5 WPM × 6 SNR)
        metrics:
          - type: cer
            value: 0.1317
            name: Overall CER
          - type: cer
            value: 0.0000
            name: CER at +20 dB SNR
          - type: cer
            value: 0.0000
            name: CER at 0 dB SNR
          - type: cer
            value: 0.0280
            name: CER at −5 dB SNR
          - type: cer
            value: 0.7620
            name: CER at −10 dB SNR
---

# morseformer

**Open-source transformer-based Morse / CW decoder.** Conformer encoder with shared CTC and RNN-T heads, trained on synthetic HF-channel audio. Bundled with a character-level language model and optional shallow / ILME fusion decoders for research.

- **Repository**: https://github.com/sderhy/morseformer
- **Paper**: in preparation
- **License**: Apache 2.0
- **Language**: Morse-code text (English letters, digits, punctuation, Morse prosigns — 46-token vocabulary)

This repository hosts the **v0.1.0 release** of morseformer, the first public release of the project. It is intended as a reproducible baseline for amateur-radio CW decoding and as a reference implementation for future transformer-based CW decoders.

## Model artifacts

| file                 | params  | description                                             |
|----------------------|---------|---------------------------------------------------------|
| `rnnt_phase3_0.pt`   | 4.13 M  | Main acoustic model — Conformer encoder + CTC + RNN-T.  |
| `lm_phase4_0.pt`     | 4.76 M  | Character-level language model (optional, research).   |

## Intended use

- **Primary**: decoding Morse-code audio (8 kHz sample rate, carrier around 600 Hz, 500 Hz RX bandwidth) into text. Targeted at amateur-radio receivers and SDR applications.
- **Secondary (research)**: a reference implementation for RNN-T and LM-fusion research on a small-vocabulary (46-token) acoustic task with a fully reproducible synthetic data pipeline.

## Limitations

Be honest about what v0.1 does and does not do:

1. **Trained on synthetic audio only.** The model has never seen a real amateur-radio recording during training. Real-world audio includes channel effects (QRM, QRN, selective fading, multipath, operator timing beyond the trained jitter envelope) that were not modelled in Phase 3.0. Expect degraded performance on real SDR recordings. Phase 3.1 (realistic synthetic channel) and Phase 5 (real-audio finetuning) are the planned fixes.
2. **Fixed training-clip length (6 seconds).** The RNN-T greedy decoder collapses on audio much longer than that when run in a single forward pass. The provided `scripts/decode_audio.py` splits long input into non-overlapping 6 s chunks and concatenates the per-chunk hypotheses — this is the recommended way to decode longer recordings.
3. **AWGN ceiling at −10 dB.** The model's CER at −10 dB SNR is 76 %. This is close to the intrinsic floor for this task at this SNR — a trained human operator struggles in the same regime. For comparison, the rule-based DSP baseline is 1941 % CER (i.e. hallucinating) at the same operating point.
4. **LM fusion gives no measurable gain on synthetic data.** Both shallow fusion and ILME / density-ratio fusion were implemented and swept (see `scripts/eval_fusion.py`). The λ curve is flat inside the ±0.3 pp noise floor at n = 1200. Root cause is that the LM and the RNN-T prediction network share the exact same training text distribution — fusion would need a ham-realistic LM corpus (planned for v0.2) to surface a gain.
5. **WPM range**: the model was trained on uniform WPM ∈ [16, 28]. Outside that range (very slow < 10 WPM or very fast > 35 WPM), expect degraded accuracy.
6. **No callsign / Q-code awareness** in v0.1. The LM biases toward in-distribution text but there is no regex-constrained beam search or ITU-prefix prior yet — that is Phase 7 on the roadmap.

## How to use

```bash
pip install torch torchaudio scipy numpy
git clone https://github.com/sderhy/morseformer
cd morseformer
pip install -e ".[audio]"

# Download the checkpoint
pip install huggingface_hub
hf download sderhy/morseformer rnnt_phase3_0.pt \
    --local-dir checkpoints/phase3_0

# Decode a .wav file (any length — chunked into 6 s windows automatically)
python -m scripts.decode_audio my_recording.wav \
    --ckpt checkpoints/phase3_0/rnnt_phase3_0.pt \
    --freq 600 --sample-rate 8000
```

Expected output on a clean synthetic `CQ DE F4HYY K` @ 20 WPM / +20 dB:

```
[decode] audio: test.wav (6.00 s @ 8000 Hz, carrier 600 Hz)
[decode] model: ... RNN-T, 4,127,212 params, EMA on
[decode] chunks: 1 × 6.00 s

CTC  : 'CQ DE F4HYY K'
RNN-T: 'CQ DE F4HYY K'
```

For research use with the optional LM fusion decoder, see `morseformer/models/fusion.py` and `scripts/eval_fusion.py`.

## Training data

All training audio is **synthetic**, generated on the fly by the `morse_synth` package:

- **Text corpus**: random-callsign strings weighted by real ITU prefix-activity distributions, contest QSO exchanges (CQ WW / ARRL-DX / SS formats), Q-codes and abbreviations, RST reports, ragchew fragments. See `morseformer/data/text.py`.
- **Waveform renderer**: parametric operator model (WPM, element jitter, gap jitter), Morse keying with a sine carrier, HF-channel simulator (AWGN, 500 Hz RX bandpass at 600 Hz centre). See `morse_synth/`.
- **Curriculum for Phase 3.0**: AWGN SNR ∈ U(0, 30) dB, element jitter ∈ U(0, 0.05), gap jitter ∈ U(0, 0.10), WPM ∈ U(16, 28), fixed 6 s utterances at 8 kHz. 80 k training steps.

**No real amateur-radio recordings were used for training v0.1.**

## Training procedure

- **Architecture**: Conformer encoder (d_model = 144, 8 layers, 4 attention heads, 4× time sub-sampling, RoPE attention, LayerNorm-based conv module) + CTC head (single linear) + RNN-T head (PredictionNetwork: 1-layer LSTM d_pred = 128, JointNetwork: d_joint = 256). Blank token index 0.
- **Objective**: multi-task `ctc_weight = 0.3` + `rnnt_weight = 1.0`. Encoder weights bootstrapped from Phase 2.1 (CTC-only) for 80 k steps.
- **Optimizer**: AdamW (lr 3e-4 peak, 2 k warm-up, cosine decay, weight decay 0.1), bf16 autocast, EMA 0.9999.
- **Hardware**: single NVIDIA RTX 3060 (6 GB VRAM), ~10 h wall-clock.

## Evaluation

In-distribution SNR-ladder validation (1200 samples: 40 per WPM × 5 WPM × 6 SNR; synthetic AWGN + 500 Hz RX filter; same operator-jitter distribution as training):

| SNR (dB) | CER      | WER      |
|----------|----------|----------|
| +20      | 0.0000   | 0.0000   |
| +10      | 0.0000   | 0.0000   |
|  +5      | 0.0000   | 0.0000   |
|   0      | 0.0000   | 0.0000   |
|  −5      | 0.0280   | 0.0737   |
| −10      | 0.7620   | 0.9460   |
| **overall** | **0.1317** | **0.1718** |

Comparison to the rule-based DSP baseline (`fldigi`-class threshold decoder), same bench, same seed:

| SNR (dB) | baseline CER | Phase 3.0 CER | speed-up |
|----------|--------------|---------------|----------|
|   0      |   2.5355     |   0.0000      | ∞        |
|  −5      |   5.8327     |   0.0280      | 208 ×    |
| −10      |  19.4111     |   0.7620      |  25 ×    |

At SNR ≥ 0 dB, morseformer is effectively perfect. The overall 13.17 % CER is driven entirely by the −10 dB bin, which is near the intrinsic floor for this task at this SNR.

## Environmental impact

Training budget (including Phase 0 → Phase 4.1): roughly 50 h of single-RTX-3060 wall-clock time (~0.08 kWh/h at load → ~4 kWh total). No multi-GPU, no multi-node, no hyperparameter search. Carbon impact at EU-grid median (~300 gCO₂/kWh): ~1.2 kgCO₂.

## Technical specifications

**Acoustic model** (`rnnt_phase3_0.pt`, 4.13 M params):
- Encoder: Conformer d=144, L=8, H=4, ff_expansion=4, conv_kernel=31, RoPE, LayerNorm conv, 4× subsample
- CTC head: Linear(144 → 46)
- PredictionNetwork: Embedding(46, 128) + LSTM(128, 128, 1 layer)
- JointNetwork: Linear(144 → 256) + Linear(128 → 256) + tanh + Linear(256 → 46)

**Language model** (`lm_phase4_0.pt`, 4.76 M params):
- Decoder-only GPT, d_model=256, L=6, H=4, ff=1024 (SwiGLU), context=256, RoPE, RMSNorm, tied embeddings, causal SDPA

**Vocabulary**: 46 tokens — blank (index 0), 26 uppercase letters A–Z, 10 digits 0–9, 9 punctuation / Morse prosigns ('.', ',', '?', '/', '-', '=', '+', ' ', ''').

**Front-end**: complex bandpass around the carrier (default 600 Hz ± 250 Hz), magnitude, 4× downsample to a 500 Hz frame rate, scalar normalization. Input shape `[B, T, 1]`.

## Citation

```bibtex
@software{morseformer_v0_1_2026,
  author       = {Derhy, Serge},
  title        = {morseformer: open-source transformer-based Morse / CW decoder},
  year         = 2026,
  version      = {v0.1.0},
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
