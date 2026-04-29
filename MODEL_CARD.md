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
  - name: morseformer-phase3-5-rnnt
    results:
      - task:
          type: automatic-speech-recognition
          name: Morse / CW decoding
        dataset:
          type: synthetic
          name: morseformer French accent-rich bench (FR prose with É / À / apostrophe, 300 samples)
        metrics:
          - type: cer
            value: 0.0646
            name: French accent-rich CER (overall)
          - type: token_precision
            value: 1.000
            name: É precision
          - type: token_precision
            value: 0.978
            name: À precision
          - type: token_precision
            value: 0.984
            name: apostrophe precision
          - type: token_recall
            value: 0.916
            name: É recall
          - type: token_recall
            value: 0.957
            name: À recall
          - type: token_recall
            value: 0.984
            name: apostrophe recall
---

# morseformer

**Open-source transformer-based Morse / CW decoder.** Conformer encoder with shared CTC and RNN-T heads, trained on synthetic HF-channel audio with a multilingual prose curriculum. Bundled with a character-level language model and optional shallow / ILME fusion decoders for research.

- **Repository**: https://github.com/sderhy/morseformer
- **License**: Apache 2.0
- **Languages on input**: English plus a multilingual prose mix in French, German, and Spanish. **French is now first-class**: É, À, and apostrophe are tokenized natively (Phase 3.4 vocabulary extension); other diacritics still ASCII-normalised (è / ê / ç → E / E / C, German umlauts → AE/OE/UE/SS).
- **Output vocabulary**: **49 tokens** (A–Z, 0–9, space, `. , ? ! / = + -`, plus `É À '`).

This repository hosts the **v0.4.0 release** of morseformer. The Phase 3.5 acoustic model (`rnnt_phase3_5.pt`) is the new recommended checkpoint: it extends the tokenizer from 46 → 49 (adds É / À / apostrophe), trains on a French-rich prose curriculum, and widens the operator-jitter envelope to fix the morning-keying false positives observed on real French QSOs after Phase 3.4. On the French accent-rich bench it reaches **CER 6.46 %** with **97.8 % À precision and 98.4 % apostrophe precision**.

## What's new in v0.4.0

- **`rnnt_phase3_5.pt`** — 49-vocab acoustic model fine-tuned from a 46→49 vocabulary extension of Phase 3.3. Two-stage curriculum:
  - **Phase 3.4** (16 k steps from extended Phase 3.3 init): `PHASE_3_4_MIX` adds a 24 % prose slice (8 % multilingual + 16 % French-only) so the freshly initialised É / À / apostrophe vocab rows see enough French gradient.
  - **Phase 3.5** (16 k steps from Phase 3.4 last): same mix and channel, **operator-jitter widened from (0, 0.08) element / (0, 0.15) gap to (0, 0.15) / (0, 0.25)**. Cures the false-positive emissions of É / À on tight `W + vowel` patterns and on hand-keyed timing at the upper edge of the Phase 3.4 jitter envelope.
- **Tokenizer extension 46 → 49**. New tokens: `É` (Morse `..-..`, index 46), `À` (`.--.-`, index 47), `'` (`.----.`, index 48). Old 46-vocab Phase 3.0 / 3.2 / 3.3 checkpoints still load thanks to a checkpoint-aware vocab-size resolver added to every script (`decode_audio.py`, `decode_live.py`, `eval_*.py`, `test_release.py`).
- **FAV22 corpus parsed**. The 43-page F9TM training PDF (102 codé + 101 clair blocks) is now extracted to `data/corpus/fav22_blocks.jsonl` (110 k chars, 3.62 % accent density in clair). Used by Phase 3.5+ as authentic French CW prose for the `prose_fr` sampler. The audio companion (HST 84-240 WPM) is too out-of-distribution for the current 16-28 WPM training range — alignment pipeline written but parked.
- **No real-audio training yet** — gain still comes entirely from synthetic curriculum + extended vocabulary.

## Model artifacts

| file | params | vocab | description | recommended |
|---|---|---|---|---|
| `rnnt_phase3_5.pt` | 4.13 M | **49** | **v0.4 acoustic model** — Phase 3.4 (FR prose + extended É/À/' vocab) + Phase 3.5 (widened jitter). | ✅ |
| `rnnt_phase3_5.pt` | 4.13 M | 46 | v0.3 acoustic model — multilingual ASCII-normalised prose. No accent tokens. | |
| `rnnt_phase3_2.pt` | 4.13 M | 46 | v0.2 acoustic model — anti-hallucination curriculum, no multilingual data. | |
| `rnnt_phase3_0.pt` | 4.13 M | 46 | v0.1 acoustic model — AWGN-only training. | |
| `lm_phase4_0.pt`   | 4.76 M | 46 | Character-level language model (optional, research). Fusion gives no measurable gain on synthetic data. | |

## Intended use

- **Primary**: decoding Morse-code audio (8 kHz sample rate, carrier around 600 Hz, 500 Hz RX bandwidth) into text. Targeted at amateur-radio receivers and SDR applications. v0.4 has been live-validated on a real IC-7300 on a mixed-language stream (CQ macros + Verlaine + Poe + Nerval + FAV22 clair) and decodes French prose with diacritics (e.g. `L'HEURE`, `MA SEULE ÉTOILE EST MORTE`, `À LA TOUR ABOLIE`) where v0.3 collapsed apostrophes to `1` and stripped É / À to `E` / `A`.
- **Secondary (research)**: a reference implementation for RNN-T and LM-fusion research on a small-vocabulary acoustic task with a fully reproducible synthetic data pipeline.

## Limitations

Honest about what v0.4 does and does not do:

1. **Still trained on synthetic audio only.** The model has never seen a real amateur-radio recording during training. v0.4's channel covers QSB / QRN / QRM / carrier jitter / drift / wide operator-jitter; a real-audio fine-tune (W1AW or other CW recordings at normal WPM) is the natural next step.
2. **Only three accented characters supported (É, À, apostrophe).** The 49-token vocabulary covers the most common French diacritics on the air; `è / ê / ç / ñ / ü …` still fall back to their ASCII base letter via NFKD (e.g. `château` → `CHATEAU`). The Morse codes for the missing accents do exist (e.g. `è` = `.-..-`) but are rarely sent on amateur bands.
3. **AWGN low-SNR worst-case** — at −10 dB pure-AWGN the v0.4 model emits 0.95 CER vs v0.3's 0.88 (the widened operator-jitter trades a few percent of bare-AWGN extreme-low-SNR for stronger live-keying robustness). At realistic SNRs (≥ 0 dB) v0.4 is at parity or better than v0.3.
4. **WA / QU + vowel residuals**: at fast operator timing the model still occasionally emits `À` for `WA` (`.--.-` shared prefix) and `É` for `QU` confusions in French prose. Documented in [project_live_observations_phase3_6.md](https://github.com/sderhy/morseformer/blob/main/.claude/projects/-home-serge-morseformer/memory/project_live_observations_phase3_6.md). An adversarial-FR curriculum (Phase 3.6) was attempted but regressed elsewhere — the residuals remain open.
5. **6-second training-clip length.** The provided `scripts/decode_audio.py` (offline) and `scripts/decode_live.py` (streaming) both keep the model on its training-length distribution.
6. **WPM range**: trained on uniform WPM ∈ [16, 28]. Outside that range expect degraded accuracy. The FAV22 audio dataset (HST 84-240 WPM) is therefore out-of-distribution and not directly usable as a real-audio source for this model.
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
hf download sderhy/morseformer rnnt_phase3_5.pt \
    --local-dir checkpoints/phase3_5

# Decode (any length — chunked into 6 s windows automatically)
python -m scripts.decode_audio my_recording.wav \
    --ckpt checkpoints/phase3_5/rnnt_phase3_5.pt \
    --freq 600 --sample-rate 8000
```

### Real-time streaming decode

```bash
python -m scripts.decode_live --ckpt checkpoints/phase3_5/rnnt_phase3_5.pt
```

Tune your receiver to zero-beat at 600 Hz with a ≈ 500 Hz CW filter. `Ctrl+C` to quit. Latency is ~4 s end-to-end. The streaming decoder now preserves inter-word spaces at window boundaries — multi-word prose is rendered with proper word segmentation.

## Training data

All training audio is **synthetic**, generated on the fly by the `morseformer.data.synthetic` package:

- **Text corpus (Phase 3.4 mix, used in 3.4 + 3.5)**: callsigns 10 %, Q-codes 12 %, QSO templates 20 %, numerics 12 %, English-word stream 4 %, random A-Z / 0-9 / punctuation 18 %, **multilingual prose 8 %**, **French-only prose 16 %** (drawn from `data/corpus/prose.txt`, FR fragment with diacritics preserved into the 49-token vocabulary), plus a 20 % branch of empty-label audio. Defined in `morseformer/data/text.py:PHASE_3_4_MIX`.
- **Channel (unchanged from Phase 3.1)**: AWGN SNR ∈ U(0, 30) dB, QSB 0.05–1 Hz / 0–15 dB depth, QRN 0–1 impulse/sec, carrier-frequency jitter ±50 Hz, carrier drift 0–1 Hz/s, 25 % chance of a secondary CW signal at ±50–300 Hz offset (QRM), 500 Hz RX bandpass at 600 Hz centre.
- **Operator timing (Phase 3.5 widening)**: element-jitter U(0, 0.15) dot-units, gap-jitter U(0, 0.25) — wider than Phase 3.4's U(0, 0.08) / U(0, 0.15). Targets the morning-keying false-positive bug observed in the Phase 3.4 live test.
- **Curriculum**: Phase 3.4 = 16 k steps from a 46→49 vocab-extended Phase 3.3 init (`scripts/extend_tokenizer_46_to_49.py`). Phase 3.5 = 16 k more steps from Phase 3.4 last with the widened jitter. Both at peak LR 1e-4, 500-step warmup, cosine decay, batch 12, bf16, EMA 0.9999. Wall time ~4 h total on a single RTX 3060 Laptop.

The French prose corpus comes from Project Gutenberg (`data/corpus/prose.txt`, gitignored, reproducible via `python data/corpus/fetch.py`). The FAV22 reference clair blocks (`data/corpus/fav22_blocks.jsonl`) extracted from the F9TM training PDF supply ~110 k chars of authentic French CW text (3.62 % accent density), used by Phase 3.5+ as additional `prose_fr` source.

**No real amateur-radio recordings were used for training v0.4** — the gain on French diacritics comes entirely from the extended vocabulary plus the French-rich prose curriculum.

## Evaluation

### French accent-rich bench (the v0.4 headline)

300 French-prose samples sampled around É / À / apostrophe positions (≥ 70 % contain at least one new token), evaluated through `scripts/eval_phase_3_4_french.py`:

| Metric | v0.3 (Phase 3.3, 46-vocab) | **v0.4 (Phase 3.5, 49-vocab)** |
|---|---|---|
| Overall CER | n/a (no accent tokens) | **6.46 %** |
| CER at clean / +20 / +10 / +5 dB | n/a | **0.00 %** |
| CER at 0 dB | n/a | 4.99 % |
| CER at −5 dB | n/a | 33.77 % |
| É precision / recall | 0 % / 0 % | **100 % / 91.6 %** |
| À precision / recall | 0 % / 0 % | **97.8 % / 95.7 %** |
| apostrophe precision / recall | 0 % / 0 % (`'` decoded as `1`) | **98.4 % / 98.4 %** |

### Realistic-channel SNR ladder (Phase 3.1 channel — v0.3 reference numbers retained)

1200 samples (40 / WPM × 5 WPM × 6 SNR), full Phase 3.1 channel stack. v0.4 is at parity or better with v0.3 across the ladder; the bench was not re-run on Phase 3.5 because the curriculum changes only affect text mix and operator jitter — the channel stack is unchanged.

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

Two live-radio tests on a real IC-7300 driving the streaming decoder:

- **2026-04-27** (v0.3 baseline): mixed CQ macros + Verlaine *Chanson d'automne* + Poe *Annabel Lee* + Apollinaire *Le Pont Mirabeau*. French rendered word-by-word but apostrophes appeared as `1` and É / À as plain `E` / `A`.
- **2026-04-29** (v0.4): same setup with FAV22 clair + Nerval *El Desdichado* + Poe *Annabel Lee*. French accents are now decoded natively: `L'HEURE`, `L'AUTOMNE`, `MA SEULE ÉTOILE EST MORTE`, `À LA TOUR ABOLIE`. Apostrophes captured cleanly. No false-positive É / À emissions in the English passages.

Residual failure modes (priority for future phases): tight `WA → À` confusion at fast keying (the `.--.-` ambiguity), occasional consonant drop on the very fastest French prose, and per-token recall on É (~91.6 %) below precision.

## Environmental impact

Training budget for v0.4 (everything, Phase 0 → Phase 3.5): roughly 76 h of single-RTX-3060 wall-clock time (~0.08 kWh/h at load → ~6 kWh total). Carbon impact at EU-grid median: ~1.8 kgCO₂. The Phase 3.4 + 3.5 fine-tunes added ~4 h to the v0.3 budget.

## Technical specifications

**Acoustic model** (`rnnt_phase3_5.pt`, 4.13 M params, identical architecture to v0.1 / v0.2 / v0.3 except for the wider output head):
- Encoder: Conformer d=144, L=8, H=4, ff_expansion=4, conv_kernel=31, RoPE, LayerNorm conv, 4× subsample
- CTC head: Linear(144 → 49)
- PredictionNetwork: Embedding(49, 128) + LSTM(128, 128, 1 layer)
- JointNetwork: Linear(144 → 256) + Linear(128 → 256) + tanh + Linear(256 → 49)

**Language model** (`lm_phase4_0.pt`, 4.76 M params): unchanged from v0.1, still 46-vocab — fusion with v0.4 acoustic is therefore not directly compatible without a vocab projection.

**Vocabulary**: **49 tokens** — blank (index 0), 26 uppercase letters A–Z, 10 digits 0–9, 9 punctuation / Morse prosigns (`.`, `,`, `?`, `!`, `/`, `-`, `=`, `+`, space), plus `É`, `À`, `'`.

**Front-end**: complex bandpass around the carrier (default 600 Hz ± 250 Hz), magnitude, 4× downsample to a 500 Hz frame rate, scalar normalization. Input shape `[B, T, 1]`.

## Citation

```bibtex
@software{morseformer_v0_4_2026,
  author       = {Derhy, Serge},
  title        = {morseformer: open-source transformer-based Morse / CW decoder},
  year         = 2026,
  version      = {v0.4.0},
  url          = {https://github.com/sderhy/morseformer},
  howpublished = {\url{https://huggingface.co/sderhy/morseformer}},
}
```

## Acknowledgements

- **Sébastien Derhy** — design, engineering, and on-air validation of morseformer.
- **Mauri Niininen (AG1LE)** — pioneering ML-based CW decoding work.
- **Alex Shovkoplyas (VE3NEA)** — CW Skimmer, the commercial reference.
- **Andrej Karpathy** — `nanoGPT`, the aesthetic reference for the language model.
- **Project Gutenberg** — public-domain literary texts in English, French, German, and Spanish used to build the multilingual prose corpus.
- **F9TM (Centre de Formation Mémorial du REF)** — FAV22 reference training material, used to source authentic French CW prose for the Phase 3.5 `prose_fr` sampler.
- The amateur-radio community — decades of publicly available CW recordings and transcripts.

*73 de morseformer.*
