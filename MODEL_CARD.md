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
  - name: morseformer-phase5-5-rnnt
    results:
      - task:
          type: automatic-speech-recognition
          name: Morse / CW decoding
        dataset:
          type: mixed
          name: morseformer LCWO + websdr bench v1 (6 clips)
        metrics:
          - type: cer
            value: 0.0285
            name: mean CER
---

# morseformer

**Open-source transformer-based Morse / CW decoder.** Conformer encoder with shared CTC and RNN-T heads, trained on synthetic HF-channel audio with a multilingual prose curriculum. Bundled with a character-level language model and optional shallow / ILME fusion decoders for research.

- **Repository**: https://github.com/sderhy/morseformer
- **License**: Apache 2.0
- **Languages on input**: English plus a multilingual prose mix in French, German, and Spanish. **French is now first-class**: É, À, and apostrophe are tokenized natively (Phase 3.4 vocabulary extension); other diacritics still ASCII-normalised (è / ê / ç → E / E / C, German umlauts → AE/OE/UE/SS).
- **Output vocabulary**: **49 tokens** (A–Z, 0–9, space, `. , ? ! / = + -`, plus `É À '`).

This repository hosts the **v0.6.3 release** of morseformer. The recommended acoustic checkpoint is `rnnt_phase5_5.pt` (4.13 M params, 49-vocab), paired optionally with `lm_phase5_2.pt` for offline shallow-fusion decoding. v0.6.3 is a packaging / CI refresh on top of the v0.6.2 acoustic revert from `rnnt_phase5_8.pt` back to `rnnt_phase5_5.pt`, after the reproducible LCWO + websdr bench showed a **24 % relative mean-CER win** for `rnnt_phase5_5`.

## What's new in v0.6.3

- Packaging refresh: `pyproject.toml` and `morseformer.__version__` are aligned at `0.6.3`.
- CI runs Python 3.10 / 3.11 / 3.12 with CPU PyTorch wheels, `ruff check`, and `pytest`.
- No model change from v0.6.2. The recommended acoustic remains `rnnt_phase5_5.pt`; the recommended LM remains `lm_phase5_2.pt`.

## What's new in v0.6.2

The first reproducible real-audio bench (`eval/bench_lcwo.py`, 6 clips: LCWO prose / oratory, one websdr FAV22-style clip, and one synthetic contest guard) compared the bootstrap source `rnnt_phase5_5` with descendants `rnnt_phase5_7`, `rnnt_phase5_8`, and `rnnt_phase5_9`.

| Acoustic | mean CER |
|---|---:|
| **`rnnt_phase5_5`** | **2.85 %** |
| `rnnt_phase5_7` | 3.81 % |
| `rnnt_phase5_8` | 3.93 % |
| `rnnt_phase5_9` | 3.93 % |

`rnnt_phase5_5` wins 5 / 6 clips and beats the v0.6.0/v0.6.1 acoustic (`rnnt_phase5_8`) by **24 % relative mean CER**. The registry and all presets therefore reverted to `rnnt_phase5_5`.

## What's new in v0.6.0

- `decode_audio` switched to the same central-zone-commit sliding-window decoder as live streaming, fixing visible word cuts at 6 s chunk boundaries.
- `rnnt_phase5_8.pt` added an English-literary curriculum, but was later demoted by the v0.6.2 bench.

## What's new in v0.5.2

The v0.5.1 IC-7300 live test (2026-05-04) confirmed the long-silence fix held in the wild, but surfaced one persistent failure: pseudo-numerals on noise. The regular `--confidence-threshold 0.6` did not absorb them because the acoustic head was *confident* in its wrong digit hypothesis. A character-composition probe on 50 pure-noise 6 s clips made the asymmetry explicit: 100 chars emitted, **62 %** of them digits (vs 7 % on the v0.5.0 acoustic).

- **Class-conditional digit threshold.** `RnntModel.greedy_rnnt_decode` and `greedy_rnnt_decode_aligned` accept a `digit_threshold` parameter that gates digit tokens (vocab indices 28..37 = 0-9) at a stricter level than the regular `confidence_threshold`. The streaming decoder forwards this through `StreamingConfig.digit_threshold`. New CLI flag `--digit-threshold` on `decode_live` (default `0.90`), `decode_audio` (no default), and `eval_false_positive` (no default). Pass `0.0` to disable.
- One retraining attempt (Phase 5.6 with a pseudo-Morse empty-sample mode at `empty_sample_probability=0.40`) made the digit-on-noise failure *worse* (62 % → 94 %). Kept in the codebase for reference (commit fa2b461) but not the v0.5.2 path.
- LM fusion (`greedy_rnnt_decode_with_lm`) does not yet propagate `digit_threshold`. Fusion + digit gating is a small plumbing pass for a follow-up.

### Caveats

- The cost of `digit_threshold=0.90` on the v0.5.1 acoustic: −1 word out of 30 on the `bench_word_gap` 6× inter-word-silence bench (27 → 26), and −3 % relative legitimate digit recall on a 10-phrase synthetic ham-radio set (RST 599, callsigns, QTH).
- The threshold is a fixed scalar — context-aware gating (allow digits if recently inside an RST / callsign emission) is a future enhancement.
- ILME / density-ratio fusion remains catastrophic; do **not** pass `--ilm-weight > 0`.

## What's new in v0.5.1

The v0.5.0 live test surfaced one remaining failure mode: when several seconds of silence sat between two words, the model occasionally fused them or emitted spurious characters during the silence. Diagnosis: the synthetic training distribution had only ever shown the canonical 7-dit Farnsworth gap (~0.42 s at 20 WPM), so longer silences were out-of-domain. The acoustic head, asked to decode something it had never seen, behaved as if every silence longer than 7 dits was the start of a new emission.

- **`rnnt_phase5_5.pt`** — new recommended acoustic. Bootstrap chain: `phase3_5/best → phase5_3/best → phase5_4/last → phase5_5/last`.
  - **Phase 5.5 (15 k steps)** introduces one new `OperatorConfig` knob: `word_gap_inflation`, a per-utterance multiplier on every Farnsworth inter-word gap, sampled from `U(1.0, 8.0)`. At 20 WPM this stretches the canonical 7-dit space up to ~56 dits (~3.4 s of silence between two words). All other knobs (jitter, dash:dot ratio, gap inflation, channel, text mix, empty-sample prior) are kept identical to Phase 5.3 / 5.4 — strict ablation.
- The targeted bench (`scripts/bench_word_gap.py`, 30 short word pairs at 25 WPM, SNR 20 dB):
  - At 6× inflation (~2 s of inter-word silence): word recall **8 / 30 → 27 / 30**, CER **17.4 % → 1.5 %**.
  - At 4× inflation: word recall **20 / 30 → 28 / 30**, CER **7.7 % → 0.8 %**.
  - At 1× / 2×: 30 / 30 word recall preserved, CER 0 %.
- Out-of-domain wins are preserved: Alice ebook2cw (n=120, score≥0.5, greedy) **18.82 % → 16.39 % CER**. The hand-keyed `test_manu.wav` (Wordsworth, threshold 0.6) drops from **26.5 % → 23.5 % CER** (-11 % relative).
- False positives stay at zero in `decode_live` mode (threshold 0.6: 0.00 → 0.01 chars/sample). Without the threshold, Phase 5.5 emits ~1.97 chars per silence sample vs 0.28 for v0.5.0 — the v0.4.1+ confidence gate absorbs that entirely.

### Caveats

- The single regression is on synthetic French at SNR = −5 dB (`overall CER 0.68 % → 1.12 %`, with the deficit confined to the −5 dB row: 4.08 % → 6.71 %). Clean-to-0-dB FR remains 0 % CER and É / À / ' precision is preserved at 100 / 100 / 100 % (recall 100 / 100 / 98.3 %).
- The synthetic validation set saturates at ~0 % CER for the 5.5 acoustic (a known dormant `ValidationConfig.matching()` bug since Phase 4.0b — operator ranges are not propagated to the val set). The real signal is `bench_word_gap.py`, the Alice prose bench, and the live `test_manu` audio. `best_rnnt.pt` in `checkpoints/phase5_5/` is therefore frozen at step 1000 by the saturated val signal — **`last.pt` is the released checkpoint**.
- ILME / density-ratio fusion is still catastrophic on this distribution; do **not** pass `--ilm-weight > 0`.

## What's new in v0.5.0

The motivating live test (2026-05-02) was a 7-minute hand-keyed `test.wav` with a mix of contest exchanges, random-character blocks, FAV22 clair extracts, and the *Daffodils* poem. v0.4.1 produced unreadable letter-soup on the keyed sections (`F = ..-.` decoded as `A + V`, repeated emissions like `WWWVVUT TEST` for `IK3VUT TEST`). Diagnosis: real human keying has dot/dash ratios and inter-element gaps outside the synthetic curriculum's envelope, and confidence-threshold + LM-fusion at inference time cannot compensate (the acoustic head is *confident* on its wrong interpretations).

- **`rnnt_phase5_4.pt`** — new recommended acoustic. Bootstrap chain: `phase3_5/best → phase5_3/best → phase5_4/last`.
  - **Phase 5.3 (15 k steps)** widens the synthetic operator envelope: element-jitter `(0, 0.30)` (was `(0, 0.15)`), gap-jitter `(0, 0.50)` (was `(0, 0.25)`), and two new `OperatorConfig` knobs sampled per utterance — `dash_dot_ratio ∈ U(2.5, 4.5)` (was a fixed `3.0`) and `gap_inflation ∈ U(0.8, 1.6)` (multiplicative on inter-element gaps, was identity). The synthesis stack is otherwise identical to Phase 3.5.
  - **Phase 5.4 (12 k steps)** mixes 30 % of every batch from a real-audio JSONL of 42 ground-truth-aligned 6 s chunks of human-keyed CW (the `test.wav` corpus, aligned via `scripts/align_ebook_cw.py` against a hand-written transcript). The synthetic 70 % keeps the Phase 5.3 wider-envelope curriculum so the model does not regress.
- The **wider-jitter alone (Phase 5.3) was insufficient**: byte-for-byte identical decode to v0.4.1 on the keyed audio. The real-audio prior in Phase 5.4 was the unblock.
- The **improvement is not localised**: out-of-domain Alice ebook2cw prose (n=120, score≥0.7) drops from **36.44 % CER → 19.16 %** (greedy) and **32.30 % → 16.59 %** (with fusion λ=0.7). Synthetic FR-mix samples drop from **50.0 % → 12.5 %**. FP bench (threshold 0.6 + fusion 0.7): **0 / 150** (was 0.17 mean in v0.4.1).
- **`scripts/train_rnnt.py` learns `--curriculum phase5_3`**, `--real-audio-jsonl`, `--real-audio-probability`, and `--real-audio-score-threshold` for reproducing v0.5.0.
- LM fusion is still **offline only**; streaming-decoder integration is parked for a follow-up.

### Caveats

- The 42 real-audio chunks are from a single recording session; that audio is in-distribution for v0.5.0. Out-of-domain validation lives in the Alice and synthetic FR benches above.
- ILME / density-ratio fusion is still catastrophic on this distribution; do **not** pass `--ilm-weight > 0`.

## What's new in v0.4.1

- **`lm_phase5_2.pt`** — new 4.76 M-param 49-vocab character-level LM, trained on `PHASE_3_4_MIX` (the same text mix the Phase 3.5 acoustic model saw: callsigns 10 %, Q-codes 12 %, QSO 20 %, numerics 12 %, English-words 4 %, random 18 %, multilingual prose 8 %, French-only prose 16 %). 20 k steps, val_ppl 5.626, EMA 0.999. **Replaces `lm_phase4_0.pt` as the recommended fusion LM** — the legacy v0.1 LM was trained on 100 % ham-radio mix (no prose) and is hostile to general prose decoding (PPL 9 on QSO vs 100-200 on Alice). Pair with the Phase 3.5 RNN-T at λ = 0.7 (sweep-tuned).
- **Streaming false-positive gate**: `decode_live.py --confidence-threshold` default is now `0.6`. On the 150-sample 3-mode noise bench: 1.50 → 0.17 mean characters per noise sample (−90 %). No measurable accuracy loss down to −5 dB SNR. Pass `--confidence-threshold 0.0` to recover v0.4.0 behaviour.
- **`scripts/decode_audio.py` learns LM fusion**: new `--lm-ckpt`, `--fusion-weight`, `--confidence-threshold` flags. Confidence gating is applied to the *acoustic* softmax pre-LM so threshold and fusion stack cleanly (the LM cannot rescue noise-driven low-confidence emissions).
- **Autonomous prose validation pipeline**: `data/real/aligned/all_alice.jsonl` (9482 ground-truthed ebook2cw chunks from Alice in Wonderland chapters 1–12) is now the project's prose bench, paired with `scripts/eval_fusion_realaudio.py`. The v0.4.1 fusion verdict was reached without a live IC-7300 test.
- **`scripts/train_lm.py` learns text mixes**: new `--mix` and `--vocab-size` flags so future LMs can be matched to whatever curriculum the acoustic model was trained on.

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
| `rnnt_phase5_5.pt` | 4.13 M | **49** | **v0.6.2 / v0.6.3 recommended acoustic** — Phase 5.5 long inter-word-silence curriculum on top of the Phase 5.4 real-audio mix. Re-promoted after the LCWO + websdr bench beat Phase 5.7 / 5.8 / 5.9 descendants. Pair with `--confidence-threshold 0.6` and `--digit-threshold 0.90` for the shipped live behaviour. | ✅ |
| `lm_phase5_2.pt`   | 4.76 M | **49** | v0.4.1 LM — matched to the Phase 3.5 text mix (multilingual prose + ham radio). val_ppl 5.626. **Use this for shallow fusion at λ = 0.7.** | ✅ |
| `rnnt_phase5_9.pt` | 4.13 M | 49 | Phase 5.9 failed retrain — strict letter-group densification. Regressed on 6 / 6 LCWO + websdr clips, including its target websdr clip. Kept for reproducibility. | |
| `rnnt_phase5_8.pt` | 4.13 M | 49 | v0.6.0 / v0.6.1 acoustic — English-literary curriculum. Demoted at v0.6.2 after the real-audio bench lost ~24 % relative mean CER vs `rnnt_phase5_5`. | |
| `rnnt_phase5_7.pt` | 4.13 M | 49 | v0.5.3 acoustic — amateur-idiom curriculum (`5NN` cut-numbers + run-on `UR/SK/KN/BK`). Kept for diff. | |
| `rnnt_phase5_4.pt` | 4.13 M | 49 | v0.5.0 acoustic — Phase 5.3 (wider jitter + dash:dot ratio) + Phase 5.4 (30 % real-audio mix). Kept for diff. | |
| `rnnt_phase3_5.pt` | 4.13 M | 49 | v0.4.0 / v0.4.1 acoustic — synthetic-only. Kept for diff. | |
| `lm_phase4_0.pt`   | 4.76 M | 46 | Legacy v0.1-era LM, 100 % ham-radio text mix. Kept for research / reproducibility; **not recommended for fusion**. | |
| `rnnt_phase3_3.pt` | 4.13 M | 46 | v0.3 acoustic model — multilingual ASCII-normalised prose. No accent tokens. | |
| `rnnt_phase3_2.pt` | 4.13 M | 46 | v0.2 acoustic model — anti-hallucination curriculum, no multilingual data. | |
| `rnnt_phase3_0.pt` | 4.13 M | 46 | v0.1 acoustic model — AWGN-only training. | |

## Intended use

- **Primary**: decoding Morse-code audio (8 kHz sample rate, carrier around 600 Hz, 500 Hz RX bandwidth) into text. Targeted at amateur-radio receivers and SDR applications. v0.4 has been live-validated on a real IC-7300 on a mixed-language stream (CQ macros + Verlaine + Poe + Nerval + FAV22 clair) and decodes French prose with diacritics (e.g. `L'HEURE`, `MA SEULE ÉTOILE EST MORTE`, `À LA TOUR ABOLIE`) where v0.3 collapsed apostrophes to `1` and stripped É / À to `E` / `A`.
- **Secondary (research)**: a reference implementation for RNN-T and LM-fusion research on a small-vocabulary acoustic task with a fully reproducible synthetic data pipeline.

## Limitations

Honest about what v0.6.3 does and does not do:

1. **Real-audio coverage is still narrow.** The recommended acoustic includes a small real-audio mix from one hand-keyed recording session, but it is not trained on a broad, diverse amateur-radio corpus. More operators, receivers, bands, speeds, and propagation conditions are the main data gap.
2. **Only three accented characters are supported (É, À, apostrophe).** The 49-token vocabulary covers the French tokens already added in Phase 3.4; other diacritics still fall back to their ASCII base letter via NFKD (e.g. `château` → `CHATEAU`).
3. **6-second model context.** Training clips are 6 s. Offline and live decoders keep inference on that distribution with sliding 6 s windows and central-zone commits.
4. **WPM range.** The release models target normal CW speeds around 16-28 WPM. Very high-speed HST / FAV22 audio (84-240 WPM) is out-of-distribution.
5. **Pile-ups and overlapping stations are not solved.** Multi-source CW at the same or nearby frequency is a source-separation problem, not just an acoustic-decoding problem.
6. **LM fusion is offline only.** `morseformer decode --preset prose` can use `lm_phase5_2`; `morseformer live` runs acoustic-only even when a preset names an LM.
7. **Beam search / callsign priors are experimental.** Scaffolding exists in the codebase, but the shipped presets use the validated greedy RNN-T path.

## How to use

### Offline decode of a `.wav` file

```bash
pip install morseformer
morseformer decode my_recording.wav

# Optional LM shallow fusion for prose / ragchew recordings.
morseformer decode my_recording.wav --preset prose

# Inspect or pre-download model artifacts.
morseformer models list
morseformer models download rnnt_phase5_5
```

### Real-time streaming decode

```bash
pip install "morseformer[live]"
morseformer live
```

Tune your receiver to zero-beat at 600 Hz with a roughly 500 Hz CW filter. `Ctrl+C` to quit. Latency is about 4 s end-to-end. The shipped live preset uses `rnnt_phase5_5.pt`, `--confidence-threshold 0.6`, and `--digit-threshold 0.90` to suppress noise-driven false positives and pseudo-numerals.

## Training data

Most training audio is **synthetic**, generated on the fly by the `morseformer.data.synthetic` package. The current recommended acoustic (`rnnt_phase5_5.pt`) descends from:

- **Phase 3.4 / 3.5 text mix**: callsigns, Q-codes, QSO templates, numerics, English words, random characters, multilingual prose, and French prose with É / À / apostrophe preserved.
- **Realistic HF channel**: AWGN, QSB, QRN, carrier-frequency jitter, drift, QRM, and a 500 Hz receiver bandpass around 600 Hz.
- **Human-keying envelope**: widened element and gap jitter, dash:dot ratio variation, inter-element gap inflation, and long inter-word-silence inflation.
- **Phase 5.4 real-audio mix**: a small set of hand-keyed, ground-truth-aligned 6 s chunks mixed into training.
- **Phase 5.5 long word gaps**: `word_gap_inflation ∈ U(1.0, 8.0)`, which is the acoustic checkpoint now recommended again in v0.6.3.

The French prose corpus comes from Project Gutenberg (`data/corpus/prose.txt`, gitignored, reproducible via `python data/corpus/fetch.py`). The FAV22 reference clair blocks (`data/corpus/fav22_blocks.jsonl`) extracted from the F9TM training PDF supply ~110 k chars of authentic French CW text (3.62 % accent density), used by Phase 3.5+ as additional `prose_fr` source.

The real-audio component is useful but narrow; expanding it is the highest-value data improvement for future releases.

## Evaluation

### French accent-rich bench (historical v0.4 headline)

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

### Realistic-channel SNR ladder (historical v0.3 reference numbers retained)

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

### AWGN guard ladder (historical no-channel regression check)

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

### Historical v0.4.1 false-positive grid (Phase 3.5 acoustic, threshold × fusion)

`scripts/eval_false_positive.py --candidate-ckpt rnnt_phase3_5 --lm-ckpt lm_phase5_2 --fusion-weight λ_lm --confidence-threshold thr`, 50 samples / mode × 3 modes:

| `confidence_threshold` | `fusion_weight` (λ_lm) | mean chars / noise | max | % > 5 chars |
|---|---|---|---|---|
| 0.0 | 0.0 | 1.50 | 2 | 0.0 % |
| 0.0 | 0.7 | 1.01 | 2 | 0.0 % |
| **0.6** | 0.0 | **0.17** | 1 | 0.0 % |
| **0.6** | **0.7** | **0.17** | 1 | 0.0 % |

Threshold 0.6 (the v0.4.1 streaming default) cuts the FP mean by ≈ 90 %. Adding fusion λ = 0.7 on top **does not regress FP** because gating is applied to the acoustic head pre-LM — the LM cannot rescue a low-confidence noise emission. Threshold and fusion stack cleanly.

### Historical v0.4.1 LM rescoring on prose audio (Alice in Wonderland CW, ground-truth aligned)

`scripts/eval_fusion_realaudio.py --rnnt-ckpt phase3_5 --lm-ckpt lm_phase5_2 --jsonl all_alice.jsonl`, n = 120, score-filter ≥ 0.7 to focus on chunks where the baseline still has CER headroom:

| λ_lm (shallow fusion) | CER | Δ vs baseline |
|---|---|---|
| 0.0 (baseline, greedy) | 36.44 % | — |
| 0.3 | 35.81 % | −1.7 % |
| 0.5 | 33.46 % | −8.2 % |
| **0.7** | **32.30 %** | **−11.4 %** ← peak |
| 1.0 | 33.66 % | −7.6 % |

The Alice corpus is ebook2cw renderings of public-domain English prose (chapters 1–12 of *Alice in Wonderland*), char-aligned by `scripts/align_ebook_cw.py` against the source text — a corpus that was prepared for Phase 3.7 real-audio finetuning and re-purposed here as v0.4.1's autonomous prose bench. The 9482 chunks are deterministic ebook2cw audio, not real-radio captures, but they exercise the model on real linguistic structure (story narrative) where the v0.4.0 acoustic still has headroom and the LM can contribute.

ILME (density-ratio fusion, λ_ilm > 0) is **catastrophically harmful** on this distribution: λ_lm = λ_ilm = 0.2 → CER +18 %, λ_lm = λ_ilm = 0.3 → CER +83 %. The fusion code keeps ILME for research / completeness; it is *not* the recommended path.

### Why the legacy LM (`lm_phase4_0`) does not help

PPL of `lm_phase4_0.pt` (the v0.1 LM, trained on `DEFAULT_MIX` = 100 % ham radio) on representative texts:

| text | PPL |
|---|---|
| Alice prose ("ALICE WAS BEGINNING TO GET VERY TIRED OF SITTING …") | 99.5 |
| Alice prose ("DOWN THE RABBIT HOLE INTO A LARGE RABBIT HOLE") | 213.9 |
| QSO line ("CQ CQ CQ DE F4ABC F4ABC PSE K") | 8.9 |
| RST exchange ("UR RST 599 599 NAME TOM TOM 73") | 15.4 |

The legacy LM is calibrated for ham radio and is *hostile* to general prose decoding — fusion with it on Alice audio actively hurts CER (33.86 % → 35.79 % at λ_lm = 0.2 in our re-bench), confirming the Phase 4.1 null-result was a text-distribution mismatch and not a fusion-method problem. `lm_phase5_2.pt` was trained on the same `PHASE_3_4_MIX` the Phase 3.5 acoustic saw, which makes shallow fusion finally useful.

### Live validation

Selected live-radio tests on a real IC-7300 driving the streaming decoder:

- **2026-04-27** (v0.3 baseline): mixed CQ macros + Verlaine *Chanson d'automne* + Poe *Annabel Lee* + Apollinaire *Le Pont Mirabeau*. French rendered word-by-word but apostrophes appeared as `1` and É / À as plain `E` / `A`.
- **2026-04-29** (v0.4): same setup with FAV22 clair + Nerval *El Desdichado* + Poe *Annabel Lee*. French accents are now decoded natively: `L'HEURE`, `L'AUTOMNE`, `MA SEULE ÉTOILE EST MORTE`, `À LA TOUR ABOLIE`. Apostrophes captured cleanly. No false-positive É / À emissions in the English passages.

Residual failure modes observed across later releases include hard callsigns under noise and jitter, occasional prosign over-emission on pile-ups, and overlapping stations at the same frequency.

## Environmental impact

Training budget through v0.6.x is small by modern ASR standards: single-GPU fine-tunes on an RTX 3060 Laptop, typically a few hours per phase. The current recommended acoustic is still a compact 4.13 M-param model intended for CPU inference.

## Technical specifications

**Acoustic model** (`rnnt_phase5_5.pt`, 4.13 M params):
- Encoder: Conformer d=144, L=8, H=4, ff_expansion=4, conv_kernel=31, RoPE, LayerNorm conv, 4× subsample
- CTC head: Linear(144 → 49)
- PredictionNetwork: Embedding(49, 128) + LSTM(128, 128, 1 layer)
- JointNetwork: Linear(144 → 256) + Linear(128 → 256) + tanh + Linear(256 → 49)

**Language model** (`lm_phase5_2.pt`, 4.76 M params): decoder-only GPT (RMSNorm + SwiGLU + RoPE + tied embeddings + causal SDPA), d=256, L=6, H=4, dropout=0.1, 49-vocab. Trained 20 k steps on `PHASE_3_4_MIX`, AdamW peak_lr 3e-4, batch 128, bf16, EMA 0.999, val_ppl 5.626. Pair with the RNN-T at λ = 0.7 for offline prose decoding. The legacy `lm_phase4_0.pt` (46-vocab, 100 % ham-radio mix, val_ppl 3.75 on its own distribution) is kept for research but is not the recommended fusion partner.

**Vocabulary**: **49 tokens** — blank (index 0), 26 uppercase letters A–Z, 10 digits 0–9, 9 punctuation / Morse prosigns (`.`, `,`, `?`, `!`, `/`, `-`, `=`, `+`, space), plus `É`, `À`, `'`.

**Front-end**: complex bandpass around the carrier (default 600 Hz ± 250 Hz), magnitude, 4× downsample to a 500 Hz frame rate, scalar normalization. Input shape `[B, T, 1]`.

## Citation

```bibtex
@software{morseformer_v0_6_3_2026,
  author       = {Derhy, Serge},
  title        = {morseformer: open-source transformer-based Morse / CW decoder},
  year         = 2026,
  version      = {v0.6.3},
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
