# morse_synth

`morse_synth` generates CW/Morse signals as `numpy.float32` arrays. It is used
to produce clean clips or clips passed through a synthetic HF channel for
MorseFormer training, tests, and benchmarks.

The module exposes two API levels:

- `synthesize(...)`: convenience wrapper for clean CW audio.
- `render(...)`: full `text -> operator -> keying -> HF channel` pipeline.

## Installation

From the repository root:

```bash
uv pip install -e .
```

Audio outputs are mono `float32` arrays with values centered around `0.0`.
Use `scipy.io.wavfile.write` to write them as WAV files.

## Minimal Example

```python
from scipy.io import wavfile

from morse_synth.core import synthesize

sample_rate = 8000
audio = synthesize(
    "CQ CQ DE F4HYY K",
    wpm=20,
    freq=600.0,
    sample_rate=sample_rate,
)

wavfile.write("cq_clean.wav", sample_rate, audio)
```

`synthesize` uses clean `raised_cosine` keying and applies no channel noise or
propagation effects. It is the right entry point when you need a quick,
readable, stable CW signal.

## Full Pipeline

Use `render` to simulate an imperfect operator and a noisy HF channel:

```python
from scipy.io import wavfile

from morse_synth.channel import ChannelConfig
from morse_synth.core import render
from morse_synth.keying import KeyingConfig
from morse_synth.operator import OperatorConfig

sample_rate = 8000

audio = render(
    "CQ CQ DE F4HYY K",
    operator=OperatorConfig(
        wpm=22,
        element_jitter=0.08,
        gap_jitter=0.12,
        dash_dot_ratio=3.2,
        seed=123,
    ),
    keying=KeyingConfig(
        shape="raised_cosine",
        rise_ms=5.0,
    ),
    channel=ChannelConfig(
        snr_db=5.0,
        qrn_rate_per_sec=3.0,
        qsb_rate_hz=0.25,
        qsb_depth_db=12.0,
        rx_filter_bw=500.0,
        rx_filter_centre=600.0,
        seed=456,
    ),
    freq=600.0,
    sample_rate=sample_rate,
)

wavfile.write("cq_hf.wav", sample_rate, audio)
```

## Quick API Reference

### `synthesize`

```python
audio = synthesize(
    text,
    wpm=20.0,
    freq=600.0,
    sample_rate=8000,
    rise_ms=5.0,
    amplitude=0.5,
    tail_ms=50.0,
)
```

Main parameters:

- `text`: message to encode. Unknown characters are ignored.
- `wpm`: speed in words per minute, measured with the PARIS standard.
- `freq`: CW audio carrier frequency in Hz.
- `sample_rate`: output sample rate in Hz.
- `rise_ms`: edge rise/fall time in milliseconds.
- `amplitude`: signal peak amplitude, usually between `0.0` and `1.0`.
- `tail_ms`: silence appended after the last event.

Useful errors:

- `wpm <= 0` raises `ValueError`.
- a `sample_rate` too low for the requested speed raises `ValueError`.

### `render`

```python
audio = render(
    text,
    operator=OperatorConfig(...),
    keying=KeyingConfig(...),
    channel=ChannelConfig(...),
    freq=600.0,
    sample_rate=8000,
    amplitude=0.5,
    tail_ms=50.0,
)
```

`render` runs three stages:

1. `build_events`: converts text into `(is_on, duration_seconds)` events.
2. `render_events`: converts those events into clean audio.
3. `apply_channel`: applies HF channel effects.

Pass `ChannelConfig(snr_db=float("inf"))` or leave the default configuration
for a channel output with no added noise. Passing `channel=None` is also
accepted and uses the default configuration.

## Configuration Objects

### `OperatorConfig`

Controls how the Morse is sent:

- `wpm`: nominal speed.
- `element_jitter`: random dit/dah duration variation, in dit units.
- `gap_jitter`: random silence duration variation.
- `farnsworth_char_gap`: inter-character gap in dit units.
- `farnsworth_word_gap`: inter-word gap in dit units.
- `dash_dot_ratio`: dah length relative to a dit.
- `gap_inflation`: multiplier for intra-character silences.
- `word_gap_inflation`: multiplier for inter-word silences.
- `run_on_pairs`: character pairs that may be fused, for example
  `(("U", "R", 0.5), ("S", "K", 0.25))`.
- `seed`: seed for reproducible random draws.

### `KeyingConfig`

Controls keying edge shape:

- `shape`: `"rect"`, `"raised_cosine"`, or `"gauss"`.
- `rise_ms`: edge smoothing duration. Ignored with `"rect"`.
- `chirp_hz_per_unit`: linear frequency drift during each keydown, in Hz per
  dit unit.

### `ChannelConfig`

Controls the HF simulation:

- `snr_db`: target signal-to-noise ratio. `float("inf")` disables AWGN.
- `qrn_rate_per_sec`: mean QRN impulse rate per second.
- `qrn_amplitude_db`: impulse amplitude relative to the signal peak.
- `qrn_decay_ms`: QRN impulse decay constant.
- `qsb_rate_hz`: slow fading frequency.
- `qsb_depth_db`: fading depth.
- `carrier_drift_hz_per_s`: carrier random-walk strength.
- `rx_filter_bw`: receiver filter bandwidth in Hz, or `None` to disable it.
- `rx_filter_centre`: receiver filter center in Hz.
- `seed`: seed for reproducible channel effects.

## Reproducibility

Random draws are controlled separately:

```python
operator = OperatorConfig(wpm=20, element_jitter=0.1, seed=1)
channel = ChannelConfig(snr_db=0.0, qrn_rate_per_sec=5.0, seed=2)
```

Use fixed seeds when you need to regenerate the exact same clip. Use different
seeds to separate operator variation from channel variation.

## Stage-by-Stage Usage

You can inspect or reuse each pipeline stage:

```python
from morse_synth.channel import ChannelConfig, apply_channel
from morse_synth.keying import KeyingConfig, render_events
from morse_synth.operator import OperatorConfig, build_events

sample_rate = 8000
events = build_events("PARIS", OperatorConfig(wpm=18))
clean = render_events(events, keying=KeyingConfig(), sample_rate=sample_rate)
noisy = apply_channel(clean, sample_rate, ChannelConfig(snr_db=-5.0, seed=42))
```

This form is useful for testing timing, comparing several channels on the same
clean signal, or injecting your own event sequence.

## Practical Notes

- Text is uppercased internally and split into words with `split()`.
- Characters missing from the project's Morse table are ignored.
- Output is mono; no automatic 16-bit WAV normalization is performed.
- Effects are applied in this order: QSB, carrier drift, AWGN, QRN, RX filter.
- SNR is calibrated on keydown segments, not on full-clip RMS.
