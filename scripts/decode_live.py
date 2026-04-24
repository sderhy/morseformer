"""Real-time Morse decoder — v0, non-overlapping 6 s chunks.

Captures audio continuously from the default PulseAudio input device,
decodes every ``--chunk-seconds`` window through the loaded RNN-T
checkpoint, and prints the hypothesis to stdout with a timestamp.

This is a deliberately simple first pass — the 6 s window matches the
training-clip length, which is the only length at which the model is
known to behave well, and non-overlapping chunks avoid the
deduplication problem entirely. Latency is therefore equal to
``--chunk-seconds`` (default 6 s). A sliding-window variant with
deduplication is planned as v1.

Usage::

    python -m scripts.decode_live
    python -m scripts.decode_live --ckpt release/rnnt_phase3_0.pt
    python -m scripts.decode_live --carrier 700 --capture-rate 16000

Setup: tune the receiver so the CW tone lands at ``--carrier`` Hz
(default 600, matching training). The receiver's CW filter should be
≈ 500 Hz wide, also matching training. Run ``pavucontrol`` if you need
to pick a specific PulseAudio source — or pass ``--device`` with the
source name.

``Ctrl+C`` to quit cleanly.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import ctc_greedy_decode, decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


def _candidate_paths() -> tuple[Path, ...]:
    return (
        Path("release/rnnt_phase3_0.pt"),
        Path("checkpoints/phase3_0/best_rnnt.pt"),
    )


def _resolve_ckpt(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise SystemExit(f"[decode_live] --ckpt {explicit} not found.")
        return explicit
    for p in _candidate_paths():
        if p.exists():
            return p
    raise SystemExit(
        "[decode_live] no checkpoint found. Tried:\n  - "
        + "\n  - ".join(str(p) for p in _candidate_paths())
        + "\nDownload the release weights with:\n"
        "  pip install huggingface_hub\n"
        "  hf download sderhy/morseformer rnnt_phase3_0.pt "
        "--local-dir checkpoints/phase3_0"
    )


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=cfg["model"]["d_pred"],
        pred_lstm_layers=cfg["model"]["pred_lstm_layers"],
        d_joint=cfg["model"]["d_joint"],
        dropout=cfg["model"]["dropout"],
    )
    model = RnntModel(rnnt_cfg).to(device)
    state = dict(ckpt["model"])
    ema = ckpt.get("ema")
    if ema:
        for k, v in ema.items():
            if k in state:
                state[k] = v
    model.load_state_dict(state)
    model.eval()
    return model


def _open_recorder(capture_rate: int, device_name: str | None):
    """Open a PulseAudio record stream at ``capture_rate`` Hz, mono,
    S16LE. Return the :class:`PaSimple` object. Raises ``SystemExit``
    with a friendly message if ``pasimple`` is not installed."""
    try:
        import pasimple
    except ImportError as e:
        raise SystemExit(
            "[decode_live] pasimple is not installed. "
            "Install it with: pip install -e '.[audio]'"
        ) from e
    return pasimple.PaSimple(
        direction=pasimple.PA_STREAM_RECORD,
        format=pasimple.PA_SAMPLE_S16LE,
        channels=1,
        rate=capture_rate,
        app_name="morseformer",
        stream_name="live-decode",
        device_name=device_name,
    )


def _s16_to_float32(buf: bytes) -> np.ndarray:
    """Convert a S16LE byte buffer into a normalised float32 array in
    ``[-1, 1]``."""
    arr = np.frombuffer(buf, dtype="<i2").astype(np.float32)
    arr /= 32768.0
    return arr


def _resample_if_needed(
    audio: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(
        np.float32, copy=False
    )


def _estimate_snr_db(audio: np.ndarray, carrier_hz: float, rate: int) -> float:
    """Cheap SNR estimate: the power in a narrow band around
    ``carrier_hz`` vs the power in the rest of the spectrum. Rough
    sanity signal for the user — not a calibrated measurement."""
    from numpy.fft import rfft, rfftfreq
    # Hann window to avoid skirt leakage.
    win = np.hanning(audio.size).astype(np.float32)
    spec = np.abs(rfft(audio * win)) ** 2
    freqs = rfftfreq(audio.size, 1.0 / rate)
    band = (freqs > carrier_hz - 100) & (freqs < carrier_hz + 100)
    signal_p = spec[band].sum() + 1e-12
    noise_p = spec[~band].sum() + 1e-12
    # Narrow band is ~200 Hz out of ~4000 Hz (rate / 2), so normalise by
    # the fractional bandwidth so that "white noise only" ≈ 0 dB.
    scale = (spec.size - band.sum()) / max(band.sum(), 1)
    return 10.0 * np.log10((signal_p * scale) / noise_p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="path to an RNN-T checkpoint; auto-detects "
                        "release/ or checkpoints/phase3_0/.")
    p.add_argument("--chunk-seconds", type=float, default=6.0,
                   help="window length to decode. 6.0 matches training.")
    p.add_argument("--carrier", type=float, default=600.0,
                   help="CW tone frequency in Hz the model expects; "
                        "tune your RX to zero-beat at this frequency.")
    p.add_argument("--bandwidth", type=float, default=200.0,
                   help="front-end complex BPF half-width.")
    p.add_argument("--model-rate", type=int, default=8000,
                   help="rate the model sees after resampling.")
    p.add_argument("--capture-rate", type=int, default=16000,
                   help="rate we ask PulseAudio to capture at. Audio "
                        "is resampled to --model-rate before decoding. "
                        "Try 48000 if 16000 is rejected.")
    p.add_argument("--frame-rate", type=int, default=500,
                   help="front-end output frame rate.")
    p.add_argument("--device", default=None,
                   help="cpu / cuda for inference (default: auto)")
    p.add_argument("--audio-device", default=None,
                   help="PulseAudio source name (default: system default)")
    p.add_argument("--use-ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")
    return p


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    ckpt_path = _resolve_ckpt(args.ckpt)
    model = _load_rnnt(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())

    capture_rate = args.capture_rate
    chunk_seconds = args.chunk_seconds
    chunk_bytes = int(round(chunk_seconds * capture_rate)) * 2  # S16 = 2 B / sample

    fcfg = FrontendConfig(
        tone_freq=args.carrier,
        bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )

    print(f"[decode_live] device={device}  params={n_params:,}")
    print(f"[decode_live] ckpt={ckpt_path}")
    print(f"[decode_live] audio: capture={capture_rate} Hz → model={args.model_rate} Hz  "
          f"carrier={args.carrier:.0f} Hz  chunk={chunk_seconds:.1f} s")
    print(f"[decode_live] tune your RX to zero-beat at {args.carrier:.0f} Hz "
          f"with a ≈ 500 Hz CW filter. Ctrl+C to quit.")
    print()

    # Clean Ctrl-C: signal handler just sets a flag so the current
    # decode finishes before we tear down.
    should_stop = {"flag": False}

    def _on_sigint(_signum, _frame):  # noqa: ANN001
        should_stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    recorder = _open_recorder(capture_rate, args.audio_device)
    try:
        chunk_idx = 0
        while not should_stop["flag"]:
            # Block until ``chunk_bytes`` of audio have been captured.
            buf = recorder.read(chunk_bytes)
            audio = _s16_to_float32(buf)
            audio = _resample_if_needed(audio, capture_rate, args.model_rate)

            snr = _estimate_snr_db(audio, args.carrier, args.model_rate)
            feats = extract_features(audio, args.model_rate, fcfg)
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            lengths = torch.tensor(
                [feats.shape[0]], dtype=torch.long, device=device
            )

            with torch.no_grad():
                enc_out, enc_lengths = model.acoustic.encode(x, lengths)
                ctc_argmax = model.acoustic.head(enc_out).argmax(-1)[0].cpu().tolist()
                ctc_len = int(enc_lengths[0].item())
                ctc_hyp = ctc_greedy_decode(ctc_argmax[:ctc_len])
                rnnt_tokens = model.greedy_rnnt_decode(x, lengths)[0]
                rnnt_hyp = decode(rnnt_tokens)

            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] chunk {chunk_idx:>3d}  SNR~{snr:+5.1f} dB")
            print(f"    CTC  : {ctc_hyp!r}")
            print(f"    RNN-T: {rnnt_hyp!r}")
            print()
            chunk_idx += 1
    finally:
        recorder.close()
        print("[decode_live] stream closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
