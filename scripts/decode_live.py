"""Real-time Morse decoder — v1, sliding 6 s window with central-zone commit.

Captures audio continuously from the default PulseAudio input device and
decodes it through a sliding 6 s window that advances every
``--hop-seconds`` (default 2 s). For each window we commit only the
emissions whose absolute timestamps fall in the *central* zone of the
window — the part that won't be revisited by the next slide. Adjacent
central zones tile the audio without gap or overlap, so the decoder
never re-emits a token and never sees a chunk boundary in the committed
text. This kills both first-character stutter (``CCCCQ``) and
word-boundary cuts (``F4HY|Y``) that v0 surfaced on real QSO audio.

Decoder logic lives in :mod:`morseformer.decoding.streaming`; this
script is a thin PulseAudio + stdout wrapper.

Latency: ``window/2 + hop/2`` worst-case (4 s with the defaults), down
from 6 s in v0.

Usage::

    python -m scripts.decode_live
    python -m scripts.decode_live --ckpt checkpoints/phase3_1/best_rnnt.pt
    python -m scripts.decode_live --carrier 700 --hop-seconds 1.5

Setup: tune the receiver so the CW tone lands at ``--carrier`` Hz (default
600, matching training). The receiver's CW filter should be ≈ 500 Hz wide.
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

from morseformer.decoding.streaming import StreamingConfig, StreamingDecoder
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


def _candidate_paths() -> tuple[Path, ...]:
    return (
        Path("release/rnnt_phase3_2.pt"),
        Path("checkpoints/phase3_2/last.pt"),
        Path("checkpoints/phase3_2/best_rnnt.pt"),
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
        + "\nDownload the v0.2 release weights with:\n"
        "  pip install huggingface_hub\n"
        "  hf download sderhy/morseformer rnnt_phase3_2.pt "
        "--local-dir release"
    )


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
    # Honour the checkpoint's recorded vocab so legacy 46-vocab
    # checkpoints (Phase 3.0–3.3) keep loading cleanly after the
    # tokenizer was extended to 49 in Phase 3.4. The decoded token
    # indices stay valid against the current 49-vocab tokenizer
    # because indices 0..45 mean the same thing in both layouts.
    ckpt_vocab = cfg["model"].get("vocab_size")
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
        **({"vocab_size": ckpt_vocab} if ckpt_vocab is not None else {}),
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=cfg["model"]["d_pred"],
        pred_lstm_layers=cfg["model"]["pred_lstm_layers"],
        d_joint=cfg["model"]["d_joint"],
        dropout=cfg["model"]["dropout"],
        **({"vocab_size": ckpt_vocab} if ckpt_vocab is not None else {}),
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="path to an RNN-T checkpoint; auto-detects "
                        "release/rnnt_phase3_2.pt, then checkpoints/phase3_2/, "
                        "with v0.1 fallbacks.")
    p.add_argument("--window-seconds", type=float, default=6.0,
                   help="decoder window length. Must equal training "
                        "clip length (6.0 s) — model is not robust to "
                        "other lengths.")
    p.add_argument("--hop-seconds", type=float, default=2.0,
                   help="how often we re-decode. Smaller hop → lower "
                        "latency. Default 2.0 s.")
    p.add_argument("--carrier", type=float, default=600.0,
                   help="CW tone frequency in Hz the model expects.")
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
    p.add_argument("--read-chunk-seconds", type=float, default=0.25,
                   help="how often we pull from the audio device. "
                        "Smaller = smoother but more syscalls.")
    p.add_argument("--confidence-threshold", type=float, default=0.0,
                   help="drop emissions with softmax probability below "
                        "this threshold. 0.0 disables. Try 0.3 - 0.5 on "
                        "Phase 3.0/3.1 to suppress letter-soup on weak "
                        "signal.")
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

    sd_cfg = StreamingConfig(
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        sample_rate=args.model_rate,
        frame_rate=args.frame_rate,
        carrier_hz=args.carrier,
        bandwidth_hz=args.bandwidth,
        confidence_threshold=args.confidence_threshold,
    )
    sd = StreamingDecoder(model, sd_cfg, device=device)

    capture_rate = args.capture_rate
    read_bytes = (
        int(round(args.read_chunk_seconds * capture_rate)) * 2  # S16 = 2 B/sample
    )

    print(f"[decode_live] device={device}  params={n_params:,}")
    print(f"[decode_live] ckpt={ckpt_path}")
    print(f"[decode_live] audio: capture={capture_rate} Hz → model={args.model_rate} Hz  "
          f"carrier={args.carrier:.0f} Hz")
    print(f"[decode_live] window={args.window_seconds:.1f}s  hop={args.hop_seconds:.1f}s  "
          f"latency≈{(args.window_seconds + args.hop_seconds) / 2:.1f}s")
    print(f"[decode_live] tune your RX to zero-beat at {args.carrier:.0f} Hz "
          f"with a ≈ 500 Hz CW filter. Ctrl+C to quit.")
    print()

    should_stop = {"flag": False}

    def _on_sigint(_signum, _frame):  # noqa: ANN001
        should_stop["flag"] = True
    signal.signal(signal.SIGINT, _on_sigint)

    recorder = _open_recorder(capture_rate, args.audio_device)
    started_at = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] listening…", flush=True)

    try:
        while not should_stop["flag"]:
            buf = recorder.read(read_bytes)
            audio = _s16_to_float32(buf)
            audio = _resample_if_needed(
                audio, capture_rate, args.model_rate
            )
            for fragment in sd.feed(audio):
                if fragment:
                    print(fragment, end="", flush=True)
        # Final flush on Ctrl+C — commit the trailing audio.
        tail = sd.flush()
        if tail:
            print(tail, end="", flush=True)
        print()  # newline after stream end
    finally:
        recorder.close()
        elapsed = time.time() - started_at
        print(f"[decode_live] stream closed after {elapsed:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
