"""Measure decode latency for a morseformer release checkpoint.

Reports three numbers:

* **Encoder ms / frame** — pure forward pass through the Conformer
  encoder, divided by the number of output frames. The lower bound on
  what any decoder built on this acoustic can deliver.
* **Greedy decode ms / window** — one full ``RnntModel.greedy_rnnt_decode``
  call on a single ``window_seconds`` audio buffer. The number a user
  paying attention to the "decode this WAV" path will feel.
* **Streaming feed ms / hop** — wall-clock latency between handing the
  decoder a fresh hop's worth of audio and getting the next committed
  fragment back. The number that gates "live" usability — should stay
  comfortably under ``hop_seconds * 1000``.

Run::

    python -m eval.bench_latency
    python -m eval.bench_latency --device cuda --trials 50
    python -m eval.bench_latency --ckpt release/rnnt_phase5_7.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from morse_synth import core as synth
from morse_synth.operator import OperatorConfig
from morseformer.cli.registry import RECOMMENDED_ACOUSTIC, resolve_model
from morseformer.decoding.streaming import StreamingConfig, StreamingDecoder
from morseformer.features.frontend import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel

_DEFAULT_WINDOW_S = 6.0
_DEFAULT_HOP_S = 2.0
_DEFAULT_SAMPLE_RATE = 8000


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for k, v in (ckpt.get("ema") or {}).items():
        if k in state:
            state[k] = v
    enc = ckpt["config"]["model"]["encoder"]
    vocab_size = ckpt["config"]["model"].get("vocab_size")
    extra = {"vocab_size": vocab_size} if vocab_size is not None else {}
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"], **extra,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=ckpt["config"]["model"]["d_pred"],
        pred_lstm_layers=ckpt["config"]["model"]["pred_lstm_layers"],
        d_joint=ckpt["config"]["model"]["d_joint"],
        dropout=ckpt["config"]["model"]["dropout"],
        **extra,
    )
    model = RnntModel(rnnt_cfg).to(device).eval()
    model.load_state_dict(state)
    return model


def _make_window_features(
    sample_rate: int, window_s: float, frame_rate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = synth.render(
        "CQ DE F4HYY K CQ DE F4HYY K",
        operator=OperatorConfig(wpm=22.0, seed=0),
        freq=600.0, sample_rate=sample_rate, amplitude=0.5,
    )
    target_n = int(round(window_s * sample_rate))
    if audio.size < target_n:
        audio = np.pad(audio, (0, target_n - audio.size))
    else:
        audio = audio[:target_n]
    features = extract_features(audio, sample_rate, FrontendConfig(frame_rate=frame_rate))
    f = torch.from_numpy(features).unsqueeze(0)            # [1, T, 1]
    n_frames = torch.tensor([f.shape[1]], dtype=torch.long)
    return f, n_frames


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _measure(fn, *, warmup: int, trials: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    times: list[float] = []
    for _ in range(trials):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times) * 1000.0
    return float(arr.mean()), float(arr.std())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, default=None,
                   help="checkpoint path. Default: resolve the recommended "
                        "acoustic via the model registry.")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--window-seconds", type=float, default=_DEFAULT_WINDOW_S)
    p.add_argument("--hop-seconds", type=float, default=_DEFAULT_HOP_S)
    p.add_argument("--sample-rate", type=int, default=_DEFAULT_SAMPLE_RATE)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args(argv)

    device = torch.device(args.device or _auto_device())
    ckpt_path = args.ckpt if args.ckpt is not None else resolve_model(RECOMMENDED_ACOUSTIC)
    print(f"[bench_latency] device={device} ckpt={ckpt_path}")
    model = _load_rnnt(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bench_latency] params={n_params:,}")

    features, lengths = _make_window_features(
        args.sample_rate, args.window_seconds, frame_rate=500,
    )
    features = features.to(device)
    lengths = lengths.to(device)
    n_frames = int(lengths.item())
    print(f"[bench_latency] window={args.window_seconds}s → {n_frames} frames")

    # 1. Encoder forward.
    def _encoder_only() -> None:
        with torch.no_grad():
            model.acoustic(features, lengths)

    enc_mean, enc_std = _measure(
        _encoder_only, warmup=args.warmup, trials=args.trials, device=device,
    )

    # 2. Greedy RNN-T decode (encoder + pred + joint).
    def _greedy_decode() -> None:
        with torch.no_grad():
            model.greedy_rnnt_decode(features, lengths)

    dec_mean, dec_std = _measure(
        _greedy_decode, warmup=args.warmup, trials=args.trials, device=device,
    )

    # 3. Streaming feed latency per hop.
    sd_cfg = StreamingConfig(
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        sample_rate=args.sample_rate,
    )
    sd = StreamingDecoder(model, sd_cfg, device=device)
    hop_samples = sd._hop_samples
    rng = np.random.default_rng(0)
    # Prime so that the next feed triggers a decode.
    sd.feed(rng.standard_normal(sd._window_samples).astype(np.float32))

    def _stream_hop() -> None:
        sd.feed(rng.standard_normal(hop_samples).astype(np.float32))

    stream_mean, stream_std = _measure(
        _stream_hop, warmup=args.warmup, trials=args.trials, device=device,
    )

    hop_budget_ms = args.hop_seconds * 1000.0
    rtf = stream_mean / hop_budget_ms

    print()
    print(f"  {'stage':<28} | {'mean (ms)':>10} | {'±std':>8} | notes")
    print(f"  {'-'*28}-+-{'-'*10}-+-{'-'*8}-+------------------------------")
    print(f"  {'encoder forward (window)':<28} | {enc_mean:>10.2f} | "
          f"{enc_std:>8.2f} | ≈ {enc_mean / max(n_frames, 1):.3f} ms/frame")
    print(f"  {'greedy RNN-T decode':<28} | {dec_mean:>10.2f} | "
          f"{dec_std:>8.2f} | encoder + pred + joint, 1 window")
    print(f"  {'streaming feed (1 hop)':<28} | {stream_mean:>10.2f} | "
          f"{stream_std:>8.2f} | RTF = {rtf:.3f} (< 1.0 = real-time)")
    print()
    print(f"[bench_latency] hop budget = {hop_budget_ms:.0f} ms; "
          f"real-time {'OK' if rtf < 1.0 else 'OVER BUDGET'} on {device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
