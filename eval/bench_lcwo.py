"""Real-audio benchmark — runs morseformer through ``data/bench/manifest.jsonl``.

Usage::

    python -m eval.bench_lcwo
    python -m eval.bench_lcwo --models rnnt_phase5_7,rnnt_phase5_8
    python -m eval.bench_lcwo --presets live,prose

Reads the manifest, loads each acoustic checkpoint (resolved through the
CLI registry — local first, HF cache fallback), runs the streaming
offline decoder on every WAV, normalises both the hypothesis and the
ground-truth with :func:`morseformer.data.text._normalize_prose`, and
reports CER / WER per (model × preset × clip) plus a per-(model, preset)
mean.

Designed to be the reproducible source-of-truth that NEXT.md asks for
*before* any further acoustic retrain. Baseline decoders (cwget, fldigi)
will plug in via a similar ``Decoder`` callable; not wired in here yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import torch

from eval.benchmark import run as run_benchmark
from eval.datasets import Sample
from morseformer.cli.presets import PRESETS, get_preset
from morseformer.cli.registry import resolve_model
from morseformer.data.text import _normalize_prose
from morseformer.decoding.postprocess import format_output
from morseformer.decoding.streaming import StreamingConfig, decode_offline
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.lm import GptLM, LmConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenchEntry:
    id: str
    audio: Path
    gt: Path
    lang: str
    wpm: float | None
    source: str
    kind: str

    @classmethod
    def from_json(cls, obj: dict, repo_root: Path) -> "BenchEntry":
        return cls(
            id=obj["id"],
            audio=repo_root / obj["audio"],
            gt=repo_root / obj["gt"],
            lang=obj.get("lang", "en"),
            wpm=obj.get("wpm"),
            source=obj.get("source", ""),
            kind=obj.get("kind", "prose"),
        )


def load_manifest(path: Path, repo_root: Path) -> list[BenchEntry]:
    entries: list[BenchEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(BenchEntry.from_json(json.loads(line), repo_root))
    return entries


# --------------------------------------------------------------------------- #
# Audio + checkpoint loading
# --------------------------------------------------------------------------- #


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile

    sr, audio = wavfile.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 1.5:
        audio = audio / max_abs
    if sr != target_sr:
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        audio = resample_poly(
            audio, target_sr // g, sr // g
        ).astype(np.float32)
    return audio


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
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
    if ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    model.load_state_dict(state)
    model.eval()
    return model


def _load_lm(path: Path, device: torch.device) -> GptLM:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    lm = GptLM(LmConfig(
        vocab_size=mcfg["vocab_size"], d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"], n_layers=mcfg["n_layers"],
        dropout=mcfg["dropout"],
    )).to(device)
    state = dict(ckpt["model"])
    if ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    lm.load_state_dict(state)
    lm.eval()
    return lm


# --------------------------------------------------------------------------- #
# Decoder factory
# --------------------------------------------------------------------------- #


def make_morseformer_decoder(
    model: RnntModel,
    *,
    device: torch.device,
    confidence_threshold: float,
    digit_threshold: float,
    lm: GptLM | None = None,
    fusion_weight: float = 0.0,
    sample_rate: int = 8000,
    carrier_hz: float = 600.0,
    beam_width: int = 1,
    beam_emit_bonus: float = 0.0,
    callsign_prior_weight: float = 0.0,
):
    def _decode(audio: np.ndarray, sr: int) -> str:
        if sr != sample_rate:
            from scipy.signal import resample_poly
            g = gcd(sr, sample_rate)
            audio = resample_poly(
                audio, sample_rate // g, sr // g
            ).astype(np.float32)
        cfg = StreamingConfig(
            window_seconds=6.0,
            hop_seconds=2.0,
            sample_rate=sample_rate,
            frame_rate=500,
            carrier_hz=carrier_hz,
            confidence_threshold=confidence_threshold,
            digit_threshold=digit_threshold,
            beam_width=beam_width,
            beam_emit_bonus=beam_emit_bonus,
            callsign_prior_weight=callsign_prior_weight,
        )
        with torch.no_grad():
            hyp = decode_offline(
                model, audio, cfg, device,
                lm=lm, fusion_weight=fusion_weight,
            )
        return format_output(hyp)

    return _decode


# --------------------------------------------------------------------------- #
# Run + reporting
# --------------------------------------------------------------------------- #


def _entries_to_samples(
    entries: list[BenchEntry], sample_rate: int
) -> list[tuple[BenchEntry, Sample]]:
    """Load audio + normalised GT for each entry."""
    out: list[tuple[BenchEntry, Sample]] = []
    for e in entries:
        audio = _load_audio(e.audio, sample_rate)
        gt_raw = e.gt.read_text(encoding="utf-8")
        gt_norm = _normalize_prose(gt_raw, e.lang)
        out.append(
            (
                e,
                Sample(
                    sample_id=e.id,
                    text=gt_norm,
                    audio=audio,
                    sample_rate=sample_rate,
                ),
            )
        )
    return out


def _format_table(rows: list[dict]) -> str:
    """Render a Markdown table. Rows: list of dicts with consistent keys."""
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "| " + " | ".join("-" * widths[c] for c in cols) + " |"
    head = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[c]).ljust(widths[c]) for c in cols) + " |"
        for r in rows
    )
    return "\n".join([head, sep, body])


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/bench/manifest.jsonl"),
    )
    p.add_argument(
        "--models", default="rnnt_phase5_8",
        help="comma-separated registry names (e.g. "
             "rnnt_phase5_7,rnnt_phase5_8)",
    )
    p.add_argument(
        "--presets", default="live",
        help="comma-separated preset names (live,prose,contest,conservative)",
    )
    p.add_argument("--device", default=None)
    p.add_argument(
        "--show-hyp", action="store_true",
        help="also print first 200 chars of each hypothesis (debugging)",
    )
    p.add_argument(
        "--beam-width", type=int, default=1,
        help="RNN-T decode beam width (Phase 7.0). 1 = greedy (default). "
             ">1 disables LM fusion in the streaming path.",
    )
    p.add_argument(
        "--beam-emit-bonus", type=float, default=0.0,
        help="Length-bias correction for beam search (nats per non-"
             "blank emission). Only applied when --beam-width > 1.",
    )
    p.add_argument(
        "--callsign-prior-weight", type=float, default=0.0,
        help="Phase 7.1 ITU-shape rescorer weight, multiplied into "
             "score_callsign() on every word-boundary emission. "
             "Only consulted when --beam-width > 1.",
    )
    args = p.parse_args(argv)

    repo_root = Path.cwd()
    device = torch.device(args.device or _auto_device())

    entries = load_manifest(args.manifest, repo_root)
    if not entries:
        print(f"[bench_lcwo] manifest {args.manifest} is empty.", file=sys.stderr)
        return 2

    pairs = _entries_to_samples(entries, sample_rate=8000)
    print(f"[bench_lcwo] {len(entries)} clips, "
          f"device={device}, manifest={args.manifest}")

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    preset_names = [n.strip() for n in args.presets.split(",") if n.strip()]

    detail_rows: list[dict] = []
    summary_rows: list[dict] = []

    cache: dict[str, RnntModel] = {}
    lm_cache: dict[str, GptLM] = {}

    for model_name in model_names:
        if model_name not in cache:
            ckpt_path = resolve_model(model_name)
            cache[model_name] = _load_rnnt(ckpt_path, device)
        model = cache[model_name]

        for preset_name in preset_names:
            preset = get_preset(preset_name)
            lm = None
            fusion_weight = 0.0
            if preset.lm and preset.fusion_weight > 0:
                if preset.lm not in lm_cache:
                    lm_cache[preset.lm] = _load_lm(
                        resolve_model(preset.lm), device
                    )
                lm = lm_cache[preset.lm]
                fusion_weight = preset.fusion_weight

            if args.beam_width > 1 and lm is not None:
                # Cannot combine in current path — drop LM with a warning
                # so a single --beam-width run still produces a clean
                # acoustic-beam A/B vs the same preset's greedy numbers.
                print(
                    f"[bench_lcwo] beam_width={args.beam_width} > 1 — "
                    f"skipping LM fusion for preset {preset_name}.",
                    file=sys.stderr,
                )
                lm = None
                fusion_weight = 0.0
            decoder = make_morseformer_decoder(
                model, device=device,
                confidence_threshold=preset.confidence_threshold,
                digit_threshold=preset.digit_threshold,
                lm=lm, fusion_weight=fusion_weight,
                beam_width=args.beam_width,
                beam_emit_bonus=args.beam_emit_bonus,
                callsign_prior_weight=args.callsign_prior_weight,
            )
            result = run_benchmark(decoder, [s for _, s in pairs])
            for (e, _), r in zip(pairs, result.per_sample):
                row = {
                    "model": model_name,
                    "preset": preset_name,
                    "clip": e.id,
                    "lang": e.lang,
                    "wpm": e.wpm if e.wpm is not None else "-",
                    "CER%": f"{r.cer*100:.2f}",
                    "WER%": f"{r.wer*100:.2f}",
                }
                if args.show_hyp:
                    row["hyp[:120]"] = r.hypothesis[:120]
                detail_rows.append(row)
            summary_rows.append({
                "model": model_name,
                "preset": preset_name,
                "n": result.n_samples,
                "mean_CER%": f"{result.mean_cer*100:.2f}",
                "mean_WER%": f"{result.mean_wer*100:.2f}",
            })

    print()
    print("## Per-clip results")
    print(_format_table(detail_rows))
    print()
    print("## Summary")
    print(_format_table(summary_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
