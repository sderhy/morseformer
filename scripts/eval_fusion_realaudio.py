"""Evaluate LM fusion on a real-audio aligned bench.

Unlike :mod:`scripts.eval_fusion` which sweeps over the synthetic
SNR ladder, this script runs the RNN-T-with-LM decoder on the
ground-truth-aligned ebook2cw corpus produced by
:mod:`scripts.align_ebook_cw` (e.g. Alice in Wonderland chapters).

Phase 4.1 found λ-flat results on synthetic CW because the synthetic
text distribution (callsigns / contest macros / short prose) is
narrower than what a char-level English LM can usefully prior. Real
prose audio is the natural test bed: this is where fusion either
shows signal or definitively does not.

Usage::

    python -m scripts.eval_fusion_realaudio \
        --rnnt-ckpt   checkpoints/phase3_5/best_rnnt.pt \
        --lm-ckpt     checkpoints/lm_phase4_0/best.pt \
        --jsonl       data/real/aligned/all_alice.jsonl \
        --n-chunks    500 \
        --score-min   0.5 \
        --fusion-points "0.0:0.0,0.1:0.0,0.2:0.0,0.3:0.0,0.5:0.0"
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from eval.metrics import character_error_rate, word_error_rate
from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import _pad_or_truncate
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.fusion import (
    FusionConfig,
    greedy_rnnt_decode_with_lm,
)
from morseformer.models.lm import GptLM, LmConfig
from scripts.eval_fusion import _load_rnnt, _parse_fusion_points


def _load_lm(path: Path, device: torch.device) -> GptLM:
    """Reconstruct a :class:`GptLM` from a Phase 4 checkpoint.

    Pulls ``vocab_size`` from the saved config so that legacy 46-token
    LMs reload cleanly even after the tokenizer default moved to 49.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    lm_cfg = LmConfig(
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"], dropout=mcfg["dropout"],
    )
    model = GptLM(lm_cfg).to(device)
    state = dict(ckpt["model"])
    ema = ckpt.get("ema")
    if ema:
        for k, v in ema.items():
            if k in state:
                state[k] = v
    model.load_state_dict(state)
    model.eval()
    return model


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rnnt-ckpt", type=Path, required=True)
    p.add_argument("--lm-ckpt", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, required=True,
                   help="aligned JSONL produced by scripts/align_ebook_cw.py")
    p.add_argument("--n-chunks", type=int, default=500,
                   help="number of chunks to sample uniformly at random.")
    p.add_argument("--score-min", type=float, default=0.5,
                   help="drop chunks whose alignment score is below this.")
    p.add_argument("--score-max", type=float, default=1.0,
                   help="upper score bound. Set < 1.0 to focus on chunks "
                        "where the baseline still has CER headroom — useful "
                        "for fusion sensitivity tests.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fusion-points",
        default="0.0:0.0,0.1:0.0,0.2:0.0,0.3:0.0,0.5:0.0",
        help="Comma-separated 'λ_lm:λ_ilm' pairs.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-duration-s", type=float, default=6.0)
    p.add_argument("--sample-rate", type=int, default=8000)
    p.add_argument("--freq-hz", type=float, default=600.0)
    p.add_argument("--lm-temperature", type=float, default=1.0)
    p.add_argument("--device", default=None)
    return p


def _load_wav_to_float32(path: Path, target_sr: int) -> np.ndarray:
    sr, audio = wavfile.read(str(path))
    if sr != target_sr:
        raise RuntimeError(f"{path}: expected sr {target_sr}, got {sr}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio / float(np.iinfo(audio.dtype).max)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs > 1.5:
        audio = audio / max_abs
    return audio


def _load_chunks(args) -> list[dict]:
    """Sample ``n_chunks`` records from the aligned JSONL.

    The JSONL is filtered by ``score`` to drop alignment garbage at the
    bottom and (optionally) cap the top to focus on chunks where the
    baseline has CER headroom. Sampling is uniform without replacement
    under a fixed seed for reproducibility.
    """
    rows: list[dict] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (args.score_min <= float(r["score"]) <= args.score_max):
                continue
            rows.append(r)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    return rows[: args.n_chunks]


def _build_features(
    chunks: list[dict],
    target_samples: int,
    sample_rate: int,
    frontend: FrontendConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for every chunk and stack into a [N, T, F] tensor.

    Audio files are cached so a JSONL referencing 10 wav files reads
    each one exactly once. Features are computed on CPU; the caller
    moves them to GPU per batch.
    """
    cache: dict[str, np.ndarray] = {}
    feats: list[np.ndarray] = []
    n_frames: list[int] = []
    for r in chunks:
        path = r["audio_path"]
        if path not in cache:
            cache[path] = _load_wav_to_float32(Path(path), sample_rate)
        audio_full = cache[path]
        start = int(round(float(r["start_s"]) * sample_rate))
        end = int(round(float(r["end_s"]) * sample_rate))
        audio = audio_full[start:end].astype(np.float32, copy=True)
        audio = _pad_or_truncate(audio, target_samples)
        f = extract_features(audio, sample_rate, frontend)
        feats.append(f)
        n_frames.append(int(f.shape[0]))
    features = torch.from_numpy(np.stack(feats, axis=0))
    lengths = torch.tensor(n_frames, dtype=torch.int64)
    return features, lengths


def _evaluate_point(
    rnnt,
    lm,
    chunks: list[dict],
    features: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
    device: torch.device,
    lm_weight: float,
    ilm_weight: float,
    lm_temperature: float,
) -> dict:
    fusion_cfg = FusionConfig(
        fusion_weight=lm_weight,
        ilm_weight=ilm_weight,
        lm_temperature=lm_temperature,
    )
    cer_sum = 0.0
    wer_sum = 0.0
    n = features.size(0)
    hyps_all: list[str] = []
    for i in range(0, n, batch_size):
        f_batch = features[i : i + batch_size].to(device)
        l_batch = lengths[i : i + batch_size].to(device)
        hyps = greedy_rnnt_decode_with_lm(rnnt, lm, f_batch, l_batch, fusion_cfg)
        for j, h in enumerate(hyps):
            hyp_text = decode(h)
            ref = chunks[i + j]["label"]
            hyps_all.append(hyp_text)
            cer_sum += character_error_rate(ref, hyp_text)
            wer_sum += word_error_rate(ref, hyp_text)
    return {
        "lm_weight": lm_weight,
        "ilm_weight": ilm_weight,
        "cer": cer_sum / n,
        "wer": wer_sum / n,
        "hyps": hyps_all,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    print(f"[eval_fusion_realaudio] device={device}")
    print(f"[eval_fusion_realaudio] rnnt={args.rnnt_ckpt}")
    print(f"[eval_fusion_realaudio] lm  ={args.lm_ckpt}")
    print(f"[eval_fusion_realaudio] jsonl={args.jsonl}")

    rnnt = _load_rnnt(args.rnnt_ckpt, device)
    lm = _load_lm(args.lm_ckpt, device)
    print(f"[eval_fusion_realaudio] rnnt vocab={rnnt.cfg.vocab_size} "
          f"lm vocab={lm.cfg.vocab_size}")

    chunks = _load_chunks(args)
    if not chunks:
        raise SystemExit(
            f"no chunks left after filtering (score in "
            f"[{args.score_min}, {args.score_max}])"
        )
    print(f"[eval_fusion_realaudio] n_chunks={len(chunks)} "
          f"(score in [{args.score_min}, {args.score_max}])")

    target_samples = int(round(args.target_duration_s * args.sample_rate))
    frontend = FrontendConfig()
    print(f"[eval_fusion_realaudio] extracting features …")
    features, lengths = _build_features(
        chunks, target_samples, args.sample_rate, frontend
    )
    print(f"[eval_fusion_realaudio] features shape: "
          f"{tuple(features.shape)}  dtype={features.dtype}")

    points = _parse_fusion_points(args.fusion_points)
    results: list[dict] = []
    for lm_w, ilm_w in points:
        print(flush=True)
        print(f"=== λ_lm={lm_w:.2f}  λ_ilm={ilm_w:.2f} ===", flush=True)
        r = _evaluate_point(
            rnnt, lm, chunks, features, lengths,
            args.batch_size, device, lm_w, ilm_w, args.lm_temperature,
        )
        results.append(r)
        print(f"  cer={r['cer']:.4f}  wer={r['wer']:.4f}", flush=True)

    baseline = results[0]
    print()
    print("=== summary (Δ vs baseline λ_lm=λ_ilm=0) ===")
    for r in results:
        delta = r["cer"] - baseline["cer"]
        rel = (100.0 * delta / baseline["cer"]) if baseline["cer"] else 0.0
        print(
            f"  λ_lm={r['lm_weight']:.2f}  λ_ilm={r['ilm_weight']:.2f}  "
            f"cer={r['cer']:.4f}  Δcer={delta:+.4f} ({rel:+.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
