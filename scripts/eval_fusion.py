"""Evaluate shallow fusion of a trained RNN-T model with a trained LM.

Sweeps several fusion weights over the same SNR-ladder validation set
we use for the acoustic model and reports CER / WER, per-SNR CER, and
the absolute / relative improvement against the no-fusion baseline.

Usage::

    python -m scripts.eval_fusion \
        --rnnt-ckpt checkpoints/phase3_0/best_rnnt.pt \
        --lm-ckpt   checkpoints/lm_phase4_0/best.pt \
        --fusion-weights "0.0,0.1,0.2,0.3,0.5" \
        --snrs "+20,+10,+5,0,-5,-10"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import collate
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_snr_ladder_validation,
)
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.fusion import (
    FusionConfig,
    greedy_rnnt_decode_with_lm,
)
from morseformer.models.lm import GptLM, LmConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rnnt-ckpt", type=Path, required=True)
    p.add_argument("--lm-ckpt", type=Path, required=True)
    p.add_argument("--fusion-weights", default="0.0,0.1,0.2,0.3,0.5")
    p.add_argument("--snrs", default="+20,+10,+5,0,-5,-10")
    p.add_argument("--rx-filter-bw", type=float, default=500.0)
    p.add_argument("--n-per-wpm", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--lm-temperature", type=float, default=1.0)
    return p


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_floats(spec: str) -> tuple[float, ...]:
    out: list[float] = []
    for t in spec.split(","):
        t = t.strip()
        if t:
            out.append(float(t))
    return tuple(out)


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    """Reconstruct an :class:`RnntModel` from a Phase 3 checkpoint.

    The saved ``config`` dict carries the encoder widths / RNN-T head
    sizes so we can rebuild the exact architecture before loading
    weights. EMA weights are preferred when available.
    """
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


def _load_lm(path: Path, device: torch.device) -> GptLM:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    lm_cfg = LmConfig(
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


def _val_batches(samples: list[ValidationSample], batch_size: int) -> list[dict]:
    out = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        out.append(collate([s.as_batch_item() for s in chunk]))
    return out


def _evaluate_weight(
    rnnt: RnntModel,
    lm: GptLM,
    val_samples: list[ValidationSample],
    batch_size: int,
    device: torch.device,
    fusion_weight: float,
    lm_temperature: float,
) -> dict:
    from eval.metrics import character_error_rate, word_error_rate
    fusion_cfg = FusionConfig(
        fusion_weight=fusion_weight, lm_temperature=lm_temperature
    )
    cer_sum = 0.0
    wer_sum = 0.0
    count = 0
    per_snr: dict[float, list[float]] = {}
    for batch in _val_batches(val_samples, batch_size):
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)
        hyps = greedy_rnnt_decode_with_lm(
            rnnt, lm, features, lengths, fusion_cfg
        )
        for j in range(features.size(0)):
            ref_sample = val_samples[count]
            hyp_text = decode(hyps[j])
            cer = character_error_rate(ref_sample.text, hyp_text)
            wer = word_error_rate(ref_sample.text, hyp_text)
            cer_sum += cer
            wer_sum += wer
            per_snr.setdefault(ref_sample.snr_db, []).append(cer)
            count += 1
    return {
        "fusion_weight": fusion_weight,
        "cer": cer_sum / count,
        "wer": wer_sum / count,
        "per_snr": {
            ("inf" if math.isinf(k) else k): sum(v) / len(v)
            for k, v in sorted(per_snr.items(), key=lambda kv: (math.isinf(kv[0]), kv[0]))
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())

    print(f"[eval_fusion] device={device}")
    print(f"[eval_fusion] rnnt={args.rnnt_ckpt}")
    print(f"[eval_fusion] lm  ={args.lm_ckpt}")

    rnnt = _load_rnnt(args.rnnt_ckpt, device)
    lm = _load_lm(args.lm_ckpt, device)
    print(f"[eval_fusion] rnnt params={rnnt.num_parameters():,}")
    print(f"[eval_fusion] lm   params={lm.num_parameters():,}")

    snrs = _parse_floats(args.snrs)
    val_cfg = ValidationConfig(n_per_wpm=args.n_per_wpm)
    val_samples = build_snr_ladder_validation(
        snrs, cfg=val_cfg, rx_filter_bw=args.rx_filter_bw
    )
    print(f"[eval_fusion] val-set: {len(val_samples)} samples "
          f"({val_cfg.n_per_wpm}/wpm × {len(val_cfg.wpm_bins)} wpm × {len(snrs)} snr)")

    weights = _parse_floats(args.fusion_weights)
    results: list[dict] = []
    for w in weights:
        r = _evaluate_weight(
            rnnt, lm, val_samples, args.batch_size, device, w,
            args.lm_temperature,
        )
        results.append(r)
        print(flush=True)
        print(f"=== fusion_weight = {w:.2f} ===", flush=True)
        print(f"  cer={r['cer']:.4f}  wer={r['wer']:.4f}", flush=True)
        for snr, v in r["per_snr"].items():
            print(f"    SNR={snr:>6}  CER={v:.4f}", flush=True)

    baseline = results[0]
    print()
    print("=== summary (Δ vs baseline) ===")
    for r in results:
        delta = r["cer"] - baseline["cer"]
        rel = (100.0 * delta / baseline["cer"]) if baseline["cer"] else 0.0
        print(f"  λ={r['fusion_weight']:.2f}  cer={r['cer']:.4f}  "
              f"Δcer={delta:+.4f} ({rel:+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
