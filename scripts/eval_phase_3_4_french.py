"""French-specific evaluation for Phase 3.4 — measures how well the
model handles the three new tokens (É / À / apostrophe) added to the
49-vocab tokenizer.

The training-time validation uses ``DEFAULT_MIX`` which contains no
French prose, so it cannot tell whether the freshly initialised vocab
rows for É / À / ' are actually being learned. This script fills that
gap: it generates synthetic CW from French prose biased toward
fragments that contain the new tokens, decodes with greedy RNN-T, and
reports

    * overall CER / WER on French prose at each SNR (including clean),
    * per-token presence recall for É / À / ' (proxy for "did the model
      emit the new token when the reference expected one?"),
    * a few qualitative reference / hypothesis pairs for inspection.

Usage::

    python -m scripts.eval_phase_3_4_french \
        --ckpt checkpoints/phase3_4/best_rnnt.pt \
        --snrs "clean,+20,+10,0,-5" \
        --n-per-snr 40
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from eval.metrics import character_error_rate, word_error_rate
from morseformer.core.tokenizer import decode
from morseformer.data.synthetic import (
    DatasetConfig,
    _FALLBACK_SHORT_TEXTS,
    collate,
    estimate_cw_duration_s,
)
from morseformer.data.text import sample_prose_fr
from morseformer.data.validation import ValidationSample, _render_one
from morseformer.features import extract_features
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel
from morseformer.core.tokenizer import encode

NEW_TOKENS: tuple[str, ...] = ("É", "À", "'")


def _parse_snrs(spec: str) -> tuple[float, ...]:
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == "clean":
            out.append(math.inf)
        else:
            out.append(float(tok))
    return tuple(out)


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_rnnt(path: Path, device: torch.device) -> RnntModel:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    enc = cfg["model"]["encoder"]
    ckpt_vocab = cfg["model"].get("vocab_size")
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
        # Honour the checkpoint's own vocab — required to load a 46-vocab
        # baseline (Phase 3.3 and earlier) into the same script. The
        # tokenizer module is still 49-vocab, but the model itself can
        # be built at the smaller size; ``decode()`` then only uses the
        # first 46 indices, which exactly reproduces the legacy output.
        vocab_size=ckpt_vocab if ckpt_vocab is not None else 49,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=cfg["model"]["d_pred"],
        pred_lstm_layers=cfg["model"]["pred_lstm_layers"],
        d_joint=cfg["model"]["d_joint"],
        dropout=cfg["model"]["dropout"],
        vocab_size=ckpt_vocab if ckpt_vocab is not None else 49,
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


def _sample_fr_text_with_new_token(
    rng: np.random.Generator,
    target_duration_s: float,
    wpm: float,
    require_new_token: bool,
    max_retries: int = 16,
) -> str:
    """Draw a French prose fragment that fits within the WPM budget.

    When ``require_new_token`` is set, instead of rejection-sampling
    (which is wasteful at low WPM where most fragments overshoot the
    audio budget) we look up an É / À / ' position directly in the FR
    corpus and extract a window around it. That guarantees a new token
    *and* lets us tune the fragment length to the WPM budget without
    losing samples to the retry loop.
    """
    budget = target_duration_s * 0.9
    if require_new_token:
        from morseformer.data.text import _load_prose, _snap_to_word_boundary
        prose = _load_prose()
        fr = prose.get("fr", "")
        if fr:
            positions = [i for i, c in enumerate(fr) if c in NEW_TOKENS]
            if positions:
                for _ in range(max_retries):
                    pos = positions[int(rng.integers(0, len(positions)))]
                    half = int(rng.integers(8, 22))
                    start = max(0, pos - half)
                    end = min(len(fr), pos + half)
                    start, end = _snap_to_word_boundary(fr, start, end)
                    text = fr[start:end].strip()
                    if not text or not any(t in text for t in NEW_TOKENS):
                        continue
                    if estimate_cw_duration_s(text, wpm) <= budget:
                        return text
        # Fall through to plain rejection sampling if positional
        # extraction failed.
    for _ in range(max_retries):
        text = sample_prose_fr(rng, min_chars=10, max_chars=40)
        if estimate_cw_duration_s(text, wpm) > budget:
            continue
        if require_new_token and not any(t in text for t in NEW_TOKENS):
            continue
        return text
    short = _FALLBACK_SHORT_TEXTS()
    return short[int(rng.integers(0, len(short)))]


def _build_french_validation_set(
    snrs: tuple[float, ...],
    n_per_snr: int,
    *,
    wpm_bins: tuple[float, ...] = (18.0, 22.0, 26.0),
    target_duration_s: float = 6.0,
    sample_rate: int = 8000,
    freq_hz: float = 600.0,
    rx_filter_bw: float | None = 500.0,
    require_new_token_fraction: float = 0.7,
    seed: int = 20_260_427,
) -> list[ValidationSample]:
    """Generate ``n_per_snr × len(snrs)`` French validation samples.

    ``require_new_token_fraction`` of the samples are forced to contain
    at least one of É / À / ' so the per-token presence statistics have
    enough positive examples to be meaningful. The remainder are
    unconstrained French prose, which still has ~50 % chance of
    containing a new token at the chosen length.
    """
    from morse_synth.keying import KeyingConfig
    keying = KeyingConfig(shape="raised_cosine", rise_ms=5.0)
    rng = np.random.default_rng(seed)
    target_samples = int(round(target_duration_s * sample_rate))
    samples: list[ValidationSample] = []

    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.target_duration_s = target_duration_s
    cfg.sample_rate = sample_rate
    cfg.target_samples = target_samples
    cfg.freq_hz = freq_hz
    cfg.keying = keying

    for snr in snrs:
        for k in range(n_per_snr):
            wpm = wpm_bins[k % len(wpm_bins)]
            require_new = rng.random() < require_new_token_fraction
            text = _sample_fr_text_with_new_token(
                rng, target_duration_s, wpm, require_new
            )
            channel_seed = int(rng.integers(0, 2**31 - 1))
            audio = _render_one(
                text, wpm, cfg, snr, rx_filter_bw, channel_seed
            )
            if audio.size > target_samples:
                audio = audio[:target_samples]
            elif audio.size < target_samples:
                pad = np.zeros(target_samples - audio.size, dtype=audio.dtype)
                audio = np.concatenate([audio, pad])
            from morseformer.features import FrontendConfig
            features = extract_features(audio, sample_rate, FrontendConfig())
            tok_list = encode(text)
            samples.append(
                ValidationSample(
                    features=torch.from_numpy(features),
                    tokens=torch.tensor(tok_list, dtype=torch.int64),
                    text=text,
                    wpm=wpm,
                    snr_db=snr,
                    n_frames=int(features.shape[0]),
                    n_tokens=len(tok_list),
                )
            )
    return samples


def _val_batches(samples: list[ValidationSample], batch_size: int):
    for i in range(0, len(samples), batch_size):
        yield collate([s.as_batch_item() for s in samples[i : i + batch_size]])


def _score(
    model: RnntModel,
    samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
    show_examples: int = 6,
) -> dict:
    """Decode + score the FR validation set.

    Returns a dict with overall + per-SNR + per-new-token statistics
    plus a list of ``(ref, hyp, snr)`` triples for inspection.
    """
    per_snr: dict[float, list[tuple[float, float]]] = {}
    # Per-new-token: count (ref_count, hyp_count, both_present_count)
    # so we can report a presence-recall and presence-precision proxy.
    tok_stats: dict[str, dict[str, int]] = {
        t: {"ref": 0, "hyp": 0, "both": 0, "samples_with_ref": 0}
        for t in NEW_TOKENS
    }
    examples: list[tuple[str, str, str, float]] = []  # (ref, hyp, snr, cer)
    count = 0
    for batch in _val_batches(samples, batch_size):
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)
        with torch.no_grad():
            hyps = model.greedy_rnnt_decode(features, lengths)
        for j in range(features.size(0)):
            ref = samples[count]
            hyp_text = decode(hyps[j])
            cer = character_error_rate(ref.text, hyp_text)
            wer = word_error_rate(ref.text, hyp_text)
            per_snr.setdefault(ref.snr_db, []).append((cer, wer))
            for t in NEW_TOKENS:
                ref_n = ref.text.count(t)
                hyp_n = hyp_text.count(t)
                tok_stats[t]["ref"] += ref_n
                tok_stats[t]["hyp"] += hyp_n
                tok_stats[t]["both"] += min(ref_n, hyp_n)
                if ref_n > 0:
                    tok_stats[t]["samples_with_ref"] += 1
            if len(examples) < show_examples and any(t in ref.text for t in NEW_TOKENS):
                examples.append((ref.text, hyp_text, _fmt_snr(ref.snr_db), cer))
            count += 1

    out: dict = {"per_snr": {}, "token_stats": {}, "examples": examples}
    weighted_cer = 0.0
    n_total = 0
    for snr, pairs in per_snr.items():
        cer_m = sum(c for c, _ in pairs) / len(pairs)
        wer_m = sum(w for _, w in pairs) / len(pairs)
        out["per_snr"][snr] = {"cer": cer_m, "wer": wer_m, "n": len(pairs)}
        weighted_cer += cer_m * len(pairs)
        n_total += len(pairs)
    out["overall_cer"] = weighted_cer / max(1, n_total)

    for t in NEW_TOKENS:
        s = tok_stats[t]
        out["token_stats"][t] = {
            "ref_count": s["ref"],
            "hyp_count": s["hyp"],
            "both_count": s["both"],
            "samples_with_ref": s["samples_with_ref"],
            "presence_recall": (s["both"] / s["ref"]) if s["ref"] else 0.0,
            "presence_precision": (s["both"] / s["hyp"]) if s["hyp"] else 0.0,
        }
    return out


def _fmt_snr(snr: float) -> str:
    return "clean" if math.isinf(snr) else f"{snr:+.0f}dB"


def _print_report(label: str, scored: dict, snrs: tuple[float, ...]) -> None:
    print(f"\n=== {label} ===")
    print(f"  overall CER (FR prose, accent-rich): {scored['overall_cer']*100:.2f} %")
    print()
    print(f"  {'SNR':>7} | {'n':>4} | {'CER':>7} | {'WER':>7}")
    print(f"  {'-'*7} + {'-'*4} + {'-'*7} + {'-'*7}")
    for snr in snrs:
        s = scored["per_snr"].get(snr)
        if s is None:
            continue
        print(f"  {_fmt_snr(snr):>7} | {s['n']:>4d} | "
              f"{s['cer']*100:>6.2f}% | {s['wer']*100:>6.2f}%")
    print()
    print(f"  {'token':>5} | {'ref#':>5} | {'hyp#':>5} | {'recall':>6} | {'precision':>9}")
    print(f"  {'-'*5} + {'-'*5} + {'-'*5} + {'-'*6} + {'-'*9}")
    for t in NEW_TOKENS:
        ts = scored["token_stats"][t]
        print(f"  {t:>5} | {ts['ref_count']:>5d} | {ts['hyp_count']:>5d} | "
              f"{ts['presence_recall']*100:>5.1f}% | "
              f"{ts['presence_precision']*100:>8.1f}%")
    if scored["examples"]:
        print()
        print(f"  qualitative samples (with ≥1 new token in ref):")
        for ref, hyp, snr_lbl, cer in scored["examples"]:
            print(f"    [{snr_lbl}, CER={cer*100:5.1f}%]")
            print(f"      REF: {ref!r}")
            print(f"      HYP: {hyp!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path,
                   default=Path("checkpoints/phase3_4/best_rnnt.pt"))
    p.add_argument("--baseline-ckpt", type=Path, default=None,
                   help="Optional Phase 3.3 checkpoint for side-by-side. "
                        "It will be evaluated on the same FR set, but its "
                        "tokenizer is 46-vocab, so it will *never* emit "
                        "É / À / '. The CER number is the meaningful "
                        "comparison: the new model should beat the old "
                        "on FR even before measuring per-token recall.")
    p.add_argument("--snrs", default="clean,+20,+10,+5,0,-5")
    p.add_argument("--n-per-snr", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device or _auto_device())
    snrs = _parse_snrs(args.snrs)
    print(f"[eval_fr] device={device} snrs={[_fmt_snr(s) for s in snrs]} "
          f"n_per_snr={args.n_per_snr}")

    samples = _build_french_validation_set(snrs, args.n_per_snr)
    n_with_new = sum(
        1 for s in samples if any(t in s.text for t in NEW_TOKENS)
    )
    print(f"[eval_fr] FR validation set: {len(samples)} samples "
          f"({n_with_new} contain ≥1 of É/À/')")

    print(f"[eval_fr] candidate: {args.ckpt}")
    cand = _load_rnnt(args.ckpt, device)
    cand_scored = _score(cand, samples, device, args.batch_size)
    _print_report(f"candidate {args.ckpt}", cand_scored, snrs)
    del cand
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.baseline_ckpt is not None:
        print()
        print(f"[eval_fr] baseline: {args.baseline_ckpt} "
              f"(46-vocab — will never emit new tokens)")
        base = _load_rnnt(args.baseline_ckpt, device)
        base_scored = _score(base, samples, device, args.batch_size)
        _print_report(f"baseline {args.baseline_ckpt}", base_scored, snrs)
        delta = (cand_scored["overall_cer"] - base_scored["overall_cer"]) * 100
        print()
        print(f"[eval_fr] FR-prose CER delta (cand - base): {delta:+.2f} pp "
              f"(negative = candidate wins)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
