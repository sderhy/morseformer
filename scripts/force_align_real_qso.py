"""Forced-align real-QSO aligned chunks (Étape A — Phase 11).

Takes an existing ``data/real/<op>_aligned.jsonl`` produced by
``scripts/prepare_real_qso.py``, and for each chunk runs the Phase 5.5
acoustic encoder + CTC head against the chunk's audio, then uses
``torchaudio.functional.forced_align`` to compute per-token timestamps
within the chunk. Emits a new JSONL with the same records plus two new
fields:

  - ``tokens``         : list[int]   tokenizer indices for ``label``
                                     (parallel to ``char_starts_s``)
  - ``char_starts_s``  : list[float] start time (in seconds, relative
                                     to the chunk's ``start_s``) of each
                                     emitted token

Motivation (cf NEXT.md §3 Étape A): the previous augmentation injected
silence at *linearly interpolated* positions, which doesn't correspond
to true inter-word boundaries — and the model ended up learning
"in real audio, silence ≠ word gap". With true forced-alignment
timestamps, the Phase 11 augmentation can insert silence at the actual
word boundaries.

Run::

    python -m scripts.force_align_real_qso
    python -m scripts.force_align_real_qso --operator g3ses --device cuda
    python -m scripts.force_align_real_qso --sanity-dump out/sanity.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import forced_align

from morseformer.cli.registry import resolve_model
from morseformer.core.tokenizer import BLANK_INDEX, SPACE_INDEX, encode
from morseformer.data.real_audio import _load_wav_to_float32
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.rnnt import RnntModel
from scripts.decode_audio import _auto_device, _is_rnnt_checkpoint, _rnnt_cfg_from_state

_DEFAULT_IN_DIR = Path("data/real")
_DEFAULT_OPERATORS = ("g3ses", "g6pz")
_SAMPLE_RATE = 8000
_ENCODER_SUBSAMPLE = 4  # AcousticModel ConvSubsampling 4× time reduction


def _load_model(ckpt_path: Path, device: torch.device, *, use_ema: bool = True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not _is_rnnt_checkpoint(ckpt):
        raise SystemExit(f"[force-align] {ckpt_path}: not an RNN-T checkpoint")
    state = dict(ckpt["model"])
    if use_ema and ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    cfg = _rnnt_cfg_from_state(state)
    model = RnntModel(cfg).to(device).eval()
    model.load_state_dict(state)
    return model


def _align_chunk(
    model: RnntModel,
    chunk_audio: np.ndarray,
    label: str,
    *,
    sample_rate: int,
    fcfg: FrontendConfig,
    device: torch.device,
) -> tuple[list[int], list[float], float] | None:
    """Forced-align ``chunk_audio`` (float32 [-1, 1]) to ``label``.

    Returns ``(tokens, char_starts_s, mean_score)`` on success — the
    tokens encoded from ``label``, the per-token start time (seconds,
    chunk-relative), and the mean per-frame log-prob along the path.
    Returns ``None`` if the alignment cannot be produced (empty label,
    log_probs shorter than required by CTC, etc).
    """
    tokens = encode(label)
    if not tokens:
        return None
    features = extract_features(chunk_audio, sample_rate, fcfg)
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=device)
    with torch.no_grad():
        enc_out, enc_lengths = model.acoustic.encode(x, lengths)
        logits = model.acoustic.head(enc_out)  # [B, T', V]
    log_probs = F.log_softmax(logits, dim=-1)
    t_valid = int(enc_lengths[0].item())
    if t_valid <= 0:
        return None

    # forced_align requires T >= L + N_repeat. Reject early if violated.
    n_repeat = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if t_valid < len(tokens) + n_repeat:
        return None

    target = torch.tensor([tokens], dtype=torch.int32, device=device)
    in_lens = torch.tensor([t_valid], dtype=torch.int32, device=device)
    tgt_lens = torch.tensor([len(tokens)], dtype=torch.int32, device=device)
    try:
        aligned, frame_scores = forced_align(
            log_probs[:, :t_valid, :].contiguous(),
            target,
            input_lengths=in_lens,
            target_lengths=tgt_lens,
            blank=BLANK_INDEX,
        )
    except RuntimeError:
        return None

    aligned_seq = aligned[0].tolist()
    # Extract first frame of each non-blank, non-repeat emission.
    char_start_frames: list[int] = []
    prev = BLANK_INDEX
    for f_idx, tok in enumerate(aligned_seq):
        if tok != BLANK_INDEX and tok != prev:
            char_start_frames.append(f_idx)
        prev = tok
    if len(char_start_frames) != len(tokens):
        # Alignment path didn't recover every token (shouldn't happen with
        # the T >= L + N_repeat check above, but be defensive).
        return None

    # 4× ConvSubsampling × frontend.frame_rate gives encoder frame Hz.
    enc_frame_rate = fcfg.frame_rate / _ENCODER_SUBSAMPLE
    char_starts_s = [f / enc_frame_rate for f in char_start_frames]
    mean_score = float(frame_scores[0].mean().item())
    return tokens, char_starts_s, mean_score


def _audio_chunks_iter(records: list[dict], cache: dict[str, np.ndarray], sr: int):
    """Yield (record, chunk_audio) using a per-path audio cache."""
    for rec in records:
        path = rec["audio_path"]
        if path not in cache:
            cache[path] = _load_wav_to_float32(Path(path), sr)
        audio_full = cache[path]
        start = int(round(float(rec["start_s"]) * sr))
        end = int(round(float(rec["end_s"]) * sr))
        chunk = audio_full[start:end].astype(np.float32, copy=False)
        yield rec, chunk


def _process_operator(
    op_jsonl_in: Path,
    op_jsonl_out: Path,
    *,
    model: RnntModel,
    fcfg: FrontendConfig,
    device: torch.device,
    sample_rate: int,
    sanity_dump: Path | None,
    sanity_n: int,
) -> tuple[int, int, list[dict]]:
    with op_jsonl_in.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return 0, 0, []
    cache: dict[str, np.ndarray] = {}
    out_records: list[dict] = []
    n_ok = 0
    n_skip = 0
    sanity_records: list[dict] = []
    for rec, chunk in _audio_chunks_iter(records, cache, sample_rate):
        res = _align_chunk(
            model, chunk, rec["label"],
            sample_rate=sample_rate, fcfg=fcfg, device=device,
        )
        if res is None:
            n_skip += 1
            continue
        tokens, char_starts_s, mean_score = res
        new_rec = dict(rec)
        new_rec["tokens"] = tokens
        new_rec["char_starts_s"] = char_starts_s
        new_rec["align_score"] = mean_score
        out_records.append(new_rec)
        n_ok += 1
        if sanity_dump and len(sanity_records) < sanity_n:
            # Sanity check: per-token RMS energy of the audio in a small
            # window around char_starts_s. Spaces should land in low-RMS
            # zones (inter-word silences) whereas letters should land in
            # high-RMS zones.
            sanity_records.append(_sanity_block(
                new_rec, chunk, sample_rate=sample_rate,
            ))

    op_jsonl_out.parent.mkdir(parents=True, exist_ok=True)
    with op_jsonl_out.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n_ok, n_skip, sanity_records


def _sanity_block(
    rec: dict, chunk: np.ndarray, *, sample_rate: int
) -> dict:
    """For one aligned record, compute per-token mean ``|x|`` over the
    *gap* until the next token start. Space tokens should have a
    substantially lower energy than letter tokens — if the ratio drops
    toward 1.0, the alignment is suspect."""
    tokens = rec["tokens"]
    char_starts_s = rec["char_starts_s"]
    gap_energies: list[float] = []
    for i in range(len(tokens)):
        a = int(round(char_starts_s[i] * sample_rate))
        b = (
            int(round(char_starts_s[i + 1] * sample_rate))
            if i + 1 < len(tokens)
            else chunk.size
        )
        a = max(0, min(a, chunk.size))
        b = max(a, min(b, chunk.size))
        if b > a:
            gap_energies.append(float(np.mean(np.abs(chunk[a:b]))))
        else:
            gap_energies.append(0.0)
    is_space = [tok == SPACE_INDEX for tok in tokens]
    return {
        "audio_path": rec["audio_path"],
        "chunk_idx": rec["chunk_idx"],
        "start_s": rec["start_s"],
        "end_s": rec["end_s"],
        "label": rec["label"],
        "tokens": tokens,
        "char_starts_s": char_starts_s,
        "gap_energies": gap_energies,
        "is_space": is_space,
    }


def _print_sanity_summary(blocks: list[dict]) -> None:
    """Aggregate gap-energy stats: spaces (low) vs letters (high)."""
    space_e: list[float] = []
    letter_e: list[float] = []
    for b in blocks:
        # The last token's gap extends to the end of the chunk (often
        # trailing silence) and would bias the comparison; skip it.
        for e, sp in zip(b["gap_energies"][:-1], b["is_space"][:-1]):
            (space_e if sp else letter_e).append(e)
    if not space_e or not letter_e:
        print("[sanity] not enough samples for ratio")
        return
    mean_space = float(np.mean(space_e))
    mean_letter = float(np.mean(letter_e))
    ratio = mean_letter / mean_space if mean_space > 0 else float("inf")
    print(
        f"[sanity] n_blocks={len(blocks)} "
        f"n_space_gaps={len(space_e)} mean|x|={mean_space:.4f}  "
        f"n_letter_gaps={len(letter_e)} mean|x|={mean_letter:.4f}  "
        f"letter/space={ratio:.2f}x  (≥2.5x = good alignment)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in-dir", type=Path, default=_DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_IN_DIR)
    p.add_argument(
        "--operator", action="append", default=None,
        help="Operator name(s); default = "
             f"{', '.join(_DEFAULT_OPERATORS)}",
    )
    p.add_argument(
        "--acoustic", default="rnnt_phase5_5",
        help="Registry name of the acoustic checkpoint used for CTC head.",
    )
    p.add_argument("--freq", type=float, default=600.0)
    p.add_argument("--bandwidth", type=float, default=200.0)
    p.add_argument("--frame-rate", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--sanity-dump", type=Path, default=None,
        help="Optional path to dump per-token energy stats for the "
             "first N chunks (for manual inspection).",
    )
    p.add_argument("--sanity-n", type=int, default=20)
    p.add_argument(
        "--out-suffix", default="_force_aligned",
        help="Suffix appended to the input stem for the output JSONL "
             "(default: '_force_aligned' → 'g3ses_force_aligned.jsonl')",
    )
    args = p.parse_args(argv)

    operators = args.operator or list(_DEFAULT_OPERATORS)
    device = torch.device(args.device or _auto_device())
    ckpt_path = resolve_model(args.acoustic)
    model = _load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[force-align] device={device} acoustic={args.acoustic} ({n_params:,} params)")
    fcfg = FrontendConfig(
        tone_freq=args.freq, bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )

    all_sanity: list[dict] = []
    t0 = time.time()
    for op in operators:
        op_jsonl_in = args.in_dir / f"{op}_aligned.jsonl"
        op_jsonl_out = args.out_dir / f"{op}{args.out_suffix}.jsonl"
        if not op_jsonl_in.exists():
            print(f"[force-align] {op}: {op_jsonl_in} not found, skip.")
            continue
        n_ok, n_skip, sanity = _process_operator(
            op_jsonl_in, op_jsonl_out,
            model=model, fcfg=fcfg, device=device,
            sample_rate=_SAMPLE_RATE,
            sanity_dump=args.sanity_dump,
            sanity_n=args.sanity_n,
        )
        all_sanity.extend(sanity)
        print(
            f"[force-align] {op}: ok={n_ok} skip={n_skip} → {op_jsonl_out}"
        )

    if args.sanity_dump:
        args.sanity_dump.parent.mkdir(parents=True, exist_ok=True)
        with args.sanity_dump.open("w", encoding="utf-8") as f:
            json.dump(all_sanity, f, ensure_ascii=False, indent=2)
        print(f"[force-align] sanity dump → {args.sanity_dump}")
        _print_sanity_summary(all_sanity)

    print(f"[force-align] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
