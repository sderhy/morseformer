"""Prepare a real-QSO corpus for Phase 8 fine-tuning.

For each ``<operator>/*.wav`` + ``<operator>/*.txt`` pair under
``--root`` (default ``../testlive``):

1. resample the audio to 8 kHz mono float32 and write it as a
   16-bit-PCM ``.wav`` under ``--out-dir/<operator>/<clip>.wav``
   (``RealAudioCWDataset`` refuses to resample at load time so the
   on-disk file must already be at the target rate);
2. decode the resampled audio in non-overlapping ``--chunk-seconds``
   windows with the registered acoustic checkpoint, keeping both the
   RNN-T and CTC hypotheses per chunk;
3. normalise the matching ``.txt`` (prosign brackets ``<KN>`` /
   ``<BK>`` plus the existing :func:`_normalize_prose`);
4. align the concatenated decode against the normalised ground truth
   with ``difflib.SequenceMatcher``, then map each chunk's char range
   back to a slice of the ground truth (same algorithm as
   ``scripts/align_ebook_cw.py``);
5. emit one aligned JSONL line per chunk, in the schema
   ``RealAudioCWDataset`` consumes::

       {audio_path, chunk_idx, start_s, end_s, label, decoded, score}

   to ``--out-dir/<operator>_aligned.jsonl``.

By default skips the two freq-OOD outliers identified by the
2026-05-19 audit (g6pz/G12 + g6pz/G14 — carrier at +138/+190 Hz,
hopelessly out-of-distribution).

Run::

    python -m scripts.prepare_real_qso
    python -m scripts.prepare_real_qso --operator g3ses --device cuda
    python -m scripts.prepare_real_qso --acoustic rnnt_phase5_5 --use-rnnt
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from eval.bench_lcwo import _normalize_prosign_brackets
from morseformer.cli.registry import resolve_model
from morseformer.core.tokenizer import ctc_greedy_decode, decode
from morseformer.data.text import _normalize_prose
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.rnnt import RnntModel
from scripts.align_ebook_cw import _build_char_mapping
from scripts.decode_audio import (
    _auto_device,
    _is_rnnt_checkpoint,
    _load_audio,
    _rnnt_cfg_from_state,
)

_DEFAULT_ROOT = Path("../testlive")
_DEFAULT_OUT_DIR = Path("data/real")
_DEFAULT_OPERATORS = ("g3ses", "g6pz")
_DEFAULT_SKIP_CLIPS = ("G12", "G14")  # freq-OOD per 2026-05-19 audit
_SAMPLE_RATE = 8000


def _save_wav_int16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write ``audio`` (float32 ∈ [-1, 1]) as 16-bit-PCM WAV."""
    from scipy.io import wavfile
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) or 1.0
    scaled = (audio / max(peak, 1.0) * 32767.0).clip(-32768, 32767).astype(
        np.int16
    )
    wavfile.write(str(path), sample_rate, scaled)


def _load_model(ckpt_path: Path, device: torch.device, *, use_ema: bool = True):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not _is_rnnt_checkpoint(ckpt):
        raise SystemExit(f"[prepare] {ckpt_path}: not an RNN-T checkpoint")
    state = dict(ckpt["model"])
    if use_ema and ckpt.get("ema"):
        for k, v in ckpt["ema"].items():
            if k in state:
                state[k] = v
    cfg = _rnnt_cfg_from_state(state)
    model = RnntModel(cfg).to(device).eval()
    model.load_state_dict(state)
    return model


def _decode_chunks(
    model: RnntModel,
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    sample_rate: int,
    device: torch.device,
    fcfg: FrontendConfig,
) -> list[dict]:
    """Return one decoded record per non-overlapping chunk."""
    chunk_samples = int(round(chunk_seconds * sample_rate))
    n_chunks = (audio.size + chunk_samples - 1) // chunk_samples
    padded = np.zeros(n_chunks * chunk_samples, dtype=audio.dtype)
    padded[: audio.size] = audio
    chunks = padded.reshape(n_chunks, chunk_samples)
    out: list[dict] = []
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            feats = extract_features(chunk, sample_rate, fcfg)
            x = torch.from_numpy(feats).unsqueeze(0).to(device)
            lengths = torch.tensor(
                [feats.shape[0]], dtype=torch.long, device=device
            )
            enc_out, enc_lengths = model.acoustic.encode(x, lengths)
            ctc_logits = model.acoustic.head(enc_out)
            ctc_argmax = ctc_logits.argmax(dim=-1)[0].cpu().tolist()
            ctc_len = int(enc_lengths[0].item())
            ctc_hyp = ctc_greedy_decode(ctc_argmax[:ctc_len])
            rnnt_tokens = model.greedy_rnnt_decode(x, lengths)[0]
            rnnt_hyp = decode(rnnt_tokens)
            out.append({
                "chunk_idx": i,
                "start_s": i * chunk_seconds,
                "end_s": (i + 1) * chunk_seconds,
                "rnnt_hyp": rnnt_hyp,
                "ctc_hyp": ctc_hyp,
            })
    return out


def _align_clip(
    chunks: list[dict],
    gt_text: str,
    *,
    use_rnnt: bool,
) -> tuple[list[dict], float]:
    """Run the SequenceMatcher alignment for one clip's chunks against
    its already-normalised ground truth. Returns the aligned records
    and the global decode/gt SequenceMatcher ratio for the clip."""
    field = "rnnt_hyp" if use_rnnt else "ctc_hyp"
    decoded_parts: list[str] = []
    char_ranges: list[tuple[int, int]] = []
    pos = 0
    for i, ch in enumerate(chunks):
        if i > 0:
            decoded_parts.append(" ")
            pos += 1
        txt = ch[field]
        char_ranges.append((pos, pos + len(txt)))
        decoded_parts.append(txt)
        pos += len(txt)
    decoded_text = "".join(decoded_parts)
    if not decoded_text or not gt_text:
        return [], 0.0
    mapping = _build_char_mapping(decoded_text, gt_text)
    global_ratio = difflib.SequenceMatcher(
        autojunk=False, a=decoded_text, b=gt_text
    ).ratio()
    records: list[dict] = []
    for ch, (cs, ce) in zip(chunks, char_ranges):
        gt_start = mapping[cs]
        gt_end = mapping[ce]
        label = gt_text[gt_start:gt_end].strip()
        local_ratio = difflib.SequenceMatcher(
            autojunk=False, a=ch[field], b=label
        ).ratio() if label else 0.0
        records.append({
            "chunk_idx": ch["chunk_idx"],
            "start_s": ch["start_s"],
            "end_s": ch["end_s"],
            "label": label,
            "decoded": ch[field],
            "score": local_ratio,
        })
    return records, global_ratio


def _emit_records(
    records: list[dict],
    audio_path: Path,
    min_label_chars: int,
    max_label_chars: int,
) -> list[dict]:
    """Filter out short / over-long labels (alignment drift), attach
    the audio path."""
    out: list[dict] = []
    for r in records:
        if len(r["label"]) < min_label_chars or len(r["label"]) > max_label_chars:
            continue
        out.append({
            "audio_path": str(audio_path),
            **r,
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    p.add_argument(
        "--operator", action="append", default=None,
        help="Operator subdir(s) to prepare. Repeatable. Default = "
             f"{', '.join(_DEFAULT_OPERATORS)}",
    )
    p.add_argument(
        "--acoustic", default="rnnt_phase5_5",
        help="Registry name of the acoustic used for chunk decoding.",
    )
    p.add_argument(
        "--use-ctc", action="store_true",
        help="Use the CTC hypothesis for alignment (default = RNN-T).",
    )
    p.add_argument("--use-rnnt", dest="use_ctc", action="store_false")
    p.add_argument("--chunk-seconds", type=float, default=6.0)
    p.add_argument("--freq", type=float, default=600.0)
    p.add_argument("--bandwidth", type=float, default=200.0)
    p.add_argument("--frame-rate", type=int, default=500)
    p.add_argument("--device", default=None)
    p.add_argument("--lang", default="en")
    p.add_argument(
        "--skip-clip", action="append", default=None,
        help="Clip stem(s) to skip. Default = "
             f"{', '.join(_DEFAULT_SKIP_CLIPS)} (freq-OOD per "
             "project_audit_real_qso_2026_05_19).",
    )
    p.add_argument("--min-label-chars", type=int, default=3)
    p.add_argument("--max-label-chars", type=int, default=80)
    args = p.parse_args(argv)

    root = args.root
    out_dir = args.out_dir
    if not root.is_dir():
        print(f"[prepare] root not found: {root}", file=sys.stderr)
        return 2
    operators = args.operator or list(_DEFAULT_OPERATORS)
    skip = set(args.skip_clip if args.skip_clip is not None else _DEFAULT_SKIP_CLIPS)
    device = torch.device(args.device or _auto_device())
    use_rnnt = not args.use_ctc

    print(
        f"[prepare] root={root} out={out_dir} operators={operators} "
        f"acoustic={args.acoustic} use_rnnt={use_rnnt} device={device} "
        f"skip={sorted(skip)}"
    )
    ckpt_path = resolve_model(args.acoustic)
    model = _load_model(ckpt_path, device)
    print(
        f"[prepare] model loaded "
        f"({sum(p.numel() for p in model.parameters()):,} params)"
    )
    fcfg = FrontendConfig(
        tone_freq=args.freq, bandwidth=args.bandwidth,
        frame_rate=args.frame_rate,
    )

    for op in operators:
        op_dir = root / op
        if not op_dir.is_dir():
            print(f"[prepare] {op}: directory not found, skipping.")
            continue
        op_out_audio = out_dir / op
        op_out_jsonl = out_dir / f"{op}_aligned.jsonl"
        pairs = sorted(
            (wav, wav.with_suffix(".txt"))
            for wav in op_dir.glob("*.wav")
            if wav.with_suffix(".txt").exists()
        )
        if not pairs:
            print(f"[prepare] {op}: no (wav, txt) pairs found.")
            continue
        print(f"\n[prepare] {op}: {len(pairs)} pairs found")
        all_records: list[dict] = []
        per_clip_ratios: list[tuple[str, float, int, int]] = []
        t0 = time.time()
        for wav, txt in pairs:
            stem = wav.stem
            if stem in skip:
                print(f"  skip {stem} (in --skip-clip)")
                continue
            audio = _load_audio(wav, _SAMPLE_RATE)
            dur = len(audio) / _SAMPLE_RATE
            out_wav = op_out_audio / f"{stem}.wav"
            _save_wav_int16(out_wav, audio, _SAMPLE_RATE)
            chunks = _decode_chunks(
                model, audio,
                chunk_seconds=args.chunk_seconds,
                sample_rate=_SAMPLE_RATE,
                device=device, fcfg=fcfg,
            )
            gt_raw = _normalize_prosign_brackets(
                txt.read_text(encoding="utf-8")
            )
            gt = _normalize_prose(gt_raw, args.lang)
            aligned, global_ratio = _align_clip(
                chunks, gt, use_rnnt=use_rnnt,
            )
            emitted = _emit_records(
                aligned, out_wav,
                min_label_chars=args.min_label_chars,
                max_label_chars=args.max_label_chars,
            )
            all_records.extend(emitted)
            per_clip_ratios.append((stem, global_ratio, len(chunks), len(emitted)))
            print(
                f"  {stem:<6} {dur:>6.1f}s {len(chunks):>3} chunks "
                f"global_ratio={global_ratio:.3f} kept={len(emitted)}"
            )

        op_out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with op_out_jsonl.open("w", encoding="utf-8") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        scores = [r["score"] for r in all_records]
        mean_score = float(np.mean(scores)) if scores else 0.0
        score_q25 = float(np.percentile(scores, 25)) if scores else 0.0
        score_q75 = float(np.percentile(scores, 75)) if scores else 0.0
        print(
            f"[prepare] {op}: {len(all_records)} aligned chunks "
            f"(mean_score={mean_score:.3f}, q25={score_q25:.3f}, "
            f"q75={score_q75:.3f}) → {op_out_jsonl} "
            f"(audio dir: {op_out_audio}, elapsed {elapsed:.1f}s)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
