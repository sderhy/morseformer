"""Build a callsign bench entry — two modes.

Mode A (LCWO practice file, full):
    --src <lcwo-NNN.mp3> --id callsign_lcwo_NNN
    Sibling .txt must exist with "LCWO practice text (NNN) for <call>" header.

Mode B (segment of an existing WAV, e.g. real-radio QSO):
    --src <ragchew.wav> --id callsign_ragchew_<tag> \\
        --start <sec> --duration <sec> --gt "M0MCL DE G3ZRJ"
    GT is the verbatim transcript of the segment.

Output (both modes):
    data/bench/<id>.wav  — 8000 Hz mono PCM_16
    data/bench/<id>.txt  — ground-truth transcript
    appends entry to data/bench/manifest.jsonl with kind="callsign"
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "data" / "bench"
MANIFEST = BENCH_DIR / "manifest.jsonl"
TARGET_SR = 8000


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _strip_lcwo_header(text: str) -> str:
    """Drop the 'LCWO practice text (NNN) for <call>' header line + blank."""
    lines = text.splitlines()
    out = []
    skipping = True
    for line in lines:
        if skipping:
            if line.strip().lower().startswith("lcwo practice text"):
                continue
            if not line.strip():
                continue
            skipping = False
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _manifest_has(entry_id: str) -> bool:
    if not MANIFEST.exists():
        return False
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            if json.loads(line).get("id") == entry_id:
                return True
        except json.JSONDecodeError:
            continue
    return False


def _load_audio_float(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    return audio, sr


def _normalise_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        return audio / peak
    return audio


def _write_bench(out_id: str, audio_8k: np.ndarray, gt_text: str, *,
                 wpm: int | None, source: str) -> None:
    out_wav = BENCH_DIR / f"{out_id}.wav"
    out_txt = BENCH_DIR / f"{out_id}.txt"
    sf.write(str(out_wav), audio_8k, TARGET_SR, subtype="PCM_16")
    out_txt.write_text(gt_text if gt_text.endswith("\n") else gt_text + "\n",
                       encoding="utf-8")
    dur = len(audio_8k) / TARGET_SR
    print(f"[build] wrote {out_wav.name} ({dur:.1f}s @ {TARGET_SR} Hz) "
          f"+ {out_txt.name}")
    if _manifest_has(out_id):
        print(f"[build] manifest already has '{out_id}', skipping append.")
        return
    entry = {
        "id": out_id,
        "audio": str(out_wav.relative_to(REPO_ROOT)),
        "gt": str(out_txt.relative_to(REPO_ROOT)),
        "lang": "en",
        "wpm": wpm,
        "source": source,
        "kind": "callsign",
    }
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[build] appended entry '{out_id}' to {MANIFEST.relative_to(REPO_ROOT)}")


def build_lcwo(src: Path, out_id: str, *, wpm: int | None, source: str) -> None:
    src_txt = src.with_suffix(".txt")
    if not src.exists():
        raise FileNotFoundError(src)
    if not src_txt.exists():
        raise FileNotFoundError(src_txt)
    audio, sr = _load_audio_float(src)
    audio = _resample(audio, sr, TARGET_SR)
    audio = _normalise_peak(audio)
    gt = _strip_lcwo_header(src_txt.read_text(encoding="utf-8"))
    _write_bench(out_id, audio, gt, wpm=wpm, source=source)


def build_segment(src: Path, out_id: str, *, start: float, duration: float,
                  gt: str, wpm: int | None, source: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if duration <= 0:
        raise ValueError("--duration must be > 0")
    if not gt.strip():
        raise ValueError("--gt is required and must be non-empty")
    audio, sr = _load_audio_float(src)
    start_s = int(round(start * sr))
    end_s = int(round((start + duration) * sr))
    end_s = min(end_s, len(audio))
    if start_s >= len(audio):
        raise ValueError(f"--start {start}s past end of audio ({len(audio)/sr:.1f}s)")
    seg = audio[start_s:end_s]
    seg = _resample(seg, sr, TARGET_SR)
    seg = _normalise_peak(seg)
    _write_bench(out_id, seg, gt.strip(), wpm=wpm, source=source)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True,
                   help="LCWO mp3 (mode A) or source WAV/MP3 to segment (mode B)")
    p.add_argument("--id", required=True,
                   help="bench id (e.g. callsign_lcwo_001 or callsign_ragchew_g3zrj)")
    p.add_argument("--wpm", type=int, default=None,
                   help="WPM tag for manifest (optional)")
    p.add_argument("--source", default=None,
                   help="manifest source tag (defaults: lcwo_callsigns / ragchew)")
    p.add_argument("--start", type=float, default=None,
                   help="(mode B) segment start in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="(mode B) segment duration in seconds")
    p.add_argument("--gt", type=str, default=None,
                   help="(mode B) verbatim ground-truth for the segment")
    args = p.parse_args()

    if args.start is not None or args.duration is not None or args.gt is not None:
        if args.start is None or args.duration is None or args.gt is None:
            raise SystemExit("[build] mode B needs all of --start, --duration, --gt")
        source = args.source or "ragchew"
        build_segment(args.src, args.id, start=args.start, duration=args.duration,
                      gt=args.gt, wpm=args.wpm, source=source)
    else:
        source = args.source or "lcwo_callsigns"
        build_lcwo(args.src, args.id, wpm=args.wpm, source=source)


if __name__ == "__main__":
    main()
