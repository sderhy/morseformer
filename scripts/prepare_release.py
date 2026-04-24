"""Strip training artefacts from Phase-3 / Phase-4 checkpoints to
produce lightweight release files for HuggingFace Hub.

A training checkpoint includes the optimizer state (AdamW moments),
scheduler state, full metrics history, etc. None of this is useful
for inference — only the ``model`` weights, the EMA shadow, and the
``config`` dict are needed. Stripping those out typically cuts file
size by 2-3× and makes the download a lot friendlier.

Usage::

    python -m scripts.prepare_release \
        --rnnt-ckpt checkpoints/phase3_0/best_rnnt.pt \
        --lm-ckpt   checkpoints/lm_phase4_0/best.pt \
        --out-dir   release/

This writes ``release/rnnt_phase3_0.pt`` and ``release/lm_phase4_0.pt``,
copies the repo-level ``README.md`` and ``MODEL_CARD.md`` into the
same directory, and prints a summary of file sizes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


# Keys we strip out of the checkpoint before release. ``step`` and
# ``metrics`` are kept for provenance; they are tiny.
_STRIP_KEYS = ("optimizer", "scheduler")


def _strip(ckpt_path: Path, out_path: Path) -> tuple[int, int]:
    """Read ``ckpt_path``, drop training-only keys, write ``out_path``.

    Returns ``(in_size_bytes, out_size_bytes)``.
    """
    in_size = ckpt_path.stat().st_size
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    stripped = {k: v for k, v in ckpt.items() if k not in _STRIP_KEYS}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, str(out_path))
    out_size = out_path.stat().st_size
    return in_size, out_size


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rnnt-ckpt", type=Path, default=Path("checkpoints/phase3_0/best_rnnt.pt"),
    )
    p.add_argument(
        "--lm-ckpt", type=Path, default=Path("checkpoints/lm_phase4_0/best.pt"),
    )
    p.add_argument(
        "--readme", type=Path, default=Path("README.md"),
    )
    p.add_argument(
        "--model-card", type=Path, default=Path("MODEL_CARD.md"),
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("release"),
    )
    return p


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out_dir

    print(f"[prepare_release] output: {out}/")
    rnnt_out = out / "rnnt_phase3_0.pt"
    lm_out = out / "lm_phase4_0.pt"

    rnnt_in, rnnt_done = _strip(args.rnnt_ckpt, rnnt_out)
    print(f"  rnnt:  {args.rnnt_ckpt}  ({rnnt_in/1e6:.1f} MB)  "
          f"→  {rnnt_out}  ({rnnt_done/1e6:.1f} MB)")

    lm_in, lm_done = _strip(args.lm_ckpt, lm_out)
    print(f"  lm:    {args.lm_ckpt}  ({lm_in/1e6:.1f} MB)  "
          f"→  {lm_out}  ({lm_done/1e6:.1f} MB)")

    # HF expects the model card at ``README.md`` on the Hub. We give it
    # the ``MODEL_CARD.md`` body (which carries the YAML metadata and
    # the detailed use / limitations section) and keep the repo-level
    # README alongside as a secondary reference file.
    _copy(args.model_card, out / "README.md")
    print(f"  card:  {args.model_card}  →  {out / 'README.md'}  "
          f"(used as the HF Hub model card)")
    _copy(args.readme, out / "REPO_README.md")
    print(f"  repo:  {args.readme}  →  {out / 'REPO_README.md'}")

    print()
    print("[prepare_release] done. Ready to upload with scripts/push_to_hub.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
