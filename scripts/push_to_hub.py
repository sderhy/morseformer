"""Upload a prepared ``release/`` directory to a HuggingFace Hub repo.

Run ``scripts.prepare_release`` first to produce the directory, then::

    python -m scripts.push_to_hub \
        --repo-id sderhy/morseformer \
        --release-dir release/ \
        --commit-message "morseformer v0.1.0 initial release"

Authentication: set the ``HF_TOKEN`` environment variable (token scope
``write``), or run ``huggingface-cli login`` beforehand. The script
will call ``create_repo(exist_ok=True)`` so it works both for a
first-time creation and for updating an existing repo.

The ``--dry-run`` flag lists what *would* be uploaded without actually
calling the Hub — use it to sanity-check the file set before spending
bandwidth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default="sderhy/morseformer",
                   help="HF Hub repo id, ``<user>/<name>``.")
    p.add_argument("--release-dir", type=Path, default=Path("release"),
                   help="directory produced by scripts.prepare_release")
    p.add_argument("--commit-message", default="morseformer release upload")
    p.add_argument("--repo-type", default="model",
                   choices=("model", "dataset", "space"))
    p.add_argument("--private", action="store_true",
                   help="create the repo as private (only used on first "
                        "creation; ignored if the repo already exists)")
    p.add_argument("--dry-run", action="store_true",
                   help="list files without calling the Hub")
    p.add_argument("--token", default=None,
                   help="HF token. Defaults to $HF_TOKEN or cached login.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.release_dir.is_dir():
        raise SystemExit(
            f"[push_to_hub] {args.release_dir} does not exist or is not a "
            f"directory — run scripts/prepare_release.py first."
        )

    files = sorted(
        p for p in args.release_dir.iterdir() if p.is_file()
    )
    if not files:
        raise SystemExit(
            f"[push_to_hub] {args.release_dir} is empty — nothing to upload."
        )

    print(f"[push_to_hub] repo:   {args.repo_id} ({args.repo_type})")
    print(f"[push_to_hub] source: {args.release_dir}/")
    total = 0
    for f in files:
        sz = f.stat().st_size
        total += sz
        print(f"  - {f.name:<24}  ({sz/1e6:>6.1f} MB)")
    print(f"[push_to_hub] total:  {total/1e6:.1f} MB across {len(files)} files")

    if args.dry_run:
        print("[push_to_hub] --dry-run: not calling the Hub.")
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit(
            "[push_to_hub] huggingface_hub is not installed. "
            "Install it with: pip install huggingface_hub"
        ) from e

    token = args.token or os.environ.get("HF_TOKEN")
    # create_repo is idempotent with exist_ok=True — fine to call on
    # every push.
    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(args.release_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
    )
    print(f"[push_to_hub] done → https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
