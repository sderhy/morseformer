"""Extend a 46-vocab RNN-T checkpoint to the 49-vocab Phase 3.4 layout.

Phase 3.4 adds three French CW characters to the tokenizer (É / À /
apostrophe). The tokenizer keeps the original 46 indices unchanged and
appends the new ones at indices 46–48, so a pretrained checkpoint can
be promoted without retraining from scratch — only the new rows of
the five vocab-dim tensors need fresh initialisation.

This script does that promotion. It loads a legacy checkpoint
(typically ``checkpoints/phase3_3/best_rnnt.pt``), builds a fresh
:class:`morseformer.models.rnnt.RnntModel` at the new vocab size
(picking up the standard Xavier / zero-bias init from
:func:`init_parameters`), copies every non-vocab parameter byte-for-byte,
copies rows 0..45 of the five vocab-dim tensors, leaves rows 46..48 at
the fresh-init values, and writes the result to ``--output``. The
optimiser / scheduler state is dropped (sizes are tied to the model
state) so the next training run will rebuild a fresh AdamW from the
extended weights — this is what we want for a Phase-3.4 fine-tune.

Usage::

    python -m scripts.extend_tokenizer_46_to_49 \
        --input  checkpoints/phase3_3/best_rnnt.pt \
        --output checkpoints/phase3_4/init_extended.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from morseformer.core.tokenizer import LEGACY_VOCAB_SIZE, VOCAB_SIZE
from morseformer.models.acoustic import AcousticConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


# Tensors whose first dim is the vocabulary size. Order matters — we
# slice every key by ``[:LEGACY_VOCAB_SIZE]`` along dim 0.
VOCAB_DIM_KEYS: tuple[str, ...] = (
    "acoustic.head.weight",
    "acoustic.head.bias",
    "pred.embed.weight",
    "joint.out.weight",
    "joint.out.bias",
)


def _build_fresh_model(legacy_cfg: dict) -> RnntModel:
    """Build a fresh RnntModel matching the legacy architecture at the
    new (49-token) vocabulary size."""
    enc = legacy_cfg["encoder"]
    encoder_cfg = AcousticConfig(
        input_dim=enc["input_dim"],
        d_model=enc["d_model"],
        n_heads=enc["n_heads"],
        n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"],
        conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"],
        # Pick up the Phase-3.4 vocab from the tokenizer.
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=legacy_cfg["d_pred"],
        pred_lstm_layers=legacy_cfg["pred_lstm_layers"],
        d_joint=legacy_cfg["d_joint"],
        dropout=legacy_cfg["dropout"],
    )
    return RnntModel(rnnt_cfg)


def _extend_state_dict(
    fresh_state: dict[str, torch.Tensor],
    legacy_state: dict[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    """Return a new state dict combining legacy + fresh-init for new rows.

    Every key in ``legacy_state`` is copied into ``fresh_state``. For
    vocab-dim tensors the legacy slice ``[:46]`` overwrites the same
    slice of the fresh tensor; rows 46–48 keep the fresh init.
    """
    out = dict(fresh_state)  # shallow copy — tensors are still shared
    extended = []
    matched = 0
    skipped: list[str] = []
    for k, v in legacy_state.items():
        if k not in out:
            skipped.append(k)
            continue
        target = out[k]
        if v.shape == target.shape:
            out[k] = v.clone()
            matched += 1
            continue
        if k in VOCAB_DIM_KEYS and v.shape[0] == LEGACY_VOCAB_SIZE \
                and target.shape[0] == VOCAB_SIZE \
                and v.shape[1:] == target.shape[1:]:
            new_v = target.clone()
            new_v[:LEGACY_VOCAB_SIZE] = v
            out[k] = new_v
            extended.append(k)
            continue
        raise RuntimeError(
            f"[{label}] shape mismatch for {k!r}: legacy {tuple(v.shape)} "
            f"vs fresh {tuple(target.shape)}"
        )
    print(f"  {label}: matched {matched} keys, extended {len(extended)} "
          f"vocab-dim keys, skipped {len(skipped)} legacy-only keys")
    if extended:
        for k in extended:
            print(f"    extended: {k}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/phase3_3/best_rnnt.pt"),
        help="Source 46-vocab checkpoint.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/phase3_4/init_extended.pt"),
        help="Destination 49-vocab checkpoint.",
    )
    args = p.parse_args(argv)

    if VOCAB_SIZE != 49 or LEGACY_VOCAB_SIZE != 46:
        raise RuntimeError(
            f"unexpected vocab sizes: VOCAB_SIZE={VOCAB_SIZE} "
            f"LEGACY_VOCAB_SIZE={LEGACY_VOCAB_SIZE} (expected 49 / 46). "
            "Has the tokenizer been edited again?"
        )

    print(f"[extend] loading {args.input}")
    ckpt = torch.load(str(args.input), map_location="cpu", weights_only=False)
    legacy_cfg = ckpt["config"]["model"]
    if legacy_cfg.get("vocab_size") != LEGACY_VOCAB_SIZE:
        raise RuntimeError(
            f"input checkpoint reports vocab_size={legacy_cfg.get('vocab_size')}, "
            f"expected {LEGACY_VOCAB_SIZE}. Refusing to extend a non-46 source."
        )

    print(f"[extend] building fresh RnntModel at vocab_size={VOCAB_SIZE}")
    torch.manual_seed(20260427)  # deterministic init for the new rows
    fresh = _build_fresh_model(legacy_cfg)
    fresh_state = fresh.state_dict()

    print("[extend] extending model weights")
    new_model = _extend_state_dict(fresh_state, ckpt["model"], "model")

    new_ema = None
    if ckpt.get("ema"):
        print("[extend] extending EMA weights")
        # Use the same fresh init as the seed for any EMA row missing in
        # the legacy state (in practice the legacy EMA mirrors `model`).
        new_ema = _extend_state_dict(deepcopy(fresh_state), ckpt["ema"], "ema")

    new_cfg = deepcopy(ckpt["config"])
    new_cfg["model"]["vocab_size"] = VOCAB_SIZE
    new_cfg["model"]["encoder"]["vocab_size"] = VOCAB_SIZE

    out_ckpt: dict = {
        "model": new_model,
        "config": new_cfg,
        # Optimiser / scheduler depend on the model state shapes — drop
        # them so the next training run rebuilds AdamW + LR schedule
        # from the fresh weights. Phase 3.4 is a fine-tune, not a resume.
        "step": 0,
        "best_ctc_cer": float("inf"),
        "best_rnnt_cer": float("inf"),
        "metrics": {},
    }
    if new_ema is not None:
        out_ckpt["ema"] = new_ema
    if "extended_from" in ckpt:
        out_ckpt["extended_chain"] = list(ckpt["extended_from"]) + [str(args.input)]
    else:
        out_ckpt["extended_from"] = str(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, str(args.output))
    print(f"[extend] wrote {args.output}")
    print(f"[extend] new vocab_size = {VOCAB_SIZE}; "
          f"non-vocab tensors preserved byte-for-byte; "
          f"rows {LEGACY_VOCAB_SIZE}..{VOCAB_SIZE - 1} of the 5 vocab-dim "
          f"tensors are fresh Xavier / N(0,1) init.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
