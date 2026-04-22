"""Smoke tests for the RNN-T + external-LM fusion decoder.

Covers the two sanity invariants we rely on:

1. ``greedy_rnnt_decode_with_lm`` with ``λ_lm = λ_ilm = 0`` must emit
   exactly the same tokens as the vanilla ``RnntModel.greedy_rnnt_decode``
   — if it doesn't, the fusion decode path has diverged from the
   reference implementation.

2. ``λ_lm = 0`` with ``λ_ilm > 0`` must still produce a valid decode
   (catches ILM-subtraction bugs, e.g. shape mismatches on the
   zero-encoder frame).
"""

from __future__ import annotations

import torch

from morseformer.models.acoustic import AcousticConfig
from morseformer.models.fusion import (
    FusionConfig,
    greedy_rnnt_decode_with_lm,
)
from morseformer.models.lm import GptLM, LmConfig
from morseformer.models.rnnt import RnntConfig, RnntModel


def _tiny_rnnt() -> RnntModel:
    enc = AcousticConfig(
        d_model=32, n_heads=4, n_layers=2, ff_expansion=2, conv_kernel=7,
        dropout=0.0,
    )
    return RnntModel(RnntConfig(
        encoder=enc, d_pred=32, pred_lstm_layers=1, d_joint=32, dropout=0.0,
    ))


def _tiny_lm() -> GptLM:
    return GptLM(LmConfig(d_model=32, n_heads=4, n_layers=2, dropout=0.0))


def test_fusion_zero_weights_matches_vanilla() -> None:
    """λ_lm = λ_ilm = 0 must reproduce the vanilla RNN-T decode bit-for-bit."""
    torch.manual_seed(0)
    rnnt = _tiny_rnnt().eval()
    lm = _tiny_lm().eval()
    features = torch.randn(2, 48, 1)
    lengths = torch.tensor([48, 40])

    vanilla = rnnt.greedy_rnnt_decode(features, lengths)
    fused = greedy_rnnt_decode_with_lm(
        rnnt, lm, features, lengths,
        FusionConfig(fusion_weight=0.0, ilm_weight=0.0),
    )
    assert vanilla == fused, (
        f"fusion decoder diverged from vanilla at zero weights:\n"
        f"  vanilla={vanilla}\n  fused  ={fused}"
    )


def test_fusion_ilm_only_is_valid() -> None:
    """With only ILM subtraction active, the decoder must still emit a
    list-of-lists of the expected length. Smoke-tests the ilm path."""
    torch.manual_seed(1)
    rnnt = _tiny_rnnt().eval()
    lm = _tiny_lm().eval()
    features = torch.randn(3, 32, 1)
    hyps = greedy_rnnt_decode_with_lm(
        rnnt, lm, features, None,
        FusionConfig(fusion_weight=0.0, ilm_weight=0.5),
    )
    assert isinstance(hyps, list) and len(hyps) == 3
    for h in hyps:
        assert isinstance(h, list)
        for tok in h:
            assert isinstance(tok, int)
            assert tok != rnnt.cfg.blank_index  # blanks must be filtered out


def test_fusion_full_density_ratio_is_valid() -> None:
    """End-to-end smoke test with both λ_lm and λ_ilm > 0."""
    torch.manual_seed(2)
    rnnt = _tiny_rnnt().eval()
    lm = _tiny_lm().eval()
    features = torch.randn(2, 40, 1)
    hyps = greedy_rnnt_decode_with_lm(
        rnnt, lm, features, None,
        FusionConfig(fusion_weight=0.3, ilm_weight=0.3),
    )
    assert isinstance(hyps, list) and len(hyps) == 2
