"""End-to-end smoke tests: audio → text → CER.

These exercise the full decode pipeline against a synthetic CW clip with
known ground truth, and assert that the recovered text stays inside a
character-error-rate budget.

Two layers:

1. **Rule-based baseline** — always runs in CI. Validates
   `morse_synth.render` → `morseformer.baselines.rule_based.decode`
   → `eval.metrics.character_error_rate`. No checkpoint, no torch
   inference. The audit's "audio → text → CER" sanity check.

2. **RNN-T release checkpoint** — opt-in. Skipped unless
   `release/rnnt_phase5_5.pt` is on disk or `MORSEFORMER_CKPT` is set in
   the environment. Runs the public `RnntModel.greedy_rnnt_decode` path
   through the project's frontend and asserts the same CER budget.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from eval.metrics import character_error_rate
from morse_synth import core as synth
from morse_synth.operator import OperatorConfig
from morseformer.baselines import rule_based
from morseformer.core.tokenizer import decode as token_decode

_GROUND_TRUTH = "CQ DE F4HYY K"
_WPM = 20.0
_FREQ = 600.0
_SAMPLE_RATE = 8000
_BASELINE_CER_BUDGET = 0.10
_RNNT_CER_BUDGET = 0.05


def _render_clean(text: str = _GROUND_TRUTH) -> np.ndarray:
    return synth.render(
        text,
        operator=OperatorConfig(wpm=_WPM, seed=42),
        freq=_FREQ,
        sample_rate=_SAMPLE_RATE,
        amplitude=0.5,
    )


def test_e2e_baseline_clean_cer() -> None:
    audio = _render_clean()
    hyp = rule_based.decode(audio, _SAMPLE_RATE, tone_freq=_FREQ)
    cer = character_error_rate(_GROUND_TRUTH, hyp)
    assert cer <= _BASELINE_CER_BUDGET, (
        f"baseline E2E regressed: ref={_GROUND_TRUTH!r} hyp={hyp!r} "
        f"cer={cer:.3f} > budget {_BASELINE_CER_BUDGET}"
    )


def _resolve_release_ckpt() -> Path | None:
    env = os.environ.get("MORSEFORMER_CKPT")
    if env:
        p = Path(env)
        return p if p.exists() else None
    p = Path("release/rnnt_phase5_5.pt")
    return p if p.exists() else None


@pytest.mark.skipif(
    _resolve_release_ckpt() is None,
    reason="no local release checkpoint (set MORSEFORMER_CKPT or place "
           "rnnt_phase5_5.pt under release/)",
)
def test_e2e_rnnt_release_cer() -> None:
    from morseformer.features.frontend import extract_features
    from morseformer.models.acoustic import AcousticConfig
    from morseformer.models.rnnt import RnntConfig, RnntModel

    ckpt_path = _resolve_release_ckpt()
    assert ckpt_path is not None
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = dict(ckpt["model"])
    for k, v in (ckpt.get("ema") or {}).items():
        if k in state:
            state[k] = v
    enc = ckpt["config"]["model"]["encoder"]
    vocab_size = ckpt["config"]["model"].get("vocab_size")
    extra = {"vocab_size": vocab_size} if vocab_size is not None else {}
    encoder_cfg = AcousticConfig(
        d_model=enc["d_model"], n_heads=enc["n_heads"], n_layers=enc["n_layers"],
        ff_expansion=enc["ff_expansion"], conv_kernel=enc["conv_kernel"],
        dropout=enc["dropout"], **extra,
    )
    rnnt_cfg = RnntConfig(
        encoder=encoder_cfg,
        d_pred=ckpt["config"]["model"]["d_pred"],
        pred_lstm_layers=ckpt["config"]["model"]["pred_lstm_layers"],
        d_joint=ckpt["config"]["model"]["d_joint"],
        dropout=ckpt["config"]["model"]["dropout"],
        **extra,
    )
    model = RnntModel(rnnt_cfg).eval()
    model.load_state_dict(state)

    audio = _render_clean()
    features = extract_features(audio, _SAMPLE_RATE)
    features_t = torch.from_numpy(features).unsqueeze(0)            # [1, T, 1]
    lengths = torch.tensor([features_t.shape[1]], dtype=torch.long)
    with torch.no_grad():
        hyps = model.greedy_rnnt_decode(features_t, lengths)
    hyp = token_decode(hyps[0])
    cer = character_error_rate(_GROUND_TRUTH, hyp)
    assert cer <= _RNNT_CER_BUDGET, (
        f"release RNN-T E2E regressed: ref={_GROUND_TRUTH!r} hyp={hyp!r} "
        f"cer={cer:.3f} > budget {_RNNT_CER_BUDGET}"
    )
