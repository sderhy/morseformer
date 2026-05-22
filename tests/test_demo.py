"""Smoke tests for the Gradio demo helpers.

The tests exercise the public demo decode path without launching a Gradio
server and without loading model checkpoints. Skipped when the optional
``gradio`` dependency isn't installed (the demo extras are not part of
the default install).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio")


def test_gradio_prose_preset_runs_acoustic_only(monkeypatch) -> None:
    """Post-714eec0 the prose preset dropped lm_phase5_2 (literary LM
    hurt amateur jargon). The demo path should now decode acoustic-
    only with no fusion."""
    from demo import app as demo_app

    calls: dict[str, object] = {}

    sentinel_model = object()

    def fake_get_model(name: str) -> object:
        calls["model_name"] = name
        return sentinel_model

    def fake_get_lm(name: str) -> object:
        calls["lm_name"] = name
        return object()

    def fake_decode_offline(model, audio, cfg, device, *, lm=None, fusion_weight=0.0):
        calls["model"] = model
        calls["audio_shape"] = audio.shape
        calls["confidence_threshold"] = cfg.confidence_threshold
        calls["digit_threshold"] = cfg.digit_threshold
        calls["lm"] = lm
        calls["fusion_weight"] = fusion_weight
        return "CQ TEST"

    monkeypatch.setattr(demo_app, "_get_model", fake_get_model)
    monkeypatch.setattr(demo_app, "_get_lm", fake_get_lm)
    monkeypatch.setattr(demo_app, "decode_offline", fake_decode_offline)

    audio = (8000, np.zeros(800, dtype=np.float32))
    out = demo_app.decode(audio, "prose")

    assert out == "CQ TEST"
    assert calls["model_name"] == "rnnt_phase11b"
    assert calls.get("lm_name") is None  # _get_lm not called when preset.lm is None
    assert calls["model"] is sentinel_model
    assert calls["lm"] is None
    assert calls["fusion_weight"] == 0.0
    assert calls["audio_shape"] == (800,)
    assert calls["confidence_threshold"] == 0.6
    assert calls["digit_threshold"] == 0.90


def test_gradio_preset_info_lists_all_presets() -> None:
    """Post-714eec0 none of the bundled presets carry a neural LM, so
    the old "first decode also downloads the LM" assertion no longer
    applies. We just check the info string mentions every preset."""
    from demo import app as demo_app
    from morseformer.cli.presets import PRESETS

    info = demo_app._preset_info()
    for name in PRESETS:
        assert f"{name}:" in info
