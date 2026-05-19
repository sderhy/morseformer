"""Smoke tests for the Gradio demo helpers.

The tests exercise the public demo decode path without launching a Gradio
server and without loading model checkpoints.
"""

from __future__ import annotations

import numpy as np


def test_gradio_prose_preset_uses_lm(monkeypatch) -> None:
    from demo import app as demo_app

    calls: dict[str, object] = {}

    sentinel_model = object()
    sentinel_lm = object()

    def fake_get_model(name: str) -> object:
        calls["model_name"] = name
        return sentinel_model

    def fake_get_lm(name: str) -> object:
        calls["lm_name"] = name
        return sentinel_lm

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
    assert calls["model_name"] == "rnnt_phase5_5"
    assert calls["lm_name"] == "lm_phase5_2"
    assert calls["model"] is sentinel_model
    assert calls["lm"] is sentinel_lm
    assert calls["fusion_weight"] == 0.7
    assert calls["audio_shape"] == (800,)
    assert calls["confidence_threshold"] == 0.6
    assert calls["digit_threshold"] == 0.90
