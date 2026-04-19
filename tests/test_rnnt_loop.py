"""Smoke + integration tests for the Phase 3 RNN-T training loop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")

from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.models.acoustic import AcousticConfig, AcousticModel  # noqa: E402
from morseformer.models.rnnt import RnntConfig  # noqa: E402
from morseformer.train.rnnt_loop import (  # noqa: E402
    RnntTrainConfig,
    load_pretrained_encoder_state,
    train,
)


def _tiny_cfg(tmp_path: Path) -> RnntTrainConfig:
    return RnntTrainConfig(
        model=RnntConfig(
            encoder=AcousticConfig(
                d_model=32, n_heads=4, n_layers=2,
                ff_expansion=2, conv_kernel=7, dropout=0.0,
            ),
            d_pred=32,
            pred_lstm_layers=1,
            d_joint=32,
            dropout=0.0,
        ),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=2, seed=1),
        ctc_weight=0.3,
        rnnt_weight=0.7,
        peak_lr=1e-3,
        warmup_steps=10,
        total_steps=60,
        batch_size=4,
        num_workers=0,
        log_every=10,
        eval_every=30,
        save_every=0,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
        ema_decay=0.99,
    )


def test_loss_descends_on_short_run(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    result = train(cfg)
    assert result["steps"] == 60

    # Pull the loss trace from the JSONL and confirm it trends down.
    losses: list[float] = []
    ctcs: list[float] = []
    rnnts: list[float] = []
    evals = 0
    with (tmp_path / "train.jsonl").open() as f:
        for line in f:
            evt = json.loads(line)
            if evt.get("event") == "step":
                losses.append(evt["loss"])
                ctcs.append(evt["ctc"])
                rnnts.append(evt["rnnt"])
            elif evt.get("event") == "eval":
                evals += 1
                # Both heads produce finite CER values, both branches of
                # per-wpm / per-snr dicts are populated.
                assert "ctc_cer" in evt and "rnnt_cer" in evt
                assert evt["ctc_cer"] >= 0.0 and evt["rnnt_cer"] >= 0.0
                assert evt["ctc_per_wpm_cer"]
                assert evt["rnnt_per_wpm_cer"]

    assert len(losses) >= 5
    assert losses[-1] < losses[0], (
        f"loss did not descend: start={losses[0]:.3f} end={losses[-1]:.3f}"
    )
    # Both components contribute: neither is frozen at zero.
    assert ctcs[0] > 0 and rnnts[0] > 0
    # At least one eval fired during the run.
    assert evals >= 1

    # Checkpoints exist.
    assert (tmp_path / "last.pt").exists()
    # best_ctc.pt / best_rnnt.pt get written on the first eval.
    assert (tmp_path / "best_ctc.pt").exists()
    assert (tmp_path / "best_rnnt.pt").exists()


def test_checkpoint_schema(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path)
    cfg.total_steps = 20
    cfg.eval_every = 20
    cfg.log_every = 5
    train(cfg)

    ckpt = torch.load(
        tmp_path / "last.pt", map_location="cpu", weights_only=False
    )
    assert ckpt["step"] == 20
    for key in ("model", "ema", "optimizer", "scheduler",
                "best_ctc_cer", "best_rnnt_cer", "metrics", "config"):
        assert key in ckpt, f"missing {key!r} in checkpoint"
    assert ckpt["metrics"] is not None
    import math
    assert math.isfinite(ckpt["metrics"]["ctc_cer"])
    assert math.isfinite(ckpt["metrics"]["rnnt_cer"])


def test_pretrained_encoder_bootstraps_weights(tmp_path: Path) -> None:
    """A Phase 2-shaped checkpoint drops into the RNN-T encoder and the
    corresponding weights match exactly before any gradient step."""
    enc_cfg = AcousticConfig(
        d_model=32, n_heads=4, n_layers=2,
        ff_expansion=2, conv_kernel=7, dropout=0.0,
    )
    # Fake a Phase 2 checkpoint: a randomised AcousticModel + a matching
    # "ema" dict (subset of parameters).
    src = AcousticModel(enc_cfg)
    with torch.no_grad():
        for p in src.parameters():
            p.normal_(0, 0.3)
    ema_state = {
        name: p.detach().clone()
        for name, p in src.named_parameters()
        if p.requires_grad and p.dtype.is_floating_point
    }
    ckpt_path = tmp_path / "phase2_fake.pt"
    torch.save(
        {
            "step": 1000,
            "model": src.state_dict(),
            "ema": ema_state,
            "optimizer": {},
            "scheduler": {},
            "config": {},
            "metrics": None,
            "best_cer": 0.01,
        },
        ckpt_path,
    )

    # load_pretrained_encoder_state: the EMA values overlay the model's.
    merged = load_pretrained_encoder_state(ckpt_path)
    for name, p in src.named_parameters():
        assert torch.allclose(merged[name], p)

    # A very short train run with --pretrained-encoder should produce a
    # model whose encoder *before* any step already matches src. We
    # verify this indirectly: run 1 training step and check that the
    # encoder weights have only moved by an amount consistent with a
    # single optimizer step from the *bootstrapped* init (i.e. they
    # didn't reset to a random init on load).
    cfg = _tiny_cfg(tmp_path)
    cfg.model = RnntConfig(
        encoder=enc_cfg, d_pred=32, pred_lstm_layers=1, d_joint=32, dropout=0.0
    )
    cfg.pretrained_encoder = ckpt_path
    cfg.warmup_steps = 0
    cfg.total_steps = 1
    cfg.log_every = 1
    cfg.eval_every = 1  # force eval + best_*.pt writes on first step
    cfg.save_every = 0
    cfg.peak_lr = 1e-6     # tiny LR so weights barely move
    cfg.ema_decay = 0.99
    train(cfg)

    loaded = torch.load(
        tmp_path / "last.pt", map_location="cpu", weights_only=False
    )
    model_state = loaded["model"]
    # Compare the acoustic-encoder weights: with LR=1e-6 + grad clipping,
    # a single step can move each param by at most ~1e-6 per element
    # (often less). Source weights should be within a loose tolerance.
    for name, p_src in src.named_parameters():
        key = f"acoustic.{name}"
        assert key in model_state, f"missing {key}"
        diff = (model_state[key] - p_src).abs().max().item()
        assert diff < 1e-2, (
            f"{key} moved by {diff:.4f} after one 1e-6 LR step — "
            "bootstrap likely didn't take effect"
        )


def test_pretrained_encoder_only_model_state(tmp_path: Path) -> None:
    """Checkpoints without an EMA key are still a valid bootstrap source."""
    enc_cfg = AcousticConfig(
        d_model=16, n_heads=2, n_layers=1,
        ff_expansion=2, conv_kernel=7, dropout=0.0,
    )
    src = AcousticModel(enc_cfg)
    ckpt_path = tmp_path / "no_ema.pt"
    torch.save({"model": src.state_dict(), "step": 0}, ckpt_path)
    merged = load_pretrained_encoder_state(ckpt_path)
    # No EMA → returns model state unchanged.
    for name, p in src.named_parameters():
        assert torch.allclose(merged[name], p)
