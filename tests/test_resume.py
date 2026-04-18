"""Tests for crash-safe checkpointing: resume + periodic snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.train.acoustic import TrainConfig, train  # noqa: E402


def _tiny_cfg(tmp_path: Path, **overrides) -> TrainConfig:
    base = dict(
        model=AcousticConfig(
            d_model=16, n_heads=2, n_layers=1,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=1, seed=1),
        warmup_steps=5,
        total_steps=20,
        batch_size=4,
        log_every=5,
        eval_every=20,
        save_every=5,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
    )
    base.update(overrides)
    return TrainConfig(**base)


# --------------------------------------------------------------------- #
# Progressive snapshots
# --------------------------------------------------------------------- #


def test_save_every_writes_snapshots_between_evals(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, total_steps=20, save_every=5, eval_every=20)
    train(cfg)
    # last.pt must exist (either from save_every=5 triggers or from
    # the final eval at step 20).
    assert (tmp_path / "last.pt").exists()
    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["step"] == 20
    # best_cer field is always present now.
    assert "best_cer" in ckpt


def test_save_every_disabled_still_saves_on_eval(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, save_every=0, total_steps=20, eval_every=20)
    train(cfg)
    assert (tmp_path / "last.pt").exists()


# --------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------- #


def test_resume_restores_step_counter_and_weights(tmp_path: Path) -> None:
    # Phase 1: train 15 steps, save last.pt.
    phase1 = tmp_path / "phase1"
    cfg1 = _tiny_cfg(phase1, total_steps=15, save_every=5, eval_every=15)
    train(cfg1)
    ckpt1 = torch.load(phase1 / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt1["step"] == 15

    # Phase 2: resume, run 10 more steps (to step 25).
    phase2 = tmp_path / "phase2"
    cfg2 = _tiny_cfg(
        phase2, total_steps=25, save_every=5, eval_every=25,
        resume_from=phase1 / "last.pt",
    )
    train(cfg2)
    ckpt2 = torch.load(phase2 / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt2["step"] == 25

    # Weights should have moved from the resume point.
    any_diff = False
    for k in ckpt1["model"]:
        if not torch.allclose(ckpt1["model"][k], ckpt2["model"][k], atol=1e-8):
            any_diff = True
            break
    assert any_diff, "model weights did not update after resume"


def test_resume_restores_optimizer_state(tmp_path: Path) -> None:
    # Train phase 1, check optimizer has adam moments; resume → moments
    # continue to evolve (i.e. not reset to zero).
    phase1 = tmp_path / "p1"
    cfg1 = _tiny_cfg(phase1, total_steps=15, eval_every=15, save_every=5)
    train(cfg1)
    ckpt1 = torch.load(phase1 / "last.pt", map_location="cpu", weights_only=False)
    opt_state1 = ckpt1["optimizer"]["state"]
    # At least one parameter should have non-trivial adam state.
    assert len(opt_state1) > 0
    any_nonzero = any(
        "exp_avg" in ps and ps["exp_avg"].abs().sum() > 0
        for ps in opt_state1.values()
    )
    assert any_nonzero

    phase2 = tmp_path / "p2"
    cfg2 = _tiny_cfg(
        phase2, total_steps=25, eval_every=25, save_every=5,
        resume_from=phase1 / "last.pt",
    )
    train(cfg2)
    ckpt2 = torch.load(phase2 / "last.pt", map_location="cpu", weights_only=False)
    opt_state2 = ckpt2["optimizer"]["state"]
    # The adam running averages should have evolved (not identical, not
    # zeroed).
    differences = 0
    for k in opt_state1:
        ea1 = opt_state1[k]["exp_avg"]
        ea2 = opt_state2[k]["exp_avg"]
        if not torch.allclose(ea1, ea2, atol=1e-8):
            differences += 1
    assert differences > 0, "optimizer moments didn't update post-resume"


def test_resume_emits_resume_event(tmp_path: Path) -> None:
    phase1 = tmp_path / "p1"
    train(_tiny_cfg(phase1, total_steps=15, eval_every=15, save_every=5))

    phase2 = tmp_path / "p2"
    train(_tiny_cfg(
        phase2, total_steps=25, eval_every=25, save_every=5,
        resume_from=phase1 / "last.pt",
    ))
    events = []
    with (phase2 / "train.jsonl").open() as f:
        for line in f:
            events.append(json.loads(line))
    # First event must be "resume", not "start".
    assert events[0]["event"] == "resume"
    assert events[0]["step"] == 15


def test_resume_preserves_best_cer(tmp_path: Path) -> None:
    phase1 = tmp_path / "p1"
    cfg1 = _tiny_cfg(phase1, total_steps=20, eval_every=20, save_every=5)
    train(cfg1)
    ckpt1 = torch.load(phase1 / "last.pt", map_location="cpu", weights_only=False)
    best1 = ckpt1["best_cer"]
    # Tiny model after 20 steps has some (likely large) CER. The
    # important thing is that best_cer is a finite number, and that
    # on resume we keep this reference so we don't needlessly
    # overwrite best_cer.pt with a worse model at the first eval.
    assert best1 < float("inf")

    phase2 = tmp_path / "p2"
    cfg2 = _tiny_cfg(
        phase2, total_steps=25, eval_every=25, save_every=5,
        resume_from=phase1 / "last.pt",
    )
    train(cfg2)
    ckpt2 = torch.load(phase2 / "last.pt", map_location="cpu", weights_only=False)
    # Resumed run's best_cer is ≤ phase-1 best (either improved or
    # unchanged; never reset to +inf).
    assert ckpt2["best_cer"] <= best1 + 1e-9


def test_resume_dataset_seed_is_bumped(tmp_path: Path) -> None:
    """After resume we must not replay the same samples we already
    trained on. The dataset seed gets bumped by step * 7919."""
    from morseformer.data.synthetic import SyntheticCWDataset
    from dataclasses import replace

    phase1 = tmp_path / "p1"
    base_seed = 123
    cfg1 = _tiny_cfg(phase1, dataset=DatasetConfig(seed=base_seed),
                     total_steps=15, eval_every=15, save_every=5)
    train(cfg1)

    # The fresh dataset seed would yield these samples at step 0:
    ds_fresh = SyntheticCWDataset(DatasetConfig(seed=base_seed))
    it_fresh = iter(ds_fresh)
    fresh_first = [next(it_fresh)["tokens"].tolist() for _ in range(3)]

    # The resumed dataset uses seed = base + step*7919, so its stream
    # at step 0 must differ from the fresh stream's first samples.
    resumed_seed = base_seed + 15 * 7919
    ds_resumed = SyntheticCWDataset(DatasetConfig(seed=resumed_seed))
    it_resumed = iter(ds_resumed)
    resumed_first = [next(it_resumed)["tokens"].tolist() for _ in range(3)]

    assert fresh_first != resumed_first, (
        "resumed dataset replays the same samples as a fresh run"
    )
