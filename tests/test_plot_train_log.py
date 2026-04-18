"""Tests for the training-curve plot utility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.plot_train_log import load_events, main


def _write_log(path: Path) -> None:
    events = [
        {"event": "start", "config": {"x": 1}, "wall_s": 0.0},
        {"event": "step", "step": 50, "loss": 5.0, "lr": 1e-4,
         "grad_norm": 0.8, "wall_s": 1.0},
        {"event": "step", "step": 100, "loss": 4.0, "lr": 3e-4,
         "grad_norm": 0.6, "wall_s": 2.0},
        {"event": "eval", "step": 100, "cer": 0.5, "wer": 0.9,
         "per_wpm_cer": {"20.0": 0.4, "25.0": 0.6}, "n_samples": 10,
         "wall_s": 2.5},
        {"event": "step", "step": 150, "loss": 3.0, "lr": 2e-4,
         "grad_norm": 0.4, "wall_s": 3.0},
    ]
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def test_load_events_parses_all_kinds(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    _write_log(log)
    data = load_events(log)
    assert data["steps"] == [50, 100, 150]
    assert data["losses"] == [5.0, 4.0, 3.0]
    assert data["eval_steps"] == [100]
    assert data["eval_cers"] == [0.5]
    assert sorted(data["per_wpm"]) == [20.0, 25.0]
    assert data["start"] == {"x": 1}


def test_main_writes_output(tmp_path: Path) -> None:
    log = tmp_path / "train.jsonl"
    out = tmp_path / "curves.png"
    _write_log(log)

    # Use Agg backend to avoid needing a display.
    import matplotlib
    matplotlib.use("Agg")

    rc = main(["--log", str(log), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    assert out.stat().st_size > 1000  # real PNG, not empty


def test_main_missing_log_returns_error(tmp_path: Path) -> None:
    assert main(["--log", str(tmp_path / "nope.jsonl")]) == 2


def test_main_empty_log_returns_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["--log", str(empty)]) == 1
