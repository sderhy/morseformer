"""Smoke test for the release-gate harness.

Builds a tiny RNN-T checkpoint, points the model registry at it,
runs ``release_gate.run_gate`` against a minimal manifest (silence +
latency only — no manifest_clip categories so we don't need the real
bench WAVs), and asserts the harness:

* returns one :class:`CategoryResult` per manifest entry,
* writes a well-formed JSON report,
* maps the verdict to the correct exit code (0 / 1) when invoked
  through ``main()``.

Verdict correctness is asserted with synthetic thresholds — a tiny
randomly-initialised model is not expected to match real ship gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.models.rnnt import RnntConfig, RnntModel  # noqa: E402


def _make_tiny_checkpoint(path: Path) -> None:
    """Save a checkpoint that ``eval.bench_lcwo._load_rnnt`` can ingest."""
    enc_cfg = AcousticConfig(
        d_model=16, n_heads=2, n_layers=1,
        ff_expansion=2, conv_kernel=7, dropout=0.0,
    )
    rnnt_cfg = RnntConfig(
        encoder=enc_cfg, d_pred=16, pred_lstm_layers=1,
        d_joint=16, dropout=0.0,
    )
    model = RnntModel(rnnt_cfg)
    torch.save(
        {
            "step": 0,
            "model": model.state_dict(),
            "ema": None,
            "optimizer": {},
            "scheduler": {},
            "best_ctc_cer": 1.0,
            "best_rnnt_cer": 1.0,
            "metrics": None,
            "config": {
                "model": {
                    "encoder": {
                        "d_model": enc_cfg.d_model,
                        "n_heads": enc_cfg.n_heads,
                        "n_layers": enc_cfg.n_layers,
                        "ff_expansion": enc_cfg.ff_expansion,
                        "conv_kernel": enc_cfg.conv_kernel,
                        "dropout": enc_cfg.dropout,
                    },
                    "d_pred": rnnt_cfg.d_pred,
                    "pred_lstm_layers": rnnt_cfg.pred_lstm_layers,
                    "d_joint": rnnt_cfg.d_joint,
                    "dropout": rnnt_cfg.dropout,
                    "vocab_size": rnnt_cfg.vocab_size,
                }
            },
        },
        path,
    )


def _mini_manifest() -> dict:
    """Manifest with only silence + latency — both run without external
    bench WAVs and finish in seconds on CPU."""
    return {
        "version": 1,
        "baseline_acoustic": "tiny_test",
        "calibration_date": "2026-05-19",
        "non_regression_margin_pp": 0.5,
        "categories": [
            {
                "id": "silence_smoke",
                "kind": "synthetic_silence",
                "preset": "live",
                "n_samples": 2,
                "wpm_bins": [20.0],
                "seed": 1,
                "baseline_chars_per_sample": 0.0,
                "max_chars_per_sample": 1000.0,
            },
            {
                "id": "latency_smoke",
                "kind": "latency",
                "window_seconds": 6.0,
                "hop_seconds": 2.0,
                "warmup": 1,
                "trials": 2,
                "max_rtf": 1e6,
            },
        ],
    }


def test_run_gate_returns_one_result_per_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt_path)

    import eval.release_gate as rg

    monkeypatch.setattr(rg, "resolve_model", lambda _name: ckpt_path)

    manifest = _mini_manifest()
    results, summary = rg.run_gate(
        manifest,
        acoustic_name="tiny_test",
        lm_name=None,
        bench_manifest_path=tmp_path / "empty_manifest.jsonl",
        device=torch.device("cpu"),
    )
    assert len(results) == len(manifest["categories"])
    ids = [r.id for r in results]
    assert ids == ["silence_smoke", "latency_smoke"]
    assert summary["acoustic"] == "tiny_test"
    assert summary["n_categories"] == 2
    assert summary["n_passed"] + summary["n_failed"] == 2
    # Generous thresholds in the mini manifest → all categories pass.
    assert summary["all_passed"]


def test_run_gate_records_failure_when_threshold_exceeded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt_path)

    import eval.release_gate as rg
    monkeypatch.setattr(rg, "resolve_model", lambda _name: ckpt_path)

    manifest = _mini_manifest()
    # Force the latency gate to fail by setting an absurdly low cap.
    for cat in manifest["categories"]:
        if cat["id"] == "latency_smoke":
            cat["max_rtf"] = -1.0
    results, summary = rg.run_gate(
        manifest,
        acoustic_name="tiny_test",
        lm_name=None,
        bench_manifest_path=tmp_path / "empty_manifest.jsonl",
        device=torch.device("cpu"),
    )
    failed = [r for r in results if not r.passed]
    assert any(r.id == "latency_smoke" for r in failed)
    assert summary["all_passed"] is False
    assert summary["n_failed"] >= 1


def test_main_writes_json_report_and_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    ckpt_path = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt_path)

    manifest_path = tmp_path / "mini_manifest.json"
    manifest_path.write_text(
        json.dumps(_mini_manifest()), encoding="utf-8"
    )

    import eval.release_gate as rg
    monkeypatch.setattr(rg, "resolve_model", lambda _name: ckpt_path)

    out_dir = tmp_path / "reports"
    rc = rg.main([
        "--manifest", str(manifest_path),
        "--acoustic", "tiny_test",
        "--device", "cpu",
        "--out-dir", str(out_dir),
        "--bench-manifest", str(tmp_path / "empty.jsonl"),
    ])
    assert rc == 0  # generous thresholds → pass
    written = list(out_dir.glob("release_gate_tiny_test_*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["summary"]["acoustic"] == "tiny_test"
    assert payload["summary"]["all_passed"] is True
    assert len(payload["categories"]) == 2
    for entry in payload["categories"]:
        for key in ("id", "kind", "metric", "measured",
                    "threshold", "passed", "extra"):
            assert key in entry


def test_main_exit_code_is_one_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_path = tmp_path / "tiny.pt"
    _make_tiny_checkpoint(ckpt_path)

    manifest = _mini_manifest()
    for cat in manifest["categories"]:
        if cat["id"] == "latency_smoke":
            cat["max_rtf"] = -1.0
    manifest_path = tmp_path / "mini_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    import eval.release_gate as rg
    monkeypatch.setattr(rg, "resolve_model", lambda _name: ckpt_path)

    rc = rg.main([
        "--manifest", str(manifest_path),
        "--acoustic", "tiny_test",
        "--device", "cpu",
        "--out-dir", str(tmp_path / "reports"),
        "--bench-manifest", str(tmp_path / "empty.jsonl"),
    ])
    assert rc == 1
