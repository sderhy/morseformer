"""Smoke tests for AMP precisions: float32 / bfloat16 / float16.

These all run on whatever device is available (CPU or CUDA). bfloat16
on CPU is supported but slow; float16 on CPU is supported for autocast
only on recent PyTorch versions but not universally, so the float16 test
is skipped unless CUDA is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from morseformer.data.synthetic import DatasetConfig  # noqa: E402
from morseformer.data.validation import ValidationConfig  # noqa: E402
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.train.acoustic import TrainConfig, train  # noqa: E402


def _tiny_cfg(tmp_path: Path, dtype: str, device: str) -> TrainConfig:
    return TrainConfig(
        model=AcousticConfig(
            d_model=16, n_heads=2, n_layers=1,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        dataset=DatasetConfig(seed=0),
        validation=ValidationConfig(n_per_wpm=1, seed=1),
        warmup_steps=5,
        total_steps=15,
        batch_size=4,
        log_every=5,
        eval_every=15,
        checkpoint_dir=tmp_path,
        jsonl_log=tmp_path / "train.jsonl",
        dtype=dtype,
        device=device,
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_amp_dtype_trains_without_crashing(dtype: str, tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, dtype=dtype, device="cpu")
    result = train(cfg)
    assert result["steps"] == 15
    assert (tmp_path / "last.pt").exists()
    # Checkpoint is always float32 because master weights are kept in fp32.
    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    sample_param = next(iter(ckpt["model"].values()))
    assert sample_param.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_float16_trains_on_cuda(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, dtype="float16", device="cuda")
    result = train(cfg)
    assert result["steps"] == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bfloat16_on_cuda_produces_finite_loss(tmp_path: Path) -> None:
    """bf16 autocast must produce finite, non-NaN training losses and
    leave the model in a trainable state. We do *not* compare the loss
    curve to fp32 at a fixed step count — 15 steps is far too early for
    the two precisions to agree numerically, and asserting similarity
    would flake regularly."""
    import json
    import math

    cfg = _tiny_cfg(tmp_path, dtype="bfloat16", device="cuda")
    cfg.dataset.seed = 1234
    train(cfg)

    losses: list[float] = []
    with (tmp_path / "train.jsonl").open() as f:
        for line in f:
            evt = json.loads(line)
            if evt.get("event") == "step":
                losses.append(evt["loss"])
    assert len(losses) > 0
    for loss in losses:
        assert math.isfinite(loss), f"non-finite loss under bf16: {loss}"
        assert loss > 0.0

    # Parameters should still be finite after training.
    ckpt = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    for name, t in ckpt["model"].items():
        assert torch.isfinite(t).all(), f"non-finite weights in {name}"


def test_bad_dtype_rejected(tmp_path: Path) -> None:
    cfg = _tiny_cfg(tmp_path, dtype="int8", device="cpu")
    with pytest.raises(ValueError):
        train(cfg)
