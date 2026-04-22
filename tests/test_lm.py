"""Smoke tests for the Phase-4 LM: shapes, param counts, dataset
invariants, and a tiny end-to-end training step that must actually
reduce the loss on a trivial target."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from morseformer.core.tokenizer import VOCAB_SIZE
from morseformer.data.lm_dataset import LmDatasetConfig, LmStreamDataset
from morseformer.models.lm import EOS_INDEX, GptLM, LmConfig


def test_forward_shapes() -> None:
    model = GptLM(LmConfig(d_model=64, n_heads=4, n_layers=2))
    toks = torch.randint(0, VOCAB_SIZE, (3, 16))
    logits, loss = model(toks, toks)
    assert logits.shape == (3, 16, VOCAB_SIZE)
    assert loss.dim() == 0


def test_embedding_is_tied() -> None:
    model = GptLM(LmConfig(d_model=32, n_heads=4, n_layers=2))
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_param_count_reasonable() -> None:
    # With the default config (d=256, L=6, heads=4), expect ~5 M params.
    model = GptLM()
    n = model.num_parameters()
    assert 3_000_000 < n < 8_000_000, f"unexpected param count: {n}"


def test_dataset_shapes_and_shift() -> None:
    ds = LmStreamDataset(LmDatasetConfig(context_length=32, seed=123))
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    assert batch["input"].shape == (4, 32)
    assert batch["target"].shape == (4, 32)
    # target[i, :-1] must equal input[i, 1:] (next-token shift)
    assert torch.equal(batch["target"][:, :-1], batch["input"][:, 1:])


def test_dataset_emits_eos_separators() -> None:
    ds = LmStreamDataset(LmDatasetConfig(context_length=128, seed=7))
    loader = DataLoader(ds, batch_size=8)
    batch = next(iter(loader))
    # At 128-token windows × 8 items = 1024 tokens, we expect plenty of
    # EOS — each sample_text() doc is <60 chars typical.
    assert (batch["input"] == EOS_INDEX).sum() > 20


def test_generate_respects_eos_stop() -> None:
    model = GptLM(LmConfig(d_model=32, n_heads=4, n_layers=2))
    prompt = torch.tensor([[EOS_INDEX, EOS_INDEX]])
    out = model.generate(prompt, max_new_tokens=10, stop_on_eos=False)
    # stop_on_eos=False → we get exactly max_new_tokens new tokens.
    assert out.size(1) == prompt.size(1) + 10


def test_one_step_training_reduces_loss() -> None:
    """Overfit a single fixed batch: loss must drop after a handful of
    optimizer steps. Catches broken grad flow, shape mismatches in the
    causal mask, EOS overload mishandling, etc."""
    torch.manual_seed(0)
    model = GptLM(LmConfig(d_model=64, n_heads=4, n_layers=2))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # Fixed batch: two short sequences the model must memorise.
    inp = torch.tensor([[EOS_INDEX, 5, 10, 15, 20, 25, 30, 35]])
    tgt = torch.tensor([[5, 10, 15, 20, 25, 30, 35, EOS_INDEX]])
    _, loss0 = model(inp, tgt)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        _, loss = model(inp, tgt)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * loss0.item(), (
        f"loss did not drop: {loss0.item():.3f} → {loss.item():.3f}"
    )


def test_dataset_train_val_streams_differ() -> None:
    """Training and val streams must produce different token sequences
    when seeded differently — otherwise val perplexity is meaningless."""
    rng_t = np.random.default_rng(0)
    rng_v = np.random.default_rng(0 + 1_000_003)
    # First few tokens from each must differ in at least some positions.
    from morseformer.data.text import sample_text
    from morseformer.core.tokenizer import encode
    t_tokens: list[int] = []
    v_tokens: list[int] = []
    while len(t_tokens) < 100:
        t_tokens.extend(encode(sample_text(rng_t)))
    while len(v_tokens) < 100:
        v_tokens.extend(encode(sample_text(rng_v)))
    assert t_tokens[:100] != v_tokens[:100]
