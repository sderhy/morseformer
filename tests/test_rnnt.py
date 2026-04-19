"""Unit tests for the RNN-T head (prediction net, joint net, full model)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")

from torchaudio.functional import rnnt_loss  # noqa: E402

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE  # noqa: E402
from morseformer.models.acoustic import AcousticConfig  # noqa: E402
from morseformer.models.rnnt import (  # noqa: E402
    JointNetwork,
    PredictionNetwork,
    RnntConfig,
    RnntModel,
)


def _tiny_cfg() -> RnntConfig:
    return RnntConfig(
        encoder=AcousticConfig(
            d_model=32, n_heads=4, n_layers=2,
            ff_expansion=2, conv_kernel=7, dropout=0.0,
        ),
        d_pred=32,
        pred_lstm_layers=1,
        d_joint=32,
        dropout=0.0,
    )


# --------------------------------------------------------------------- #
# Prediction network
# --------------------------------------------------------------------- #


def test_prediction_prepends_sos_and_shifts() -> None:
    pred = PredictionNetwork(vocab_size=VOCAB_SIZE, d_pred=16)
    targets = torch.randint(1, 46, (2, 5), dtype=torch.long)
    out = pred(targets)
    # [B, U + 1, d_pred]
    assert out.shape == (2, 6, 16)
    assert out.dtype == torch.float32


def test_prediction_step_matches_forward_on_teacher_forcing() -> None:
    """Calling step() token-by-token with the same history as forward
    gives the same output sequence."""
    torch.manual_seed(0)
    pred = PredictionNetwork(vocab_size=VOCAB_SIZE, d_pred=16).eval()
    targets = torch.tensor([[2, 5, 3, 7]], dtype=torch.long)
    with torch.no_grad():
        full = pred(targets)  # [1, 5, 16]

    # Step-by-step: blank -> 2 -> 5 -> 3 -> 7
    state = None
    inputs = [BLANK_INDEX, 2, 5, 3, 7]
    step_outs = []
    for tok in inputs:
        tok_t = torch.tensor([[tok]], dtype=torch.long)
        out, state = pred.step(tok_t, state)
        step_outs.append(out)
    stepped = torch.cat(step_outs, dim=1)  # [1, 5, 16]
    torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------- #
# Joint network
# --------------------------------------------------------------------- #


def test_joint_output_shape() -> None:
    joint = JointNetwork(d_enc=16, d_pred=12, d_joint=24, vocab_size=46)
    enc = torch.randn(2, 8, 16)
    pred = torch.randn(2, 5, 12)
    out = joint(enc, pred)
    assert out.shape == (2, 8, 5, 46)


# --------------------------------------------------------------------- #
# Full RNN-T model
# --------------------------------------------------------------------- #


def test_rnnt_forward_shapes() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).eval()
    features = torch.randn(2, 200, 1)
    targets = torch.randint(1, 46, (2, 4), dtype=torch.long)
    lengths = torch.tensor([200, 160], dtype=torch.long)
    out = model(features, targets, lengths)
    assert out["enc_out"].shape[:2] == (2, 50)          # 200 / 4
    assert out["enc_out"].shape[2] == cfg.encoder.d_model
    assert out["ctc_log_probs"].shape == (2, 50, cfg.vocab_size)
    assert out["joint_logits"].shape == (2, 50, 5, cfg.vocab_size)
    assert out["enc_lengths"].tolist() == [50, 40]


def test_ctc_loss_pathway_runs() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).train()
    features = torch.randn(2, 200, 1)
    targets = torch.tensor([[2, 5, 3], [7, 1, 9]], dtype=torch.long)
    lengths = torch.tensor([200, 200], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)

    out = model(features, targets, lengths)
    flat = torch.cat([targets[i, : target_lengths[i]] for i in range(2)])
    loss = torch.nn.functional.ctc_loss(
        out["ctc_log_probs"].transpose(0, 1),
        flat,
        out["enc_lengths"],
        target_lengths,
        blank=BLANK_INDEX,
        zero_infinity=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # CTC must flow gradient into the encoder and the CTC head.
    assert model.acoustic.head.weight.grad is not None
    assert model.acoustic.subsample.conv[0].weight.grad is not None


def test_rnnt_loss_pathway_runs() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).train()
    features = torch.randn(2, 200, 1)
    targets = torch.tensor([[2, 5, 3], [7, 1, 9]], dtype=torch.long)
    lengths = torch.tensor([200, 200], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)

    out = model(features, targets, lengths)
    loss = rnnt_loss(
        out["joint_logits"].float(),
        targets.int(),
        out["enc_lengths"].int(),
        target_lengths.int(),
        blank=BLANK_INDEX,
    )
    assert torch.isfinite(loss)
    loss.backward()
    # Gradients must reach the prediction and joint networks.
    assert model.pred.embed.weight.grad is not None
    assert model.joint.enc_proj.weight.grad is not None
    assert model.joint.pred_proj.weight.grad is not None


def test_multi_task_combined_loss() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).train()
    features = torch.randn(2, 200, 1)
    targets = torch.tensor([[2, 5, 3], [7, 1, 9]], dtype=torch.long)
    lengths = torch.tensor([200, 200], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)

    out = model(features, targets, lengths)
    flat = torch.cat([targets[i, : target_lengths[i]] for i in range(2)])
    ctc = torch.nn.functional.ctc_loss(
        out["ctc_log_probs"].transpose(0, 1),
        flat,
        out["enc_lengths"],
        target_lengths,
        blank=BLANK_INDEX,
        zero_infinity=True,
    )
    rnnt = rnnt_loss(
        out["joint_logits"].float(),
        targets.int(),
        out["enc_lengths"].int(),
        target_lengths.int(),
        blank=BLANK_INDEX,
    )
    total = 0.3 * ctc + 0.7 * rnnt
    assert torch.isfinite(total)
    total.backward()
    # Encoder receives gradient from both heads.
    assert model.acoustic.subsample.conv[0].weight.grad is not None


def test_greedy_rnnt_decode_returns_lists_of_ints() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).eval()
    features = torch.randn(3, 200, 1)
    lengths = torch.tensor([200, 160, 120], dtype=torch.long)
    hyps = model.greedy_rnnt_decode(features, lengths)
    assert len(hyps) == 3
    for h in hyps:
        assert isinstance(h, list)
        for tok in h:
            assert isinstance(tok, int)
            # Blank is never emitted.
            assert tok != BLANK_INDEX


def test_greedy_decode_respects_length_caps() -> None:
    cfg = _tiny_cfg()
    model = RnntModel(cfg).eval()
    features = torch.randn(1, 200, 1)
    lengths = torch.tensor([120], dtype=torch.long)
    hyps = model.greedy_rnnt_decode(
        features, lengths, max_emit_per_frame=3
    )
    # T' after subsample = 30; cap = 30 * 3 = 90 tokens.
    assert len(hyps[0]) <= 30 * 3


def test_load_encoder_state_dict_roundtrip() -> None:
    """A Phase-2 acoustic checkpoint drops directly into the RNN-T
    model's encoder + CTC head."""
    cfg = _tiny_cfg()
    phase2 = torch.nn.Module()  # placeholder, we need AcousticModel
    from morseformer.models.acoustic import AcousticModel
    src = AcousticModel(cfg.encoder)
    # Randomise to make the test meaningful.
    with torch.no_grad():
        for p in src.parameters():
            p.normal_(0, 0.1)
    rnnt = RnntModel(cfg)
    rnnt.load_encoder_state_dict(src.state_dict())
    for (_, p_src), (_, p_dst) in zip(
        src.named_parameters(), rnnt.acoustic.named_parameters()
    ):
        assert torch.allclose(p_src, p_dst)


def test_parameter_count_reasonable() -> None:
    """Default RNN-T config should be a few percent above the encoder."""
    cfg = RnntConfig()  # uses defaults — Phase 2 encoder
    m = RnntModel(cfg)
    counts = m.num_parameters_by_module()
    enc = counts["acoustic (encoder + CTC head)"]
    pred = counts["prediction"]
    joint = counts["joint"]
    # Encoder dominates; pred + joint combined are <10% of encoder.
    assert pred + joint < enc * 0.15
    # Total is in the 4–5 M range for the default config.
    assert 3_500_000 < m.num_parameters() < 5_000_000
