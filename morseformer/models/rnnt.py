"""Recurrent Neural Network Transducer (RNN-T) head for morseformer.

Structure
---------
An RNN-T model wraps the existing Conformer acoustic encoder and adds
two small modules on top:

    encoder     :  [B, T, F]       →  [B, T', d_enc]    (AcousticModel.encode)
    prediction  :  [B, U + 1]      →  [B, U + 1, d_pred]
    joint       :  (enc, pred)     →  [B, T', U + 1, V]

The joint network's output is fed to ``torchaudio.functional.rnnt_loss``
during training. Inference is greedy RNN-T decoding: for each frame
t, emit tokens until the predicted blank, then advance to t + 1.

We also keep the CTC head from Phase 2 on the *same* encoder output
and train both heads jointly (multi-task). CTC provides a strong
framewise signal that stabilises early training; RNN-T adds a
sequence-level prior that reduces hallucination on weak signal. At
inference time either head can be used standalone — at low SNR
RNN-T typically wins, at high SNR they agree.

Reference: Graves, 2012 — *Sequence Transduction with Recurrent
Neural Networks* (https://arxiv.org/abs/1211.3711).
Implementation notes follow the Conformer (Gulati et al. 2020)
hybrid-CTC/RNN-T recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from morseformer.core.tokenizer import BLANK_INDEX, VOCAB_SIZE
from morseformer.models.acoustic import AcousticConfig, AcousticModel
from morseformer.models.conformer import init_parameters


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class RnntConfig:
    """Hyperparameters for the RNN-T model (encoder + prediction + joint).

    The encoder config is reused verbatim from Phase 2 so a pretrained
    acoustic checkpoint can be loaded as initial weights via
    :meth:`RnntModel.load_encoder_state_dict`.
    """

    encoder: AcousticConfig = field(default_factory=AcousticConfig)
    # Prediction network width. 128 is ample for the 46-token vocab.
    d_pred: int = 128
    pred_lstm_layers: int = 1
    # Joint-network hidden width. Kept modest — the heavy lifting is in
    # the encoder; the joint is just a combiner.
    d_joint: int = 256
    vocab_size: int = VOCAB_SIZE
    blank_index: int = BLANK_INDEX
    dropout: float = 0.1


# --------------------------------------------------------------------- #
# Prediction network
# --------------------------------------------------------------------- #


class PredictionNetwork(nn.Module):
    """Stateful prediction network — one LSTM layer over target tokens.

    During training we feed ``[<blank>] + target`` so the network sees
    the whole text history with a left-shift. The output
    ``[B, U + 1, d_pred]`` is then broadcast over encoder frames in the
    joint.
    """

    def __init__(
        self,
        vocab_size: int,
        d_pred: int,
        n_layers: int = 1,
        blank_index: int = BLANK_INDEX,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_pred)
        self.lstm = nn.LSTM(
            d_pred,
            d_pred,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.blank_index = blank_index

    def forward(
        self, targets: torch.Tensor
    ) -> torch.Tensor:
        """Encode the target history.

        Args:
            targets: ``[B, U]`` int64 — *without* the leading blank.

        Returns:
            ``[B, U + 1, d_pred]`` — embedding at position 0 is the
            <blank> SOS token, position u+1 conditions on target[:u+1].
        """
        b = targets.size(0)
        sos = torch.full(
            (b, 1), self.blank_index, dtype=targets.dtype, device=targets.device
        )
        inputs = torch.cat((sos, targets), dim=1)        # [B, U + 1]
        x = self.embed(inputs)                            # [B, U + 1, d_pred]
        out, _ = self.lstm(x)                             # [B, U + 1, d_pred]
        return out

    def step(
        self,
        token: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step API for greedy / beam-search inference.

        Args:
            token: ``[B, 1]`` int64 — the previous emitted token
                   (use ``blank_index`` for the SOS position).
            state: previous ``(h, c)`` LSTM state or ``None`` to start.

        Returns:
            (output ``[B, 1, d_pred]``, new state).
        """
        x = self.embed(token)                             # [B, 1, d_pred]
        out, new_state = self.lstm(x, state)
        return out, new_state


# --------------------------------------------------------------------- #
# Joint network
# --------------------------------------------------------------------- #


class JointNetwork(nn.Module):
    """Merge encoder and prediction outputs into per-(t, u) vocab logits."""

    def __init__(
        self,
        d_enc: int,
        d_pred: int,
        d_joint: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(d_enc, d_joint, bias=False)
        self.pred_proj = nn.Linear(d_pred, d_joint, bias=False)
        self.out = nn.Linear(d_joint, vocab_size)

    def forward(
        self, enc_out: torch.Tensor, pred_out: torch.Tensor
    ) -> torch.Tensor:
        """Compute the joint-network logits for every (t, u) pair.

        Args:
            enc_out:  ``[B, T, d_enc]``  encoder features.
            pred_out: ``[B, U + 1, d_pred]`` prediction features.

        Returns:
            logits: ``[B, T, U + 1, V]``.
        """
        f = self.enc_proj(enc_out).unsqueeze(2)          # [B, T, 1, d_joint]
        g = self.pred_proj(pred_out).unsqueeze(1)        # [B, 1, U + 1, d_joint]
        combined = torch.tanh(f + g)                      # [B, T, U + 1, d_joint]
        return self.out(combined)                         # [B, T, U + 1, V]


# --------------------------------------------------------------------- #
# Top-level RNN-T model
# --------------------------------------------------------------------- #


class RnntModel(nn.Module):
    """Acoustic encoder + CTC head + RNN-T prediction and joint networks.

    Forward returns everything needed for the multi-task training
    objective (CTC on encoder, RNN-T on joint). The two heads share
    the encoder so the CTC gradient flows into the same trunk.
    """

    def __init__(self, cfg: RnntConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RnntConfig()
        c = self.cfg

        # Encoder + its own CTC head reused from Phase 2.
        self.acoustic = AcousticModel(c.encoder)

        self.pred = PredictionNetwork(
            vocab_size=c.vocab_size,
            d_pred=c.d_pred,
            n_layers=c.pred_lstm_layers,
            blank_index=c.blank_index,
            dropout=c.dropout,
        )
        self.joint = JointNetwork(
            d_enc=c.encoder.d_model,
            d_pred=c.d_pred,
            d_joint=c.d_joint,
            vocab_size=c.vocab_size,
        )

        # Re-init the new submodules with the same scheme as the encoder
        # for consistency. `acoustic` was already initialised by its own
        # __init__.
        init_parameters(self.pred)
        init_parameters(self.joint)

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict:
        """Run encoder + CTC head + RNN-T joint in one pass.

        Args:
            features: ``[B, T, F]`` float32 input.
            targets:  ``[B, U]`` int64 target tokens (no blank padding
                      inside, but the collate-pad value is the blank
                      index — the RNN-T loss' ``target_lengths`` argument
                      masks the padding out).
            lengths:  ``[B]`` valid input frames, or ``None`` for
                      all-full batches.

        Returns:
            Dict with keys:
              ``enc_out``     : ``[B, T', d_enc]`` encoder features.
              ``enc_lengths`` : ``[B]`` post-subsample lengths, or None.
              ``ctc_log_probs``: ``[B, T', V]`` log-softmax for CTC.
              ``joint_logits`` : ``[B, T', U + 1, V]`` for RNN-T loss.
        """
        enc_out, enc_lengths = self.acoustic.encode(features, lengths)
        # CTC head is the same Linear the Phase 2 model uses.
        ctc_logits = self.acoustic.head(enc_out)
        ctc_log_probs = torch.log_softmax(ctc_logits, dim=-1)

        pred_out = self.pred(targets)
        joint_logits = self.joint(enc_out, pred_out)

        return {
            "enc_out": enc_out,
            "enc_lengths": enc_lengths,
            "ctc_log_probs": ctc_log_probs,
            "joint_logits": joint_logits,
        }

    @torch.no_grad()
    def greedy_rnnt_decode(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_emit_per_frame: int = 5,
    ) -> list[list[int]]:
        """Greedy RNN-T decoding: for each encoder frame, emit tokens
        until the joint predicts blank, then advance.

        Returns one list of non-blank token indices per batch item.
        """
        aligned = self.greedy_rnnt_decode_aligned(
            features, lengths, max_emit_per_frame=max_emit_per_frame
        )
        return [[tok for tok, _ in h] for h in aligned]

    @torch.no_grad()
    def greedy_rnnt_decode_aligned(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_emit_per_frame: int = 5,
    ) -> list[list[tuple[int, int]]]:
        """Greedy RNN-T decoding that also returns the encoder frame
        index at which each non-blank token was emitted.

        Each emission is tagged with the encoder frame it was emitted on.
        Multiple tokens emitted at the same frame share that frame index.
        Frame indices are monotonically non-decreasing within a hypothesis.

        Used by the streaming decoder to convert per-token emissions into
        absolute audio timestamps so a sliding-window loop can commit only
        the central zone of each window without re-emitting tokens.

        Returns a list (one per batch item) of ``[(token, frame_idx), ...]``.
        """
        self.eval()
        enc_out, enc_lengths = self.acoustic.encode(features, lengths)
        b, t_max, _ = enc_out.shape
        if enc_lengths is None:
            enc_lengths = torch.full((b,), t_max, device=enc_out.device, dtype=torch.long)

        blank = self.cfg.blank_index
        results: list[list[tuple[int, int]]] = []
        for i in range(b):
            t_valid = int(enc_lengths[i].item())
            hyp: list[tuple[int, int]] = []
            state: tuple[torch.Tensor, torch.Tensor] | None = None
            prev_token = torch.tensor(
                [[blank]], dtype=torch.long, device=enc_out.device
            )
            pred_out, state = self.pred.step(prev_token, state)  # [1, 1, d_pred]
            for t in range(t_valid):
                emitted = 0
                while emitted < max_emit_per_frame:
                    f = enc_out[i : i + 1, t : t + 1, :]          # [1, 1, d_enc]
                    logits = self.joint(f, pred_out)               # [1, 1, 1, V]
                    tok = int(logits[0, 0, 0].argmax().item())
                    if tok == blank:
                        break
                    hyp.append((tok, t))
                    prev_token = torch.tensor(
                        [[tok]], dtype=torch.long, device=enc_out.device
                    )
                    pred_out, state = self.pred.step(prev_token, state)
                    emitted += 1
            results.append(hyp)
        return results

    def load_encoder_state_dict(
        self, acoustic_state: dict[str, torch.Tensor]
    ) -> None:
        """Load a pretrained Phase-2 AcousticModel checkpoint into the
        encoder (+ CTC head) of this RNN-T model.

        Use this to bootstrap Phase 3 training from a converged
        Phase 2.x encoder so we're not learning acoustic features from
        scratch.
        """
        self.acoustic.load_state_dict(acoustic_state)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_by_module(self) -> dict[str, int]:
        return {
            "acoustic (encoder + CTC head)":
                sum(p.numel() for p in self.acoustic.parameters() if p.requires_grad),
            "prediction":
                sum(p.numel() for p in self.pred.parameters() if p.requires_grad),
            "joint":
                sum(p.numel() for p in self.joint.parameters() if p.requires_grad),
        }
