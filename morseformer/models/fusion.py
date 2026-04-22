"""Shallow fusion of the RNN-T acoustic model with the Phase-4 LM.

Shallow fusion combines the per-step log-probabilities of a trained
acoustic model with those of a separately-trained language model at
decode time:

    log P_fused(y_u | x, y_{<u}) =
          log P_ac(y_u | x, y_{<u}) + λ · log P_lm(y_u | y_{<u})

with ``λ`` tuned on a held-out set. The blank emission is *not*
rescored — the LM does not model the acoustic "stay at this frame"
signal, and the RNN-T prediction network already encodes the
non-blank history the LM would need.

Reference: Gulati et al. 2020 (Conformer), section 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from morseformer.core.tokenizer import BLANK_INDEX
from morseformer.models.lm import EOS_INDEX, GptLM
from morseformer.models.rnnt import RnntModel


@dataclass
class FusionConfig:
    """Shallow-fusion hyperparameters."""

    fusion_weight: float = 0.2
    # Maximum tokens to emit at a single encoder frame — same as the
    # RNN-T greedy decoder default, prevents infinite emit loops.
    max_emit_per_frame: int = 5
    # Optional per-emission LM temperature. Leaving at 1.0 uses the
    # LM's calibrated distribution; raising it flattens the prior and
    # is a useful knob when the LM over-commits.
    lm_temperature: float = 1.0


@torch.no_grad()
def greedy_rnnt_decode_with_lm(
    rnnt: RnntModel,
    lm: GptLM,
    features: torch.Tensor,
    lengths: torch.Tensor | None = None,
    fusion_cfg: FusionConfig | None = None,
) -> list[list[int]]:
    """Greedy RNN-T decoding with a shallow-fusion LM prior.

    Differs from :meth:`RnntModel.greedy_rnnt_decode` only in how the
    per-(t, u) logits are scored: the LM's log-probabilities are added
    to every non-blank slot, weighted by ``fusion_cfg.fusion_weight``.

    Args:
        rnnt: trained :class:`RnntModel`. Caller is responsible for
              placing it in ``eval()`` mode and on the right device.
        lm: trained :class:`GptLM`, same device as ``rnnt``. Must share
            the 46-token vocabulary.
        features: ``[B, T, F]`` float32 input audio features.
        lengths:  ``[B]`` valid frame counts, or ``None``.
        fusion_cfg: fusion weight and emission limits (defaults applied
                    when ``None``).

    Returns:
        ``list[list[int]]`` — one list of non-blank token ids per batch
        item, same format as the vanilla RNN-T greedy decoder.
    """
    cfg = fusion_cfg or FusionConfig()
    rnnt.eval()
    lm.eval()

    if rnnt.cfg.vocab_size != lm.cfg.vocab_size:
        raise ValueError(
            f"vocab-size mismatch: rnnt={rnnt.cfg.vocab_size} "
            f"lm={lm.cfg.vocab_size}"
        )

    enc_out, enc_lengths = rnnt.acoustic.encode(features, lengths)
    b, t_max, _ = enc_out.shape
    if enc_lengths is None:
        enc_lengths = torch.full(
            (b,), t_max, device=enc_out.device, dtype=torch.long
        )

    blank = rnnt.cfg.blank_index
    vocab_size = rnnt.cfg.vocab_size
    results: list[list[int]] = []

    # Build a [V]-shaped mask that is 0 on the blank slot, 1 elsewhere.
    # We use it to zero out the LM contribution on the blank emission,
    # so argmax for blank is driven purely by the acoustic joint.
    non_blank_mask = torch.ones(vocab_size, device=enc_out.device)
    non_blank_mask[blank] = 0.0

    for i in range(b):
        t_valid = int(enc_lengths[i].item())
        hyp: list[int] = []
        # RNN-T prediction-network state — advances on every non-blank
        # emission, same as the vanilla decoder.
        pred_state: tuple[torch.Tensor, torch.Tensor] | None = None
        prev_token = torch.tensor(
            [[blank]], dtype=torch.long, device=enc_out.device
        )
        pred_out, pred_state = rnnt.pred.step(prev_token, pred_state)

        for t in range(t_valid):
            emitted = 0
            while emitted < cfg.max_emit_per_frame:
                f = enc_out[i : i + 1, t : t + 1, :]         # [1, 1, d_enc]
                logits = rnnt.joint(f, pred_out)             # [1, 1, 1, V]
                ac_logprobs = F.log_softmax(
                    logits[0, 0, 0].float(), dim=-1
                )                                            # [V]

                # Feed the emitted history (with BOS=EOS=blank sentinel)
                # to the LM and take log-probs at the final position.
                # Context length is bounded by ``hyp`` which can only
                # grow by ``max_emit_per_frame`` tokens per frame.
                lm_input = torch.tensor(
                    [[EOS_INDEX] + hyp], dtype=torch.long,
                    device=enc_out.device,
                )
                lm_logits, _ = lm(lm_input)                  # [1, L, V]
                lm_logprobs = F.log_softmax(
                    lm_logits[0, -1, :].float() / max(cfg.lm_temperature, 1e-6),
                    dim=-1,
                )                                            # [V]

                # Add weighted LM prior to non-blank slots only.
                fused = ac_logprobs + cfg.fusion_weight * (
                    lm_logprobs * non_blank_mask
                )
                tok = int(fused.argmax().item())
                if tok == blank:
                    break
                hyp.append(tok)
                prev_token = torch.tensor(
                    [[tok]], dtype=torch.long, device=enc_out.device
                )
                pred_out, pred_state = rnnt.pred.step(prev_token, pred_state)
                emitted += 1
        results.append(hyp)
    return results


@torch.no_grad()
def greedy_ctc_decode_with_lm(
    rnnt: RnntModel,
    lm: GptLM,
    features: torch.Tensor,
    lengths: torch.Tensor | None = None,
    fusion_cfg: FusionConfig | None = None,
) -> list[list[int]]:
    """Greedy CTC decoding with shallow-fusion LM prior, sharing the
    encoder of an RNN-T model (``rnnt.acoustic``).

    CTC's frame-wise decoding makes LM fusion subtler than for RNN-T
    because the LM should only be consulted when the frame emits a
    *new* non-blank token — otherwise the LM is asked to predict the
    same character twice in a row (since CTC repeats stand for
    "extend the current element"). We handle this by tracking the
    collapsed emission history and adding the LM prior only at the
    frames where the argmax would collapse to a fresh token.

    Returns ``list[list[int]]`` of non-blank, non-repeated token ids.
    """
    cfg = fusion_cfg or FusionConfig()
    rnnt.eval()
    lm.eval()

    enc_out, enc_lengths = rnnt.acoustic.encode(features, lengths)
    ctc_logits = rnnt.acoustic.head(enc_out)                     # [B, T', V]
    b, t_max, vocab_size = ctc_logits.shape
    if enc_lengths is None:
        enc_lengths = torch.full(
            (b,), t_max, device=ctc_logits.device, dtype=torch.long
        )

    blank = rnnt.cfg.blank_index
    results: list[list[int]] = []

    for i in range(b):
        t_valid = int(enc_lengths[i].item())
        hyp: list[int] = []
        prev_tok = -1
        for t in range(t_valid):
            ac_logprobs = F.log_softmax(
                ctc_logits[i, t, :].float(), dim=-1
            )
            # Tentative frame-wise argmax without fusion (to decide
            # whether this frame introduces a new token; if not, the
            # LM should not be consulted).
            tentative = int(ac_logprobs.argmax().item())

            if tentative == blank or tentative == prev_tok:
                tok = tentative
            else:
                lm_input = torch.tensor(
                    [[EOS_INDEX] + hyp], dtype=torch.long,
                    device=ctc_logits.device,
                )
                lm_logits, _ = lm(lm_input)
                lm_logprobs = F.log_softmax(
                    lm_logits[0, -1, :].float() / max(cfg.lm_temperature, 1e-6),
                    dim=-1,
                )
                # Mask out blank + previous token in the LM prior so
                # fusion cannot reroute to a CTC-illegal emission.
                non_blank_mask = torch.ones(vocab_size, device=ctc_logits.device)
                non_blank_mask[blank] = 0.0
                fused = ac_logprobs + cfg.fusion_weight * (
                    lm_logprobs * non_blank_mask
                )
                tok = int(fused.argmax().item())

            if tok != blank and tok != prev_tok:
                hyp.append(tok)
            prev_tok = tok
        results.append(hyp)
    return results
