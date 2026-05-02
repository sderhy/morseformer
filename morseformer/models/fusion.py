"""External-LM fusion for the Phase-3 RNN-T acoustic model.

Two fusion schemes are implemented, selected by the fusion weights:

* **Shallow fusion** (``ilm_weight = 0``): the baseline approach,

      log P_fused(y_u | x, y_{<u}) =
          log P_ac(y_u | x, y_{<u}) + λ_lm · log P_lm(y_u | y_{<u})

  on non-blank slots only. Reference: Gulati et al. 2020 (Conformer),
  section 3.6.

* **Density-ratio / ILME fusion** (``ilm_weight > 0``): subtracts an
  estimate of the RNN-T's *internal* LM before adding the external LM,

      log P_fused = log P_ac + λ_lm · log P_lm − λ_ilm · log P_ilm

  on non-blank slots. The internal LM is estimated by running the joint
  network with a zeroed encoder frame — this leaves only the
  prediction-network contribution, which is the RNN-T's implicit text
  prior. Subtracting it avoids the "ILM contamination" failure mode in
  which naive shallow fusion double-counts the text prior and wipes out
  the external LM's gain. Reference: Meng et al. 2021,
  *Internal Language Model Estimation for Domain-Adaptive End-to-End
  Speech Recognition* (arXiv:2011.01991).

The blank emission is driven purely by the acoustic joint in both
schemes — the LM does not model the "stay at this frame" signal.
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
    """Fusion hyperparameters.

    ``fusion_weight`` (a.k.a. ``λ_lm``) scales the external-LM prior.
    ``ilm_weight`` (a.k.a. ``λ_ilm``) scales the internal-LM estimate
    that is *subtracted* to cancel the RNN-T prediction network's own
    text prior. Setting ``ilm_weight = 0`` recovers vanilla shallow
    fusion; setting ``fusion_weight = 0`` and ``ilm_weight = 0``
    recovers the AC-only greedy RNN-T decoder.
    """

    fusion_weight: float = 0.2
    ilm_weight: float = 0.0
    # Maximum tokens to emit at a single encoder frame — same as the
    # RNN-T greedy decoder default, prevents infinite emit loops.
    max_emit_per_frame: int = 5
    # Optional per-emission LM temperature. Leaving at 1.0 uses the
    # LM's calibrated distribution; raising it flattens the prior and
    # is a useful knob when the LM over-commits.
    lm_temperature: float = 1.0
    # Same semantic as :class:`RnntModel.greedy_rnnt_decode`'s
    # ``confidence_threshold`` — gates non-blank emissions whose
    # *acoustic* softmax probability falls below the threshold. Gating
    # is intentionally on the acoustic head (not on the fused logits)
    # so that the LM cannot rescue a low-confidence acoustic emission;
    # this preserves the noise/false-positive suppression that
    # threshold 0.6 buys at decode time. ``0.0`` disables gating.
    confidence_threshold: float = 0.0


@torch.no_grad()
def greedy_rnnt_decode_with_lm(
    rnnt: RnntModel,
    lm: GptLM,
    features: torch.Tensor,
    lengths: torch.Tensor | None = None,
    fusion_cfg: FusionConfig | None = None,
) -> list[list[int]]:
    """Greedy RNN-T decoding with external-LM fusion.

    Per-(t, u) non-blank logits are rescored as

        log P_fused = log P_ac + λ_lm · log P_lm − λ_ilm · log P_ilm

    where ``λ_lm = fusion_cfg.fusion_weight``,
    ``λ_ilm = fusion_cfg.ilm_weight``, and ``P_ilm`` is the joint
    output with a zeroed encoder frame (an estimate of the RNN-T
    internal LM). Blank emissions are scored by the acoustic joint only.

    Setting ``λ_ilm = 0`` recovers vanilla shallow fusion; setting
    ``λ_lm = 0`` and ``λ_ilm = 0`` recovers
    :meth:`RnntModel.greedy_rnnt_decode`.

    Args:
        rnnt: trained :class:`RnntModel`. Caller is responsible for
              placing it in ``eval()`` mode and on the right device.
        lm: trained :class:`GptLM`, same device as ``rnnt``. Must share
            the 46-token vocabulary.
        features: ``[B, T, F]`` float32 input audio features.
        lengths:  ``[B]`` valid frame counts, or ``None``.
        fusion_cfg: fusion weights and emission limits (defaults
                    applied when ``None``).

    Returns:
        ``list[list[int]]`` — one list of non-blank token ids per batch
        item, same format as the vanilla RNN-T greedy decoder.
    """
    cfg = fusion_cfg or FusionConfig()
    rnnt.eval()
    lm.eval()

    # Both weights at 0 ⇒ pure acoustic decode. Short-circuit to the
    # vanilla path so we (a) avoid wasted LM forwards and (b) avoid
    # feeding out-of-vocab tokens to a smaller LM (e.g. a 46-token LM
    # paired with a 49-token RNN-T can otherwise see É/À/' indices in
    # the running hypothesis and crash CUBLAS at the embedding lookup).
    if cfg.fusion_weight == 0.0 and cfg.ilm_weight == 0.0:
        return rnnt.greedy_rnnt_decode(
            features, lengths,
            max_emit_per_frame=cfg.max_emit_per_frame,
            confidence_threshold=cfg.confidence_threshold,
        )

    rnnt_vocab = rnnt.cfg.vocab_size
    lm_vocab = lm.cfg.vocab_size
    if lm_vocab > rnnt_vocab:
        raise ValueError(
            f"vocab-size mismatch: lm={lm_vocab} > rnnt={rnnt_vocab}. "
            "LM must share the RNN-T's leading vocabulary indices."
        )
    # Phase 3.4 added É/À/' at the tail of the 49-token vocab. A 46-token
    # LM (Phase 4.0) is therefore semantically a strict prefix and we
    # pad the missing slots with -inf so fusion treats them as "LM
    # never picks these characters" — correct for English/synthetic
    # corpora that don't carry French accents.
    vocab_pad = rnnt_vocab - lm_vocab

    enc_out, enc_lengths = rnnt.acoustic.encode(features, lengths)
    b, t_max, _ = enc_out.shape
    if enc_lengths is None:
        enc_lengths = torch.full(
            (b,), t_max, device=enc_out.device, dtype=torch.long
        )

    blank = rnnt.cfg.blank_index
    vocab_size = rnnt_vocab
    results: list[list[int]] = []

    # Build a [V]-shaped mask that is 0 on the blank slot, 1 elsewhere.
    # We use it to zero out the LM contribution on the blank emission,
    # so argmax for blank is driven purely by the acoustic joint.
    non_blank_mask = torch.ones(vocab_size, device=enc_out.device)
    non_blank_mask[blank] = 0.0

    # Zero-encoder frame used to estimate the internal LM. Reused across
    # every emission to avoid re-allocation. Only materialised when ILM
    # subtraction is actually enabled.
    use_ilm = cfg.ilm_weight != 0.0
    zero_frame = (
        torch.zeros(
            (1, 1, enc_out.size(-1)),
            device=enc_out.device,
            dtype=enc_out.dtype,
        )
        if use_ilm
        else None
    )

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

                # Acoustic-only confidence gate (mirrors
                # RnntModel.greedy_rnnt_decode_aligned). Drop the
                # emission entirely if the *acoustic* argmax confidence
                # is below the threshold, regardless of what the LM
                # would have done — we don't want LM mass rescuing a
                # low-confidence acoustic prediction on noise. Tested
                # before computing LM logprobs so we skip the LM
                # forward on dropped frames (perf win on noise-heavy
                # audio).
                if cfg.confidence_threshold > 0.0:
                    ac_probs = ac_logprobs.exp()
                    ac_top_tok = int(ac_probs.argmax().item())
                    if (
                        ac_top_tok != blank
                        and float(ac_probs[ac_top_tok].item())
                        < cfg.confidence_threshold
                    ):
                        # Treat as blank, advance frame.
                        break

                # Feed the emitted history (with BOS=EOS=blank sentinel)
                # to the LM and take log-probs at the final position.
                # Context length is bounded by ``hyp`` which can only
                # grow by ``max_emit_per_frame`` tokens per frame.
                lm_input = torch.tensor(
                    [[EOS_INDEX] + hyp], dtype=torch.long,
                    device=enc_out.device,
                )
                lm_logits, _ = lm(lm_input)                  # [1, L, V_lm]
                lm_logprobs = F.log_softmax(
                    lm_logits[0, -1, :].float() / max(cfg.lm_temperature, 1e-6),
                    dim=-1,
                )                                            # [V_lm]
                if vocab_pad > 0:
                    pad = torch.full(
                        (vocab_pad,),
                        float("-inf"),
                        device=lm_logprobs.device,
                        dtype=lm_logprobs.dtype,
                    )
                    lm_logprobs = torch.cat([lm_logprobs, pad], dim=0)

                # Add weighted external LM prior to non-blank slots.
                fused = ac_logprobs + cfg.fusion_weight * (
                    lm_logprobs * non_blank_mask
                )
                if use_ilm:
                    ilm_logits = rnnt.joint(zero_frame, pred_out)  # [1,1,1,V]
                    ilm_logprobs = F.log_softmax(
                        ilm_logits[0, 0, 0].float(), dim=-1
                    )
                    fused = fused - cfg.ilm_weight * (
                        ilm_logprobs * non_blank_mask
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
