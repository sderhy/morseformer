"""Neural-network decoder: load an AcousticModel checkpoint, run greedy CTC.

Implements the same ``(audio, sample_rate) -> str`` signature as the
rule-based baseline so that ``eval/cli.py`` and ``eval.snr_ladder`` can
drop it in unchanged. Built around a callable class so a single
checkpoint load is amortised across a whole dataset run.

Checkpoint format matches what ``morseformer.train.acoustic`` writes:

    {
        "step":      int,
        "model":     state_dict,
        "ema":       state_dict (parameter names only),
        "config":    JSON-friendly TrainConfig dump,
        "metrics":   dict,
        ...
    }

``from_checkpoint`` reconstructs the model from ``config["model"]`` and
the front-end config from ``config["dataset"]["frontend"]`` so that
inference is bit-for-bit aligned with the preprocessing the model was
trained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from morseformer.core.tokenizer import ctc_greedy_decode
from morseformer.features import FrontendConfig, extract_features
from morseformer.models.acoustic import AcousticConfig, AcousticModel


@dataclass
class NeuralDecoderConfig:
    """Small inference-time knobs. Audio preprocessing is locked to the
    training front-end config; these are decode-time only."""

    device: str = "cpu"
    use_ema: bool = True


class NeuralDecoder:
    """Callable CW decoder backed by a trained Conformer + CTC head."""

    def __init__(
        self,
        model: AcousticModel,
        frontend_cfg: FrontendConfig,
        train_sample_rate: int,
        device: torch.device,
    ) -> None:
        self.model = model.eval().to(device)
        self.frontend_cfg = frontend_cfg
        self.train_sample_rate = train_sample_rate
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, cfg: NeuralDecoderConfig | None = None
    ) -> "NeuralDecoder":
        cfg = cfg or NeuralDecoderConfig()
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

        train_cfg = ckpt["config"]
        model_cfg = AcousticConfig(**train_cfg["model"])
        frontend_cfg = FrontendConfig(**train_cfg["dataset"]["frontend"])
        train_sample_rate = int(train_cfg["dataset"]["sample_rate"])

        model = AcousticModel(model_cfg)
        model.load_state_dict(ckpt["model"])
        if cfg.use_ema and "ema" in ckpt and ckpt["ema"]:
            # EMA keys are a subset of model.state_dict() (parameters only);
            # strict=False tolerates the missing buffers.
            model.load_state_dict(ckpt["ema"], strict=False)

        return cls(
            model=model,
            frontend_cfg=frontend_cfg,
            train_sample_rate=train_sample_rate,
            device=torch.device(cfg.device),
        )

    @torch.no_grad()
    def __call__(self, audio: np.ndarray, sample_rate: int) -> str:
        """Decode one audio clip to a text transcript."""
        if sample_rate != self.train_sample_rate:
            raise ValueError(
                f"checkpoint was trained at sample_rate={self.train_sample_rate}, "
                f"got {sample_rate}. Resample the input before decoding."
            )
        if audio.size == 0:
            return ""

        features = extract_features(audio, sample_rate, self.frontend_cfg)  # [T, 1]
        if features.shape[0] == 0:
            return ""

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)  # [1, T, 1]
        log_probs, _ = self.model(x)  # [1, T', V]
        argmax = log_probs.argmax(dim=-1).squeeze(0).cpu().tolist()
        return ctc_greedy_decode(argmax)

    def decode_batch(
        self, audios: list[np.ndarray], sample_rate: int
    ) -> list[str]:
        """Batched variant — useful for SNR-ladder style evaluation to
        amortise GPU/CPU dispatch. Pads features to the max length in
        the batch, masks padding via the model's ``lengths`` argument,
        and trims each hypothesis back to its valid frame count before
        the greedy CTC collapse.
        """
        if sample_rate != self.train_sample_rate:
            raise ValueError(
                f"checkpoint was trained at sample_rate={self.train_sample_rate}, "
                f"got {sample_rate}."
            )
        if not audios:
            return []

        feats = [extract_features(a, sample_rate, self.frontend_cfg) for a in audios]
        t_max = max(f.shape[0] for f in feats)
        if t_max == 0:
            return ["" for _ in audios]

        b = len(feats)
        batch = np.zeros((b, t_max, 1), dtype=np.float32)
        lengths = np.zeros(b, dtype=np.int64)
        for i, f in enumerate(feats):
            batch[i, : f.shape[0]] = f
            lengths[i] = f.shape[0]

        x = torch.from_numpy(batch).to(self.device)
        lens = torch.from_numpy(lengths).to(self.device)
        with torch.no_grad():
            log_probs, lens_out = self.model(x, lengths=lens)
        assert lens_out is not None
        argmax = log_probs.argmax(dim=-1).cpu()  # [B, T']
        lens_out_list = lens_out.cpu().tolist()

        out: list[str] = []
        for i in range(b):
            out.append(
                ctc_greedy_decode(argmax[i, : lens_out_list[i]].tolist())
                if lens_out_list[i] > 0 else ""
            )
        return out
