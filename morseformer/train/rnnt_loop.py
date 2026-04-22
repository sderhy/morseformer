"""Training loop for the RNN-T multi-task model (Phase 3).

Pipeline, per step:

    1. Pull a batch from SyntheticCWDataset via DataLoader.
    2. Forward through RnntModel → encoder features, CTC log-probs on the
       encoder head, and the joint-network logits ``[B, T', U + 1, V]``.
    3. Compute CTC loss on the encoder head + RNN-T loss on the joint.
       Total loss = ``ctc_weight * ctc + rnnt_weight * rnnt``.
    4. Backward, clip grad-norm, AdamW.step(), scheduler.step(), EMA.update().
    5. Log. Every ``eval_every`` steps: evaluate BOTH decoders (CTC greedy
       on the encoder head, RNN-T greedy via the joint) with EMA weights,
       report both CER numbers.

The encoder is optionally bootstrapped from a Phase 2 checkpoint. The
RNN-T prediction + joint heads train from random init, but with a
competent encoder under them so the run skips the long "learn acoustic
features from scratch" phase.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from morseformer.core.tokenizer import BLANK_INDEX, ctc_greedy_decode, decode
from morseformer.data.synthetic import DatasetConfig, SyntheticCWDataset, collate
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_clean_validation,
    build_snr_ladder_validation,
)
from morseformer.models.acoustic import AcousticModel
from morseformer.models.rnnt import RnntConfig, RnntModel
from morseformer.train.ema import ExponentialMovingAverage
from morseformer.train.scheduler import WarmupCosineSchedule


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class RnntTrainConfig:
    """Hyperparameters for the Phase 3 multi-task training loop."""

    # --- model / data ---
    model: RnntConfig = field(default_factory=RnntConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(n_per_wpm=40)
    )
    validation_snrs: tuple[float, ...] = ()
    validation_rx_filter_bw: float | None = 500.0

    # --- multi-task weighting ---
    ctc_weight: float = 0.3
    rnnt_weight: float = 0.7

    # --- bootstrap ---
    # Path to a Phase 2 checkpoint (e.g. checkpoints/phase2_1/best_cer.pt)
    # whose encoder + CTC head is loaded into this model's
    # ``acoustic`` submodule before training starts. The RNN-T
    # prediction + joint networks still train from random init.
    pretrained_encoder: Path | None = None
    # Path to a Phase 3 RnntModel checkpoint (e.g.
    # checkpoints/phase3_0/best_rnnt.pt) to warm-start every module.
    # Uses strict=False, so a deeper encoder (n_layers larger than the
    # source) will load the matching first layers and leave the rest
    # at random init — function-preserving scaling in depth.
    pretrained_rnnt: Path | None = None

    # --- optimisation ---
    peak_lr: float = 3.0e-4
    beta1: float = 0.9
    beta2: float = 0.98
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # --- schedule ---
    warmup_steps: int = 2_000
    total_steps: int = 100_000
    min_lr_ratio: float = 0.1

    # --- runtime ---
    batch_size: int = 16
    num_workers: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    # --- bookkeeping ---
    log_every: int = 50
    eval_every: int = 1_000
    save_every: int = 500
    checkpoint_dir: Path = Path("checkpoints/phase3_0")
    jsonl_log: Path = Path("checkpoints/phase3_0/train.jsonl")

    # --- EMA ---
    ema_decay: float = 0.9999

    # --- resume ---
    resume_from: Path | None = None


# --------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------- #


_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _resolve_amp(name: str) -> tuple[torch.dtype | None, bool]:
    if name == "float32":
        return None, False
    if name == "bfloat16":
        return torch.bfloat16, False
    if name == "float16":
        return torch.float16, True
    raise ValueError(f"unknown dtype: {name!r}")


def flatten_targets(
    tokens: torch.Tensor, n_tokens: torch.Tensor
) -> torch.Tensor:
    return torch.cat(
        [tokens[i, : int(n_tokens[i].item())] for i in range(tokens.size(0))]
    )


def load_pretrained_encoder_state(ckpt_path: Path) -> dict:
    """Load a Phase 2 checkpoint and return an AcousticModel-compatible
    state_dict with EMA weights applied if available.

    EMA weights are what produce the best-CER numbers in Phase 2 eval,
    so bootstrapping with them gives the cleanest starting point.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_state: dict[str, torch.Tensor] = ckpt["model"]
    ema_state: dict[str, torch.Tensor] | None = ckpt.get("ema")
    if ema_state:
        # EMA only covers float-params; buffers / non-param tensors come
        # from the live-model state. Overlay EMA on top.
        merged = dict(model_state)
        for k, v in ema_state.items():
            if k in merged:
                merged[k] = v
        return merged
    return model_state


def load_pretrained_rnnt_state(ckpt_path: Path) -> dict:
    """Load a Phase 3 RnntModel checkpoint and return a full-model
    state_dict with EMA weights overlaid where available.

    This is the analogue of :func:`load_pretrained_encoder_state` but
    for a whole RNN-T model (encoder + CTC head + prediction + joint).
    The caller should apply the returned dict with ``strict=False`` so
    that a deeper encoder (more ``blocks.N`` keys than the source
    checkpoint) falls back to random init for the extra layers.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_state: dict[str, torch.Tensor] = ckpt["model"]
    ema_state: dict[str, torch.Tensor] | None = ckpt.get("ema")
    if ema_state:
        merged = dict(model_state)
        for k, v in ema_state.items():
            if k in merged:
                merged[k] = v
        return merged
    return model_state


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


def _val_batches(
    samples: list[ValidationSample], batch_size: int
) -> list[dict]:
    batches = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        batches.append(collate([s.as_batch_item() for s in chunk]))
    return batches


def _snr_key(s: float) -> float | str:
    return "inf" if math.isinf(s) else s


@torch.no_grad()
def evaluate(
    model: RnntModel,
    val_samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    """Run both CTC-greedy and RNN-T-greedy decoding over the val set.

    Returns a dict with overall + per-WPM + per-SNR CER for each head.
    """
    from eval.metrics import character_error_rate, word_error_rate

    model.eval()
    ctc_cer_tot = 0.0
    rnnt_cer_tot = 0.0
    ctc_wer_tot = 0.0
    rnnt_wer_tot = 0.0
    count = 0
    ctc_per_wpm: dict[float, list[float]] = {}
    ctc_per_snr: dict[float, list[float]] = {}
    rnnt_per_wpm: dict[float, list[float]] = {}
    rnnt_per_snr: dict[float, list[float]] = {}

    for batch in _val_batches(val_samples, batch_size):
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=amp_dtype is not None,
        ):
            enc_out, enc_lengths = model.acoustic.encode(features, lengths)
            ctc_logits = model.acoustic.head(enc_out)

        ctc_argmax = ctc_logits.argmax(dim=-1).cpu()
        assert enc_lengths is not None
        enc_lengths_list = enc_lengths.cpu().tolist()

        # RNN-T greedy decode — needs fp32 pass internally.
        rnnt_hyps = model.greedy_rnnt_decode(features, lengths)

        for i in range(features.size(0)):
            ref_sample = val_samples[count]
            ref = ref_sample.text
            ctc_hyp = ctc_greedy_decode(
                ctc_argmax[i, : enc_lengths_list[i]].tolist()
            )
            rnnt_hyp = decode(rnnt_hyps[i])

            c_ctc = character_error_rate(ref, ctc_hyp)
            c_rnnt = character_error_rate(ref, rnnt_hyp)
            w_ctc = word_error_rate(ref, ctc_hyp)
            w_rnnt = word_error_rate(ref, rnnt_hyp)
            ctc_cer_tot += c_ctc
            rnnt_cer_tot += c_rnnt
            ctc_wer_tot += w_ctc
            rnnt_wer_tot += w_rnnt

            ctc_per_wpm.setdefault(ref_sample.wpm, []).append(c_ctc)
            ctc_per_snr.setdefault(ref_sample.snr_db, []).append(c_ctc)
            rnnt_per_wpm.setdefault(ref_sample.wpm, []).append(c_rnnt)
            rnnt_per_snr.setdefault(ref_sample.snr_db, []).append(c_rnnt)
            count += 1

    def _avg_dict(d: dict[float, list[float]]) -> dict:
        return {k: sum(v) / len(v) for k, v in sorted(d.items())}

    def _avg_snr_dict(d: dict[float, list[float]]) -> dict:
        return {
            _snr_key(k): sum(v) / len(v)
            for k, v in sorted(d.items(), key=lambda kv: (math.isinf(kv[0]), kv[0]))
        }

    return {
        "ctc_cer": ctc_cer_tot / count,
        "ctc_wer": ctc_wer_tot / count,
        "rnnt_cer": rnnt_cer_tot / count,
        "rnnt_wer": rnnt_wer_tot / count,
        "ctc_per_wpm_cer": _avg_dict(ctc_per_wpm),
        "ctc_per_snr_cer": _avg_snr_dict(ctc_per_snr),
        "rnnt_per_wpm_cer": _avg_dict(rnnt_per_wpm),
        "rnnt_per_snr_cer": _avg_snr_dict(rnnt_per_snr),
        "n_samples": count,
    }


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


def train(cfg: RnntTrainConfig) -> dict:
    import torchaudio.functional as AF

    device = torch.device(cfg.device)
    amp_dtype, needs_scaler = _resolve_amp(cfg.dtype)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    # --- resume ---------------------------------------------------- #
    resume_ckpt: dict | None = None
    resumed_step = 0
    if cfg.resume_from is not None:
        resume_ckpt = torch.load(
            str(cfg.resume_from), map_location="cpu", weights_only=False
        )
        resumed_step = int(resume_ckpt["step"])

    # --- data ------------------------------------------------------ #
    dataset_cfg = cfg.dataset
    if resumed_step > 0:
        from dataclasses import replace as dc_replace
        dataset_cfg = dc_replace(
            cfg.dataset, seed=cfg.dataset.seed + resumed_step * 7919
        )
    dataset = SyntheticCWDataset(dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    if cfg.validation_snrs:
        val_samples = build_snr_ladder_validation(
            tuple(cfg.validation_snrs),
            cfg=cfg.validation,
            rx_filter_bw=cfg.validation_rx_filter_bw,
        )
    else:
        val_samples = build_clean_validation(cfg.validation)

    # --- model + optim -------------------------------------------- #
    model = RnntModel(cfg.model).to(device=device, dtype=torch.float32)

    # Bootstrap the encoder before optimizer / EMA are built so the
    # initial EMA shadow matches the bootstrapped weights.
    bootstrap_info: dict | None = None
    if cfg.pretrained_encoder is not None and cfg.pretrained_rnnt is not None:
        raise ValueError(
            "Set only one of --pretrained-encoder / --pretrained-rnnt"
        )
    if cfg.pretrained_encoder is not None and resume_ckpt is None:
        enc_state = load_pretrained_encoder_state(cfg.pretrained_encoder)
        model.load_encoder_state_dict(enc_state)
    elif cfg.pretrained_rnnt is not None and resume_ckpt is None:
        full_state = load_pretrained_rnnt_state(cfg.pretrained_rnnt)
        incompat = model.load_state_dict(full_state, strict=False)
        bootstrap_info = {
            "source": str(cfg.pretrained_rnnt),
            "missing_keys": list(incompat.missing_keys),
            "unexpected_keys": list(incompat.unexpected_keys),
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    schedule = WarmupCosineSchedule(
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = ExponentialMovingAverage(model, decay=cfg.ema_decay)

    scaler = torch.amp.GradScaler(device=device.type, enabled=needs_scaler)

    ctc_loss_fn = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    best_ctc_cer = float("inf")
    best_rnnt_cer = float("inf")
    start_time = time.time()

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        best_ctc_cer = float(resume_ckpt.get("best_ctc_cer", best_ctc_cer))
        best_rnnt_cer = float(resume_ckpt.get("best_rnnt_cer", best_rnnt_cer))

    jsonl_file = cfg.jsonl_log.open("a")

    def log(event: dict) -> None:
        event["wall_s"] = round(time.time() - start_time, 2)
        jsonl_file.write(json.dumps(event) + "\n")
        jsonl_file.flush()

    if resume_ckpt is None:
        log({
            "event": "start",
            "config": _config_to_jsonable(cfg),
            "model_params": model.num_parameters(),
            "param_breakdown": model.num_parameters_by_module(),
            "pretrained_encoder": (
                str(cfg.pretrained_encoder)
                if cfg.pretrained_encoder is not None
                else None
            ),
            "pretrained_rnnt": bootstrap_info,
        })
    else:
        log({
            "event": "resume", "step": resumed_step,
            "from": str(cfg.resume_from),
            "best_ctc_cer_so_far": best_ctc_cer,
            "best_rnnt_cer_so_far": best_rnnt_cer,
        })

    model.train()
    step = resumed_step
    running_loss = 0.0
    running_ctc = 0.0
    running_rnnt = 0.0
    running_n = 0

    loader_iter = iter(loader)
    try:
        while step < cfg.total_steps:
            batch = next(loader_iter)
            features = batch["features"].to(device=device)
            lengths = batch["n_frames"].to(device)
            tokens = batch["tokens"].to(device)
            n_tokens = batch["n_tokens"].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_dtype is not None,
            ):
                out = model(features, tokens, lengths=lengths)

            enc_lengths = out["enc_lengths"]
            assert enc_lengths is not None

            # CTC loss — computed in fp32 outside autocast for stability.
            flat_targets = flatten_targets(tokens, n_tokens)
            ctc = ctc_loss_fn(
                out["ctc_log_probs"].float().transpose(0, 1),
                flat_targets,
                enc_lengths,
                n_tokens,
            )

            # RNN-T loss — torchaudio requires fp32 logits and int32
            # length / target tensors.
            rnnt = AF.rnnt_loss(
                out["joint_logits"].float(),
                tokens.int(),
                enc_lengths.int(),
                n_tokens.int(),
                blank=BLANK_INDEX,
                reduction="mean",
            )

            loss = cfg.ctc_weight * ctc + cfg.rnnt_weight * rnnt

            optimizer.zero_grad(set_to_none=True)
            if needs_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_norm
                )
                optimizer.step()
            scheduler.step()
            ema.update(model)

            running_loss += float(loss.item())
            running_ctc += float(ctc.item())
            running_rnnt += float(rnnt.item())
            running_n += 1
            step += 1

            if step % cfg.log_every == 0:
                log({
                    "event": "step",
                    "step": step,
                    "loss": running_loss / running_n,
                    "ctc": running_ctc / running_n,
                    "rnnt": running_rnnt / running_n,
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                })
                running_loss = 0.0
                running_ctc = 0.0
                running_rnnt = 0.0
                running_n = 0

            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                with ema.applied_to(model):
                    val_metrics = evaluate(
                        model, val_samples, device, cfg.batch_size,
                        amp_dtype=amp_dtype,
                    )
                model.train()
                log({"event": "eval", "step": step, **val_metrics})

                if val_metrics["ctc_cer"] < best_ctc_cer:
                    best_ctc_cer = val_metrics["ctc_cer"]
                    _save_checkpoint(
                        cfg.checkpoint_dir / "best_ctc.pt",
                        model, ema, optimizer, scheduler, step, cfg,
                        val_metrics, best_ctc_cer, best_rnnt_cer,
                    )
                if val_metrics["rnnt_cer"] < best_rnnt_cer:
                    best_rnnt_cer = val_metrics["rnnt_cer"]
                    _save_checkpoint(
                        cfg.checkpoint_dir / "best_rnnt.pt",
                        model, ema, optimizer, scheduler, step, cfg,
                        val_metrics, best_ctc_cer, best_rnnt_cer,
                    )
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    val_metrics, best_ctc_cer, best_rnnt_cer,
                )
            elif cfg.save_every > 0 and step % cfg.save_every == 0:
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    metrics=None,
                    best_ctc_cer=best_ctc_cer,
                    best_rnnt_cer=best_rnnt_cer,
                )

    finally:
        jsonl_file.close()

    return {
        "steps": step,
        "best_ctc_cer": best_ctc_cer,
        "best_rnnt_cer": best_rnnt_cer,
    }


def _save_checkpoint(
    path: Path,
    model: RnntModel,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    cfg: RnntTrainConfig,
    metrics: dict | None,
    best_ctc_cer: float,
    best_rnnt_cer: float,
) -> None:
    torch.save(
        {
            "best_ctc_cer": best_ctc_cer,
            "best_rnnt_cer": best_rnnt_cer,
            "step": step,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": _config_to_jsonable(cfg),
            "metrics": metrics,
        },
        path,
    )


def _config_to_jsonable(cfg: RnntTrainConfig) -> dict:
    out = asdict(cfg)
    out["checkpoint_dir"] = str(cfg.checkpoint_dir)
    out["jsonl_log"] = str(cfg.jsonl_log)
    if cfg.resume_from is not None:
        out["resume_from"] = str(cfg.resume_from)
    if cfg.pretrained_encoder is not None:
        out["pretrained_encoder"] = str(cfg.pretrained_encoder)
    if cfg.pretrained_rnnt is not None:
        out["pretrained_rnnt"] = str(cfg.pretrained_rnnt)
    return out
