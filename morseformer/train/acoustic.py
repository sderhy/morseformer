"""Training loop for the acoustic model (Phase 2.0 — clean audio).

Pipeline, per step:

    1. Pull a batch from the infinite SyntheticCWDataset via DataLoader.
    2. Forward through AcousticModel → per-frame CTC log-probs.
    3. Compute CTC loss against the batch's flat target sequence.
    4. Backward, clip grad-norm, AdamW.step(), scheduler.step(), EMA.update().
    5. Log. Every ``eval_every`` steps: greedy-decode the val set with
       EMA weights swapped in, compute CER, maybe save a checkpoint.

Phase 2.0 defaults reflect the decisions in
``memory/project_phase2_decisions.md``: clean audio, WPM uniform in
[16, 28], 6 s fixed utterance, no operator jitter, no channel noise.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from morseformer.core.tokenizer import BLANK_INDEX, ctc_greedy_decode
from morseformer.data.synthetic import DatasetConfig, SyntheticCWDataset, collate
from morseformer.data.validation import (
    ValidationConfig,
    ValidationSample,
    build_clean_validation,
    build_snr_ladder_validation,
)
from morseformer.models.acoustic import AcousticConfig, AcousticModel
from morseformer.train.ema import ExponentialMovingAverage
from morseformer.train.scheduler import WarmupCosineSchedule


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    """Hyperparameters for the Phase 2.0 training loop.

    Defaults are CPU-friendly; raise ``batch_size`` and ``num_workers``
    when training on GPU / multi-core boxes.
    """

    # --- model / data ---
    model: AcousticConfig = field(default_factory=AcousticConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(n_per_wpm=40)
    )
    # When ``validation_snrs`` is non-empty, the validation set is the
    # (WPM × SNR) ladder produced by ``build_snr_ladder_validation`` —
    # each cell gets ``validation.n_per_wpm`` samples. When empty, the
    # validation set is the clean set produced by
    # ``build_clean_validation``.
    validation_snrs: tuple[float, ...] = ()
    validation_rx_filter_bw: float | None = 500.0

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
    batch_size: int = 32
    num_workers: int = 0           # 0 is safer on WSL / small machines
    device: str = "cpu"            # "cuda" when available
    # Master weights always stay in float32; `dtype` controls the AMP
    # autocast precision for the forward pass. CTC loss and log-softmax
    # are always computed in fp32 for numerical stability.
    dtype: str = "float32"         # one of {float32, bfloat16, float16}

    # --- bookkeeping ---
    log_every: int = 50
    eval_every: int = 1_000
    # Periodic cheap snapshot (no validation). ``last.pt`` is always
    # written on every eval, plus every ``save_every`` steps in between
    # so a crash never loses more than this many steps of progress.
    # Set to 0 to disable (saves happen only on eval).
    save_every: int = 500
    checkpoint_dir: Path = Path("checkpoints/phase2_0")
    jsonl_log: Path = Path("checkpoints/phase2_0/train.jsonl")

    # --- EMA ---
    ema_decay: float = 0.9999

    # --- resume ---
    # If set, training picks up from this checkpoint: model + EMA +
    # optimizer + scheduler + step counter + best_cer are restored.
    # The dataset seed is bumped by the resumed step count so we do not
    # replay the first N samples — new data from the resumption point
    # onward, same distribution.
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
    """Return (autocast_dtype_or_None, needs_grad_scaler)."""
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
    """Concatenate only the valid (non-padded) target tokens, as CTC
    expects when called with separate ``target_lengths``."""
    return torch.cat([tokens[i, : int(n_tokens[i].item())] for i in range(tokens.size(0))])


def _reference_text_from_tokens(tokens: torch.Tensor, n: int) -> str:
    from morseformer.core.tokenizer import decode
    return decode(tokens[:n].tolist())


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #


def _val_batches(
    samples: list[ValidationSample], batch_size: int
) -> list[dict]:
    """Collate the in-memory validation set into fixed-size batches."""
    batches = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        batches.append(collate([s.as_batch_item() for s in chunk]))
    return batches


@torch.no_grad()
def evaluate(
    model: AcousticModel,
    val_samples: list[ValidationSample],
    device: torch.device,
    batch_size: int,
    amp_dtype: torch.dtype | None = None,
) -> dict:
    import math
    from eval.metrics import character_error_rate, word_error_rate

    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    count = 0
    per_wpm_cer: dict[float, list[float]] = {}
    per_snr_cer: dict[float, list[float]] = {}

    for batch in _val_batches(val_samples, batch_size):
        features = batch["features"].to(device)
        lengths = batch["n_frames"].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=amp_dtype is not None,
        ):
            log_probs, lengths_out = model(features, lengths=lengths)
        argmax = log_probs.argmax(dim=-1).cpu()  # [B, T']
        assert lengths_out is not None
        lengths_out_list = lengths_out.cpu().tolist()
        # Recover per-item references from the stored samples.
        # Same order as the batch because collate preserves order.
        for i in range(features.size(0)):
            ref_sample = val_samples[count]
            hyp = ctc_greedy_decode(argmax[i, : lengths_out_list[i]].tolist())
            ref = ref_sample.text
            cer = character_error_rate(ref, hyp)
            wer = word_error_rate(ref, hyp)
            total_cer += cer
            total_wer += wer
            per_wpm_cer.setdefault(ref_sample.wpm, []).append(cer)
            per_snr_cer.setdefault(ref_sample.snr_db, []).append(cer)
            count += 1

    # JSON doesn't support +inf; encode as the string "inf" for the log.
    def _snr_key(s: float) -> float | str:
        return "inf" if math.isinf(s) else s

    return {
        "cer": total_cer / count,
        "wer": total_wer / count,
        "per_wpm_cer": {
            wpm: sum(cers) / len(cers) for wpm, cers in sorted(per_wpm_cer.items())
        },
        "per_snr_cer": {
            _snr_key(snr): sum(cers) / len(cers)
            for snr, cers in sorted(
                per_snr_cer.items(),
                key=lambda kv: (math.isinf(kv[0]), kv[0]),
            )
        },
        "n_samples": count,
    }


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


def train(cfg: TrainConfig) -> dict:
    device = torch.device(cfg.device)
    amp_dtype, needs_scaler = _resolve_amp(cfg.dtype)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.jsonl_log.parent.mkdir(parents=True, exist_ok=True)

    # --- resume from checkpoint if requested ----------------------- #
    # Loaded up front so we can bump the dataset seed before building
    # the DataLoader — otherwise workers spawn on the old seed and
    # replay the first `resumed_step` samples we already trained on.
    resume_ckpt: dict | None = None
    resumed_step = 0
    if cfg.resume_from is not None:
        resume_ckpt = torch.load(
            str(cfg.resume_from), map_location="cpu", weights_only=False
        )
        resumed_step = int(resume_ckpt["step"])

    # --- data ---
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

    # --- model + optim ---
    # Master weights always in float32; AMP handles the forward-pass cast.
    model = AcousticModel(cfg.model).to(device=device, dtype=torch.float32)
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

    ctc_loss = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    best_cer = float("inf")
    start_time = time.time()

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        best_cer = float(resume_ckpt.get("best_cer", best_cer))

    jsonl_file = cfg.jsonl_log.open("a")

    def log(event: dict) -> None:
        event["wall_s"] = round(time.time() - start_time, 2)
        jsonl_file.write(json.dumps(event) + "\n")
        jsonl_file.flush()

    if resume_ckpt is None:
        log({"event": "start", "config": _config_to_jsonable(cfg),
             "model_params": model.num_parameters()})
    else:
        log({"event": "resume", "step": resumed_step,
             "from": str(cfg.resume_from),
             "best_cer_so_far": best_cer})

    model.train()
    step = resumed_step
    running_loss = 0.0
    running_n = 0

    loader_iter = iter(loader)
    try:
        while step < cfg.total_steps:
            batch = next(loader_iter)
            features = batch["features"].to(device=device)  # stays float32
            lengths = batch["n_frames"].to(device)
            tokens = batch["tokens"]
            n_tokens = batch["n_tokens"]

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_dtype is not None,
            ):
                log_probs, lengths_out = model(features, lengths=lengths)
            assert lengths_out is not None
            # log_softmax is on the fp32 autocast allow-list, so log_probs
            # is already float32 regardless of autocast mode. CTC is
            # computed outside autocast to preserve numerical stability.
            flat_targets = flatten_targets(tokens, n_tokens).to(device)
            loss = ctc_loss(
                log_probs.float().transpose(0, 1),
                flat_targets,
                lengths_out,
                n_tokens.to(device),
            )

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
            running_n += 1
            step += 1

            if step % cfg.log_every == 0:
                log({
                    "event": "step",
                    "step": step,
                    "loss": running_loss / running_n,
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                })
                running_loss = 0.0
                running_n = 0

            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                with ema.applied_to(model):
                    val_metrics = evaluate(
                        model, val_samples, device, cfg.batch_size,
                        amp_dtype=amp_dtype,
                    )
                model.train()
                log({"event": "eval", "step": step, **val_metrics})

                if val_metrics["cer"] < best_cer:
                    best_cer = val_metrics["cer"]
                    _save_checkpoint(
                        cfg.checkpoint_dir / "best_cer.pt",
                        model, ema, optimizer, scheduler, step, cfg,
                        val_metrics, best_cer,
                    )
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    val_metrics, best_cer,
                )
            elif cfg.save_every > 0 and step % cfg.save_every == 0:
                # Cheap progressive snapshot: write last.pt without
                # running an eval. Crashes never lose more than
                # `save_every` steps of progress.
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    metrics=None, best_cer=best_cer,
                )

    finally:
        jsonl_file.close()

    return {"steps": step, "best_cer": best_cer}


def _save_checkpoint(
    path: Path,
    model: AcousticModel,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    cfg: TrainConfig,
    metrics: dict | None,
    best_cer: float,
) -> None:
    torch.save(
        {
            "best_cer": best_cer,
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


def _config_to_jsonable(cfg: TrainConfig) -> dict:
    """Best-effort JSON-serialisable dump of the TrainConfig for logging."""
    out = asdict(cfg)
    # Path → str for JSON.
    out["checkpoint_dir"] = str(cfg.checkpoint_dir)
    out["jsonl_log"] = str(cfg.jsonl_log)
    if cfg.resume_from is not None:
        out["resume_from"] = str(cfg.resume_from)
    return out
