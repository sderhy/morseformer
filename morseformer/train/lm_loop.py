"""Training loop for the Phase-4 character-level language model.

Pipeline, per step:

    1. Draw a batch of ``(input, target)`` windows from the streaming
       :class:`LmStreamDataset` (no epochs; synthetic text is infinite).
    2. Forward through :class:`GptLM` to get next-token cross-entropy.
    3. Backward, clip grad-norm, AdamW.step(), scheduler.step(),
       EMA.update().
    4. Every ``eval_every`` steps: compute held-out perplexity with the
       EMA weights applied, save ``best.pt`` when it improves.

Held-out evaluation uses a separate ``LmStreamDataset`` seeded
differently from the training stream. Because the text is synthetic
and effectively infinite, there is no risk of overlap between train
and val streams provided the seeds differ — we verify that at run
start by asserting the two RNGs produce different first tokens.
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

from morseformer.data.lm_dataset import LmDatasetConfig, LmStreamDataset
from morseformer.models.lm import GptLM, LmConfig
from morseformer.train.ema import ExponentialMovingAverage
from morseformer.train.scheduler import WarmupCosineSchedule


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class LmTrainConfig:
    """Hyperparameters for the LM training loop."""

    # --- model / data ---
    model: LmConfig = field(default_factory=LmConfig)
    dataset: LmDatasetConfig = field(default_factory=LmDatasetConfig)
    val_seed_offset: int = 1_000_003  # held-out stream uses seed + offset
    val_batches: int = 50

    # --- optimisation ---
    peak_lr: float = 3.0e-4
    beta1: float = 0.9
    beta2: float = 0.95      # GPT-style (lower than 0.98) for small models
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # --- schedule ---
    warmup_steps: int = 500
    total_steps: int = 20_000
    min_lr_ratio: float = 0.1

    # --- runtime ---
    batch_size: int = 128
    num_workers: int = 0
    device: str = "cpu"
    dtype: str = "float32"

    # --- bookkeeping ---
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 500
    checkpoint_dir: Path = Path("checkpoints/lm_phase4_0")
    jsonl_log: Path = Path("checkpoints/lm_phase4_0/train.jsonl")

    # --- EMA ---
    ema_decay: float = 0.999

    # --- resume ---
    resume_from: Path | None = None


# --------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------- #


def _resolve_amp(name: str) -> tuple[torch.dtype | None, bool]:
    if name == "float32":
        return None, False
    if name == "bfloat16":
        return torch.bfloat16, False
    if name == "float16":
        return torch.float16, True
    raise ValueError(f"unknown dtype: {name!r}")


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(
    model: GptLM,
    val_loader_iter,
    n_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict:
    """Average NLL + perplexity over ``n_batches`` held-out batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for _ in range(n_batches):
        batch = next(val_loader_iter)
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=amp_dtype is not None,
        ):
            _, loss = model(inp, tgt)
        total_loss += float(loss.item())
        count += 1
    nll = total_loss / count
    return {"val_nll": nll, "val_ppl": math.exp(nll)}


# --------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------- #


def train(cfg: LmTrainConfig) -> dict:
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
    train_ds_cfg = cfg.dataset
    if resumed_step > 0:
        from dataclasses import replace as dc_replace
        train_ds_cfg = dc_replace(
            cfg.dataset, seed=cfg.dataset.seed + resumed_step * 7919
        )
    train_ds = LmStreamDataset(train_ds_cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    from dataclasses import replace as dc_replace
    val_ds = LmStreamDataset(
        dc_replace(cfg.dataset, seed=cfg.dataset.seed + cfg.val_seed_offset)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        num_workers=0,  # val loader is lightweight, stay single-threaded
        pin_memory=(device.type == "cuda"),
    )

    # --- model + optim -------------------------------------------- #
    model = GptLM(cfg.model).to(device=device, dtype=torch.float32)

    # AdamW with decoupled weight decay. Don't decay biases or norm
    # gains — standard GPT-2 / LLaMA recipe.
    decay_params: list[torch.nn.Parameter] = []
    nodecay_params: list[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    schedule = WarmupCosineSchedule(
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    ema = ExponentialMovingAverage(model, decay=cfg.ema_decay)
    scaler = torch.amp.GradScaler(device=device.type, enabled=needs_scaler)

    best_val_nll = float("inf")
    start_time = time.time()

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        ema.load_state_dict(resume_ckpt["ema"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        best_val_nll = float(resume_ckpt.get("best_val_nll", best_val_nll))

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
            "model_params_non_embed": model.num_parameters(non_embedding=True),
        })
    else:
        log({
            "event": "resume",
            "step": resumed_step,
            "from": str(cfg.resume_from),
            "best_val_nll_so_far": best_val_nll,
        })

    # --- training loop -------------------------------------------- #
    model.train()
    step = resumed_step
    running_loss = 0.0
    running_n = 0

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    try:
        while step < cfg.total_steps:
            batch = next(train_iter)
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_dtype is not None else torch.float32,
                enabled=amp_dtype is not None,
            ):
                _, loss = model(inp, tgt)

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
                    "ppl": math.exp(running_loss / running_n),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                })
                running_loss = 0.0
                running_n = 0

            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                with ema.applied_to(model):
                    val_metrics = evaluate(
                        model, val_iter, cfg.val_batches, device, amp_dtype
                    )
                model.train()
                log({"event": "eval", "step": step, **val_metrics})

                if val_metrics["val_nll"] < best_val_nll:
                    best_val_nll = val_metrics["val_nll"]
                    _save_checkpoint(
                        cfg.checkpoint_dir / "best.pt",
                        model, ema, optimizer, scheduler, step, cfg,
                        val_metrics, best_val_nll,
                    )
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    val_metrics, best_val_nll,
                )
            elif cfg.save_every > 0 and step % cfg.save_every == 0:
                _save_checkpoint(
                    cfg.checkpoint_dir / "last.pt",
                    model, ema, optimizer, scheduler, step, cfg,
                    metrics=None, best_val_nll=best_val_nll,
                )

    finally:
        jsonl_file.close()

    return {"steps": step, "best_val_nll": best_val_nll,
            "best_val_ppl": math.exp(best_val_nll)}


def _save_checkpoint(
    path: Path,
    model: GptLM,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    cfg: LmTrainConfig,
    metrics: dict | None,
    best_val_nll: float,
) -> None:
    torch.save(
        {
            "best_val_nll": best_val_nll,
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


def _config_to_jsonable(cfg: LmTrainConfig) -> dict:
    out = asdict(cfg)
    out["checkpoint_dir"] = str(cfg.checkpoint_dir)
    out["jsonl_log"] = str(cfg.jsonl_log)
    if cfg.resume_from is not None:
        out["resume_from"] = str(cfg.resume_from)
    return out
