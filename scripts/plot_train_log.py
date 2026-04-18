"""Plot loss / CER / LR curves from a training JSONL log.

Usage::

    python -m scripts.plot_train_log --log checkpoints/phase2_0/train.jsonl
    python -m scripts.plot_train_log --log ... --output curves.png

Output is either shown interactively or written to the given path. No
dependency on matplotlib in the main package — the import is local to
this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_events(path: Path) -> dict:
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    grad_norms: list[float] = []
    eval_steps: list[int] = []
    eval_cers: list[float] = []
    eval_wers: list[float] = []
    per_wpm_curves: dict[float, list[tuple[int, float]]] = {}
    start_cfg: dict | None = None

    with path.open() as f:
        for line in f:
            evt = json.loads(line)
            kind = evt.get("event")
            if kind == "step":
                steps.append(evt["step"])
                losses.append(evt["loss"])
                lrs.append(evt["lr"])
                grad_norms.append(evt["grad_norm"])
            elif kind == "eval":
                eval_steps.append(evt["step"])
                eval_cers.append(evt["cer"])
                eval_wers.append(evt["wer"])
                for wpm, cer in evt.get("per_wpm_cer", {}).items():
                    per_wpm_curves.setdefault(float(wpm), []).append(
                        (evt["step"], cer)
                    )
            elif kind == "start":
                start_cfg = evt.get("config")
    return {
        "steps": steps,
        "losses": losses,
        "lrs": lrs,
        "grad_norms": grad_norms,
        "eval_steps": eval_steps,
        "eval_cers": eval_cers,
        "eval_wers": eval_wers,
        "per_wpm": per_wpm_curves,
        "start": start_cfg,
    }


def plot(data: dict, output: Path | None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("morseformer training", fontsize=14)

    ax_loss, ax_lr = axes[0]
    ax_cer, ax_per_wpm = axes[1]

    # Loss
    if data["steps"]:
        ax_loss.plot(data["steps"], data["losses"], linewidth=1)
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("CTC loss")
        ax_loss.set_yscale("log")
        ax_loss.grid(alpha=0.3)
        ax_loss.set_title("CTC loss")

    # LR + grad norm on twin axis
    if data["steps"]:
        ax_lr.plot(data["steps"], data["lrs"], color="tab:orange",
                   label="learning rate")
        ax_lr.set_xlabel("step")
        ax_lr.set_ylabel("lr", color="tab:orange")
        ax_lr.grid(alpha=0.3)
        ax_lr.set_title("LR & grad norm")
        ax_lr2 = ax_lr.twinx()
        ax_lr2.plot(data["steps"], data["grad_norms"], color="tab:green",
                    alpha=0.6, label="grad norm")
        ax_lr2.set_ylabel("grad norm", color="tab:green")

    # CER / WER
    if data["eval_steps"]:
        ax_cer.plot(data["eval_steps"], data["eval_cers"], marker="o",
                    label="CER")
        ax_cer.plot(data["eval_steps"], data["eval_wers"], marker="s",
                    label="WER")
        ax_cer.set_xlabel("step")
        ax_cer.set_ylabel("error rate")
        ax_cer.set_yscale("log")
        ax_cer.set_ylim(bottom=1e-4)
        ax_cer.grid(alpha=0.3)
        ax_cer.legend()
        ax_cer.set_title("Validation CER / WER")

    # Per-WPM CER curves
    if data["per_wpm"]:
        for wpm in sorted(data["per_wpm"]):
            points = data["per_wpm"][wpm]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax_per_wpm.plot(xs, ys, marker=".", label=f"{wpm:.0f} WPM")
        ax_per_wpm.set_xlabel("step")
        ax_per_wpm.set_ylabel("CER")
        ax_per_wpm.set_yscale("log")
        ax_per_wpm.set_ylim(bottom=1e-4)
        ax_per_wpm.grid(alpha=0.3)
        ax_per_wpm.legend(loc="best", fontsize=8)
        ax_per_wpm.set_title("Per-WPM CER")

    plt.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=120, bbox_inches="tight")
        print(f"[plot_train_log] wrote {output}")
    else:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, required=True,
                        help="Path to the training JSONL file.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output image path (default: interactive window).")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print(f"no such file: {args.log}", file=sys.stderr)
        return 2

    data = load_events(args.log)
    if not data["steps"] and not data["eval_steps"]:
        print(f"no step/eval events found in {args.log}", file=sys.stderr)
        return 1

    plot(data, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
