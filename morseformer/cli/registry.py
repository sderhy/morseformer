"""Model registry + local resolution + auto-download from HuggingFace Hub.

The CLI knows a handful of named checkpoints. Each is a file in the HF
repo ``sderhy/morseformer``. ``resolve_model(name)`` returns a local
``Path`` to the checkpoint, downloading it on demand if necessary.

Resolution order:

1. ``release/<name>.pt`` (release tree from a git checkout)
2. ``checkpoints/<phase>/last.pt`` (dev checkpoint matching the name)
3. HF cache via ``huggingface_hub.hf_hub_download`` (lazy import)

The two repo-relative fallbacks let developers run the CLI without
having pushed to HF, while end-users (post-pip-install) hit the HF
cache directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


HF_REPO = "sderhy/morseformer"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    filename: str       # exact filename on HF
    kind: str           # "rnnt" | "lm"
    vocab: int
    description: str
    recommended: bool   # shown without --advanced
    legacy: bool = False


REGISTRY: dict[str, ModelInfo] = {
    "rnnt_phase5_8": ModelInfo(
        name="rnnt_phase5_8",
        filename="rnnt_phase5_8.pt",
        kind="rnnt", vocab=49,
        description="v0.6.0 acoustic — Phase 5.8 English-literary curriculum "
                    "(Moby Dick + Pride & Prejudice + Sherlock Holmes + "
                    "Frankenstein) on top of Phase 5.7 amateur-idiom. "
                    "Halved run-on prosign probabilities to reduce phantom "
                    "BK/+/K on continuous prose.",
        recommended=True,
    ),
    "lm_phase5_2": ModelInfo(
        name="lm_phase5_2",
        filename="lm_phase5_2.pt",
        kind="lm", vocab=49,
        description="v0.4.1 LM — matched to PHASE_3_4_MIX, val_ppl 5.626. "
                    "Use at λ=0.7 for shallow fusion on prose audio.",
        recommended=True,
    ),
    "rnnt_phase5_7": ModelInfo(
        name="rnnt_phase5_7",
        filename="rnnt_phase5_7.pt",
        kind="rnnt", vocab=49,
        description="v0.5.3 acoustic — Phase 5.7 amateur-idiom curriculum "
                    "(5NN cut-numbers + run-on UR/SK/KN/BK). Kept for diff.",
        recommended=False,
    ),
    "rnnt_phase5_5": ModelInfo(
        name="rnnt_phase5_5",
        filename="rnnt_phase5_5.pt",
        kind="rnnt", vocab=49,
        description="v0.5.1 / v0.5.2 acoustic — Phase 5.5 long inter-word "
                    "silence curriculum.",
        recommended=False,
    ),
    "rnnt_phase5_4": ModelInfo(
        name="rnnt_phase5_4",
        filename="rnnt_phase5_4.pt",
        kind="rnnt", vocab=49,
        description="v0.5.0 acoustic — Phase 5.3 wider jitter + Phase 5.4 "
                    "30 % real-audio mix.",
        recommended=False,
    ),
    "rnnt_phase3_5": ModelInfo(
        name="rnnt_phase3_5",
        filename="rnnt_phase3_5.pt",
        kind="rnnt", vocab=49,
        description="v0.4.0 / v0.4.1 acoustic — Phase 3.4 + 3.5, "
                    "synthetic-only.",
        recommended=False,
    ),
    "rnnt_phase3_3": ModelInfo(
        name="rnnt_phase3_3",
        filename="rnnt_phase3_3.pt",
        kind="rnnt", vocab=46,
        description="v0.3 acoustic — multilingual ASCII-normalised prose, "
                    "no accent tokens.",
        recommended=False, legacy=True,
    ),
    "rnnt_phase3_2": ModelInfo(
        name="rnnt_phase3_2",
        filename="rnnt_phase3_2.pt",
        kind="rnnt", vocab=46,
        description="v0.2 acoustic — anti-hallucination curriculum.",
        recommended=False, legacy=True,
    ),
    "rnnt_phase3_0": ModelInfo(
        name="rnnt_phase3_0",
        filename="rnnt_phase3_0.pt",
        kind="rnnt", vocab=46,
        description="v0.1 acoustic — AWGN-only baseline.",
        recommended=False, legacy=True,
    ),
    "lm_phase4_0": ModelInfo(
        name="lm_phase4_0",
        filename="lm_phase4_0.pt",
        kind="lm", vocab=46,
        description="Legacy v0.1-era LM, 100 % ham-radio mix. Not "
                    "recommended for fusion (research only).",
        recommended=False, legacy=True,
    ),
}

RECOMMENDED_ACOUSTIC = "rnnt_phase5_8"
RECOMMENDED_LM = "lm_phase5_2"


def known_names(*, advanced: bool = False) -> list[str]:
    """Return the set of model names visible to ``morseformer models``.

    Default (``advanced=False``) returns only the recommended set —
    matches the user-chosen "Recommended only + advanced toggle" UX.
    """
    if advanced:
        return list(REGISTRY)
    return [n for n, info in REGISTRY.items() if info.recommended]


def get_info(name: str) -> ModelInfo:
    if name not in REGISTRY:
        known = ", ".join(REGISTRY)
        raise SystemExit(
            f"[morseformer] unknown model '{name}'. Known: {known}."
        )
    return REGISTRY[name]


def _phase_dir_for(name: str) -> str:
    """Map a registry name to its checkpoints/<phase>/ directory."""
    # rnnt_phase5_7 -> phase5_7, lm_phase5_2 -> lm_phase5_2 (dev keeps
    # the LM one phase-coupled).
    if name.startswith("rnnt_"):
        return name[len("rnnt_"):]
    return name


def resolve_model(name: str, *, repo_root: Path | None = None) -> Path:
    """Return a local Path to the checkpoint, downloading it if needed.

    The repo-root resolution lets ``pytest`` and dev sessions hit local
    files; pip-installed users always go through the HF cache path.
    """
    info = get_info(name)
    root = repo_root or Path.cwd()

    release = root / "release" / info.filename
    if release.exists():
        return release

    dev = root / "checkpoints" / _phase_dir_for(name) / "last.pt"
    if dev.exists():
        return dev

    # Lazy import — the CLI is usable without huggingface_hub for local
    # files, and we don't want to force the dep on every import.
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - error path
        raise SystemExit(
            f"[morseformer] '{name}' is not in release/ or checkpoints/, "
            f"and huggingface_hub is not installed.\n"
            f"  pip install 'morseformer[hub]'\n"
            f"or download the file manually from "
            f"https://huggingface.co/{HF_REPO} and place it at "
            f"{release}."
        ) from exc

    print(f"[morseformer] downloading {info.filename} from "
          f"https://huggingface.co/{HF_REPO} (one-time)...")
    cached = hf_hub_download(repo_id=HF_REPO, filename=info.filename)
    return Path(cached)
