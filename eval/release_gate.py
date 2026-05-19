"""Release gate — single command that decides ship / no-ship for a checkpoint.

Reads ``eval/release_gate_v1.json`` (a versioned manifest of categories
+ thresholds calibrated against the baseline acoustic), runs every
category against the candidate acoustic (and optional LM), compares to
the thresholds, writes a JSON report under ``reports/``, prints a
coloured pass/fail table, and exits 0 if every category passes, 1
otherwise.

Run::

    python -m eval.release_gate
    python -m eval.release_gate --acoustic rnnt_phase5_5
    python -m eval.release_gate --acoustic rnnt_phase5_7 --lm lm_phase5_2
    python -m eval.release_gate --manifest eval/release_gate_v1.json \\
                                --bench-manifest data/bench/manifest.jsonl

Designed to be the single source of truth that NEXT.md's P0-B asks for:
no more ship-then-revert based on live impressions (cf the phase5_8 →
phase5_5 revert in ``project_phase5_9_failure``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eval.bench_lcwo import (
    _load_audio,
    _load_lm,
    _load_rnnt,
    load_manifest,
    make_morseformer_decoder,
)
from eval.benchmark import run as run_benchmark
from eval.datasets import Sample
from eval.metrics import character_error_rate
from morse_synth.core import render
from morse_synth.operator import OperatorConfig
from morseformer.cli.presets import get_preset
from morseformer.cli.registry import resolve_model
from morseformer.data.text import _normalize_prose
from morseformer.decoding.postprocess import format_output
from morseformer.features.frontend import FrontendConfig, extract_features

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD = "\033[1m"

_SAMPLE_RATE = 8000
_CARRIER_HZ = 600.0


@dataclass
class CategoryResult:
    """One row of the release gate table."""
    id: str
    kind: str
    metric: str
    measured: float
    threshold: float
    baseline: float | None
    unit: str
    passed: bool
    extra: dict[str, Any]


# --------------------------------------------------------------------------- #
# Category runners
# --------------------------------------------------------------------------- #


def _decoder_for_preset(
    preset_name: str,
    cache: dict[str, Any],
    *,
    acoustic_name: str,
    lm_name: str | None,
    device: torch.device,
):
    preset = get_preset(preset_name)
    if preset.acoustic in cache:
        acoustic = cache[preset.acoustic]
    elif acoustic_name == preset.acoustic or acoustic_name == "AUTO":
        acoustic = cache["__acoustic__"]
    else:
        # Preset names a different acoustic than the candidate. We
        # intentionally still evaluate the candidate, not the preset's
        # default, so the gate measures what would actually ship.
        acoustic = cache["__acoustic__"]

    lm = None
    fusion_weight = 0.0
    if preset.lm and preset.fusion_weight > 0:
        if lm_name is None:
            # Preset wants an LM but the caller did not provide one.
            # Run acoustic-only with a printed warning, so the table
            # still produces a measurement instead of crashing.
            print(
                f"{_YELLOW}[release_gate] preset '{preset_name}' uses LM "
                f"'{preset.lm}' but --lm was not given. Running acoustic-only."
                f"{_RESET}",
                file=sys.stderr,
            )
        else:
            if lm_name not in cache:
                cache[lm_name] = _load_lm(resolve_model(lm_name), device)
            lm = cache[lm_name]
            fusion_weight = preset.fusion_weight

    return make_morseformer_decoder(
        acoustic,
        device=device,
        confidence_threshold=preset.confidence_threshold,
        digit_threshold=preset.digit_threshold,
        lm=lm,
        fusion_weight=fusion_weight,
        sample_rate=_SAMPLE_RATE,
        carrier_hz=_CARRIER_HZ,
    )


def _run_manifest_clip(
    cat: dict,
    cache: dict[str, Any],
    bench_entries: dict[str, Any],
    *,
    acoustic_name: str,
    lm_name: str | None,
    device: torch.device,
) -> CategoryResult:
    clip_id = cat["clip_id"]
    if clip_id not in bench_entries:
        raise SystemExit(
            f"[release_gate] clip '{clip_id}' (category '{cat['id']}') "
            f"not found in the bench manifest."
        )
    entry = bench_entries[clip_id]
    audio = _load_audio(entry.audio, _SAMPLE_RATE)
    gt = _normalize_prose(entry.gt.read_text(encoding="utf-8"), entry.lang)
    sample = Sample(
        sample_id=clip_id, text=gt, audio=audio, sample_rate=_SAMPLE_RATE
    )
    decoder = _decoder_for_preset(
        cat["preset"], cache,
        acoustic_name=acoustic_name, lm_name=lm_name, device=device,
    )
    result = run_benchmark(decoder, [sample])
    measured_pp = result.mean_cer * 100.0 if cat["metric"] == "cer" else (
        result.mean_wer * 100.0
    )
    threshold_pp = float(cat["max_pp"])
    return CategoryResult(
        id=cat["id"],
        kind=cat["kind"],
        metric=cat["metric"].upper(),
        measured=measured_pp,
        threshold=threshold_pp,
        baseline=float(cat.get("baseline_pp")) if cat.get("baseline_pp") is not None else None,
        unit="%",
        passed=measured_pp <= threshold_pp,
        extra={
            "clip_id": clip_id,
            "preset": cat["preset"],
            "hypothesis_head": result.per_sample[0].hypothesis[:120],
        },
    )


def _synth_silence_audio(
    rng: np.random.Generator, duration_s: float, *, rx_filter_bw: float = 500.0,
) -> np.ndarray:
    """Pure AWGN + RX bandpass, mimicking mode 0 of
    ``build_noise_only_validation``. No CW, no QRN — the strict
    silence FP guard."""
    n_samples = int(round(duration_s * _SAMPLE_RATE))
    audio = rng.normal(0.0, 0.1, size=n_samples).astype(np.float32)
    if rx_filter_bw > 0:
        from morse_synth.channel import _apply_rx_filter
        audio = _apply_rx_filter(
            audio, _SAMPLE_RATE, rx_filter_bw, _CARRIER_HZ
        ).astype(np.float32)
    return audio


def _run_silence(
    cat: dict,
    cache: dict[str, Any],
    *,
    acoustic_name: str,
    lm_name: str | None,
    device: torch.device,
) -> CategoryResult:
    decoder = _decoder_for_preset(
        cat["preset"], cache,
        acoustic_name=acoustic_name, lm_name=lm_name, device=device,
    )
    n = int(cat["n_samples"])
    seed = int(cat["seed"])
    rng = np.random.default_rng(seed)
    total_chars = 0
    sample_chars: list[int] = []
    for _ in range(n):
        audio = _synth_silence_audio(rng, duration_s=6.0)
        hyp = decoder(audio, _SAMPLE_RATE)
        chars = len(format_output(hyp).strip())
        sample_chars.append(chars)
        total_chars += chars
    mean_chars = total_chars / n
    threshold = float(cat["max_chars_per_sample"])
    return CategoryResult(
        id=cat["id"],
        kind=cat["kind"],
        metric="chars/sample",
        measured=mean_chars,
        threshold=threshold,
        baseline=float(cat.get("baseline_chars_per_sample"))
        if cat.get("baseline_chars_per_sample") is not None else None,
        unit="",
        passed=mean_chars <= threshold,
        extra={
            "n_samples": n,
            "preset": cat["preset"],
            "samples_with_emission": int(sum(c > 0 for c in sample_chars)),
        },
    )


_WORD_GAP_PROBES: tuple[str, ...] = (
    "CQ DE F4HYY",
    "GOOD MORNING DEAR",
    "QSL TU 73",
    "THE QUICK BROWN FOX",
    "WX HERE SUNNY 25 C",
    "RIG IS IC 7300",
    "ANT IS DIPOLE",
    "PWR 100 W",
    "RST 599 OM",
    "BK QRX 5 MIN BK",
    "ALL THE BEST 73",
    "DE F4HYY K",
    "QTH PARIS FRANCE",
    "NAME IS SERGE",
    "GLAD TO MEET YOU",
    "CONDX FAIR HR",
    "VY NICE QSO TU",
    "PSE QSP 73",
    "HW CPY OM",
    "GM GM ES GD QSO",
)


def _run_word_gap(
    cat: dict,
    cache: dict[str, Any],
    *,
    acoustic_name: str,
    lm_name: str | None,
    device: torch.device,
) -> CategoryResult:
    decoder = _decoder_for_preset(
        cat["preset"], cache,
        acoustic_name=acoustic_name, lm_name=lm_name, device=device,
    )
    n = int(cat["n_samples"])
    wpm = float(cat["wpm"])
    inflation = float(cat["word_gap_inflation"])
    seed = int(cat["seed"])
    if n > len(_WORD_GAP_PROBES):
        raise SystemExit(
            f"[release_gate] word_gap category '{cat['id']}' asks for "
            f"{n} samples but only {len(_WORD_GAP_PROBES)} probes exist."
        )
    rng = np.random.default_rng(seed)
    texts = list(_WORD_GAP_PROBES[:n])
    cer_total = 0.0
    for text in texts:
        op_seed = int(rng.integers(0, 2**31 - 1))
        operator = OperatorConfig(
            wpm=wpm, word_gap_inflation=inflation, seed=op_seed,
        )
        audio = render(
            text, operator=operator,
            freq=_CARRIER_HZ, sample_rate=_SAMPLE_RATE,
            channel=None,
        )
        hyp = decoder(audio.astype(np.float32), _SAMPLE_RATE)
        cer_total += character_error_rate(text, format_output(hyp))
    mean_cer_pp = (cer_total / n) * 100.0
    threshold_pp = float(cat["max_pp"])
    return CategoryResult(
        id=cat["id"],
        kind=cat["kind"],
        metric="CER",
        measured=mean_cer_pp,
        threshold=threshold_pp,
        baseline=float(cat.get("baseline_pp"))
        if cat.get("baseline_pp") is not None else None,
        unit="%",
        passed=mean_cer_pp <= threshold_pp,
        extra={
            "n_samples": n,
            "preset": cat["preset"],
            "wpm": wpm,
            "word_gap_inflation": inflation,
        },
    )


def _run_latency(
    cat: dict,
    cache: dict[str, Any],
    *,
    device: torch.device,
) -> CategoryResult:
    """Streaming latency RTF — model-only, preset-free smoke. Returns
    fail if a single window's decode takes longer than the hop budget
    in wall-clock on the inference device."""
    import time
    model = cache["__acoustic__"]
    window_s = float(cat["window_seconds"])
    hop_s = float(cat["hop_seconds"])
    warmup = int(cat.get("warmup", 3))
    trials = int(cat.get("trials", 10))
    audio = render(
        "CQ DE F4HYY K", operator=OperatorConfig(wpm=22.0, seed=0),
        freq=_CARRIER_HZ, sample_rate=_SAMPLE_RATE, channel=None,
    )
    target_n = int(round(window_s * _SAMPLE_RATE))
    if audio.size < target_n:
        audio = np.pad(audio, (0, target_n - audio.size))
    else:
        audio = audio[:target_n]
    features = extract_features(
        audio, _SAMPLE_RATE, FrontendConfig(frame_rate=500),
    )
    f = torch.from_numpy(features).unsqueeze(0).to(device)
    n_frames = torch.tensor([f.shape[1]], dtype=torch.long, device=device)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model.greedy_rnnt_decode(f, n_frames)
        _sync()
        times: list[float] = []
        for _ in range(trials):
            _sync()
            t0 = time.perf_counter()
            model.greedy_rnnt_decode(f, n_frames)
            _sync()
            times.append(time.perf_counter() - t0)
    decode_mean_ms = float(np.mean(times)) * 1000.0
    hop_budget_ms = hop_s * 1000.0
    rtf = decode_mean_ms / hop_budget_ms
    threshold = float(cat["max_rtf"])
    return CategoryResult(
        id=cat["id"],
        kind=cat["kind"],
        metric="RTF",
        measured=rtf,
        threshold=threshold,
        baseline=None,
        unit="",
        passed=rtf <= threshold,
        extra={
            "decode_ms_per_window": decode_mean_ms,
            "hop_budget_ms": hop_budget_ms,
            "device": str(device),
            "trials": trials,
        },
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _format_row(r: CategoryResult) -> str:
    status = (
        f"{_GREEN}PASS{_RESET}" if r.passed else f"{_RED}FAIL{_RESET}"
    )
    baseline = f"{r.baseline:.2f}" if r.baseline is not None else "—"
    return (
        f"  {status}  {r.id:<28} {r.metric:<14} "
        f"measured={r.measured:>7.3f}{r.unit:<2}  "
        f"max={r.threshold:>7.3f}{r.unit:<2}  baseline={baseline}"
    )


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_gate(
    manifest: dict,
    *,
    acoustic_name: str,
    lm_name: str | None,
    bench_manifest_path: Path,
    device: torch.device,
) -> tuple[list[CategoryResult], dict[str, Any]]:
    # Only load the bench-clip manifest when at least one category needs
    # it. Manifests that ship only synthetic + latency categories don't
    # require any real WAVs on disk — keeps smoke tests self-contained.
    needs_bench = any(
        cat["kind"] == "manifest_clip" for cat in manifest["categories"]
    )
    if needs_bench:
        if not bench_manifest_path.exists():
            raise SystemExit(
                f"[release_gate] bench manifest not found: {bench_manifest_path}"
            )
        repo_root = Path.cwd()
        bench_entries_raw = load_manifest(bench_manifest_path, repo_root)
        bench_entries = {e.id: e for e in bench_entries_raw}
    else:
        bench_entries = {}

    cache: dict[str, Any] = {}
    acoustic = _load_rnnt(resolve_model(acoustic_name), device)
    cache["__acoustic__"] = acoustic
    cache[acoustic_name] = acoustic

    results: list[CategoryResult] = []
    for cat in manifest["categories"]:
        kind = cat["kind"]
        if kind == "manifest_clip":
            r = _run_manifest_clip(
                cat, cache, bench_entries,
                acoustic_name=acoustic_name, lm_name=lm_name, device=device,
            )
        elif kind == "synthetic_silence":
            r = _run_silence(
                cat, cache,
                acoustic_name=acoustic_name, lm_name=lm_name, device=device,
            )
        elif kind == "synthetic_word_gap":
            r = _run_word_gap(
                cat, cache,
                acoustic_name=acoustic_name, lm_name=lm_name, device=device,
            )
        elif kind == "latency":
            r = _run_latency(cat, cache, device=device)
        else:
            raise SystemExit(
                f"[release_gate] unknown category kind '{kind}' (id='{cat['id']}')."
            )
        results.append(r)

    summary = {
        "version": manifest["version"],
        "acoustic": acoustic_name,
        "lm": lm_name,
        "manifest_calibration_date": manifest.get("calibration_date"),
        "manifest_baseline_acoustic": manifest.get("baseline_acoustic"),
        "device": str(device),
        "n_categories": len(results),
        "n_passed": sum(r.passed for r in results),
        "n_failed": sum(not r.passed for r in results),
        "all_passed": all(r.passed for r in results),
    }
    return results, summary


def write_report(
    out_dir: Path,
    acoustic_name: str,
    results: list[CategoryResult],
    summary: dict[str, Any],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = out_dir / f"release_gate_{acoustic_name}_{today}.json"
    payload = {
        "summary": summary,
        "categories": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--manifest", type=Path,
        default=Path("eval/release_gate_v1.json"),
        help="release-gate manifest (categories + thresholds).",
    )
    p.add_argument(
        "--bench-manifest", type=Path,
        default=Path("data/bench/manifest.jsonl"),
        help="bench-clip manifest (resolved by clip_id from --manifest).",
    )
    p.add_argument(
        "--acoustic", default=None,
        help="acoustic checkpoint to gate. Defaults to the manifest's "
             "baseline acoustic (calibration target).",
    )
    p.add_argument(
        "--lm", default=None,
        help="optional LM checkpoint. Required for any category whose "
             "preset has lm + fusion_weight > 0 (e.g. 'prose').",
    )
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument(
        "--out-dir", type=Path, default=Path("reports"),
        help="where to write the JSON report.",
    )
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(
            f"[release_gate] manifest not found: {args.manifest}",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    acoustic_name = args.acoustic or manifest.get("baseline_acoustic")
    if not acoustic_name:
        print(
            "[release_gate] no --acoustic given and manifest has no "
            "'baseline_acoustic'.",
            file=sys.stderr,
        )
        return 2

    device = torch.device(args.device or _auto_device())
    print(
        f"{_BOLD}[release_gate]{_RESET} manifest={args.manifest.name} "
        f"acoustic={acoustic_name} lm={args.lm or '—'} device={device}"
    )
    print(
        f"  baseline={manifest.get('baseline_acoustic')} "
        f"calibration_date={manifest.get('calibration_date')} "
        f"margin={manifest.get('non_regression_margin_pp')} pp"
    )
    print()
    results, summary = run_gate(
        manifest,
        acoustic_name=acoustic_name,
        lm_name=args.lm,
        bench_manifest_path=args.bench_manifest,
        device=device,
    )
    print(f"  {'status':<5}  {'category':<28} {'metric':<14} "
          f"{'measured':<14}    {'max':<14}    baseline")
    print(f"  {'-'*5}  {'-'*28} {'-'*14} {'-'*16}  {'-'*16}  {'-'*8}")
    for r in results:
        print(_format_row(r))
    print()
    verdict_colour = _GREEN if summary["all_passed"] else _RED
    verdict = "PASS" if summary["all_passed"] else "FAIL"
    print(
        f"{_BOLD}[release_gate] verdict: "
        f"{verdict_colour}{verdict}{_RESET}{_BOLD} "
        f"({summary['n_passed']}/{summary['n_categories']} categories)"
        f"{_RESET}"
    )

    report_path = write_report(args.out_dir, acoustic_name, results, summary)
    print(f"[release_gate] report → {report_path}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
