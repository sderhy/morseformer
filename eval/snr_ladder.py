"""SNR ladder: CER/WER/callsign-F1 as a function of SNR.

The primary Phase 1 benchmark. Successive phases must drive the CER
curve down and to the left (i.e. better at lower SNR).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eval.benchmark import BenchmarkResult, Decoder, run
from eval.datasets import generate_noisy


@dataclass
class SnrLadderResult:
    """One BenchmarkResult per SNR step, indexed by SNR in dB."""

    snrs_db: list[float]
    per_snr: dict[float, BenchmarkResult] = field(default_factory=dict)

    def as_rows(self) -> list[dict]:
        """Return a list of {snr_db, mean_cer, mean_wer, mean_callsign_f1, n}."""
        rows = []
        for snr in self.snrs_db:
            r = self.per_snr[snr]
            rows.append(
                {
                    "snr_db": snr,
                    "n": r.n_samples,
                    "mean_cer": r.mean_cer,
                    "mean_wer": r.mean_wer,
                    "mean_callsign_f1": r.mean_callsign_f1,
                }
            )
        return rows

    def format_table(self) -> str:
        """Pretty-print the ladder as a fixed-width table."""
        header = "  SNR (dB) |  n  |    CER   |    WER   | Callsign F1"
        sep = "  ---------+-----+----------+----------+-------------"
        lines = [header, sep]
        for snr in self.snrs_db:
            r = self.per_snr[snr]
            snr_str = "   +inf" if snr == float("inf") else f"{snr:+7.1f}"
            lines.append(
                f"   {snr_str} | {r.n_samples:3d} | {r.mean_cer:8.4f} "
                f"| {r.mean_wer:8.4f} | {r.mean_callsign_f1:11.4f}"
            )
        return "\n".join(lines)


def run_snr_ladder(
    decoder: Decoder,
    snrs_db: list[float],
    *,
    n_per_snr: int = 20,
    seed: int = 42,
    sample_rate: int = 8000,
) -> SnrLadderResult:
    """Benchmark a decoder across a list of SNR levels."""
    per_snr: dict[float, BenchmarkResult] = {}
    for snr in snrs_db:
        # Derive a per-SNR seed so that adding an SNR step does not change
        # the samples generated at other SNRs.
        snr_seed = seed + int((snr if snr != float("inf") else 1000) * 100) & 0x7FFFFFFF
        dataset = generate_noisy(
            n=n_per_snr,
            snr_db=snr,
            sample_rate=sample_rate,
            seed=snr_seed,
        )
        per_snr[snr] = run(decoder, dataset)
    return SnrLadderResult(snrs_db=list(snrs_db), per_snr=per_snr)
