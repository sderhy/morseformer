"""Benchmark harness: run a decoder against a dataset, report metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from eval.datasets import Sample
from eval.metrics import callsign_scores, character_error_rate, word_error_rate

# Decoder signature: (audio, sample_rate) -> transcript text.
Decoder = Callable[[np.ndarray, int], str]


@dataclass
class SampleResult:
    sample_id: str
    reference: str
    hypothesis: str
    cer: float
    wer: float
    callsign_f1: float


@dataclass
class BenchmarkResult:
    n_samples: int
    mean_cer: float
    mean_wer: float
    mean_callsign_f1: float
    per_sample: list[SampleResult] = field(default_factory=list)


def run(decoder: Decoder, dataset: Iterable[Sample]) -> BenchmarkResult:
    per_sample: list[SampleResult] = []
    for sample in dataset:
        hyp = decoder(sample.audio, sample.sample_rate)
        cer = character_error_rate(sample.text, hyp)
        wer = word_error_rate(sample.text, hyp)
        cs = callsign_scores(sample.text, hyp)
        per_sample.append(
            SampleResult(
                sample_id=sample.sample_id,
                reference=sample.text,
                hypothesis=hyp,
                cer=cer,
                wer=wer,
                callsign_f1=cs.f1,
            )
        )
    n = len(per_sample)
    if n == 0:
        return BenchmarkResult(0, 0.0, 0.0, 0.0, [])
    return BenchmarkResult(
        n_samples=n,
        mean_cer=sum(r.cer for r in per_sample) / n,
        mean_wer=sum(r.wer for r in per_sample) / n,
        mean_callsign_f1=sum(r.callsign_f1 for r in per_sample) / n,
        per_sample=per_sample,
    )
