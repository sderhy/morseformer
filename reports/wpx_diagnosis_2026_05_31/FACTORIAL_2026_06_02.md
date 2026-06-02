# Factorial bandwidth × threshold — decision: contest = 100 Hz

2026-06-02. Closes the open item from `OPTION_C_RESULTS.md`: the first pass
shipped `contest`=60 Hz on vibes (DA1A win) but the LCWO non-regression bench
showed a `websdr-fav22` regression that was never isolated. This factorial
decomposes bandwidth vs threshold and adds the missing bw=100 evidence.

## Setup

Model `rnnt_phase11b`, full LCWO manifest (7 clips), one process / one model
load. Script: `/tmp/bench_bw_factorial.py` + `/tmp/decode_contest_bw.py`.

4 configs:
- `bw200_c06` — bw=200 conf=0.6 digit=0.90 (= live default, baseline)
- `bw100_c06` — bw=100 conf=0.6 digit=0.90 (pure bandwidth, compromise)
- `bw60_c06`  — bw=60  conf=0.6 digit=0.90 (pure bandwidth, narrow)
- `bw60_c05`  — bw=60  conf=0.5 digit=0.80 (= the shipped contest preset)

## LCWO clean/real-OTA bench

| config    | mean CER | mean WER | websdr-fav22 CER | websdr WER |
|-----------|----------|----------|------------------|------------|
| bw200_c06 | 2.56     | 8.89     | 9.33             | 21.05      |
| bw100_c06 | **2.56** | **8.77** | **9.11** (-0.22) | **19.74** (-1.32) |
| bw60_c06  | 2.88 (+0.32) | 11.11 (+2.22) | 11.33 (+2.00) | 35.53 (+14.47) |
| bw60_c05  | 2.93 (+0.37) | 11.16 (+2.27) | 12.00 (+2.67) | 36.84 (+15.79) |

Decomposition:
- **Bandwidth is the lever, not the threshold.** bw60→bw60+conf0.5 adds only
  +0.05 CER. The whole contest-preset regression is bw=60.
- **The regression is one clip.** bw=60 costs +2.0 CER / +14.5 WER on
  `websdr-fav22` (a real-OTA 5-letter mono-station clip); the other 6 clips
  move ≤0.15 CER. Over-narrowing a single clean station clips its sidebands.
- **bw=100 has zero downside.** Same mean CER as bw=200, *better* mean WER,
  and it even improves websdr (−0.22 CER / −1.32 WER).

## Contest recordings (qualitative — no ground truth)

| file   | bw=200          | bw=100                              | bw=60                          |
|--------|-----------------|-------------------------------------|--------------------------------|
| DA1A   | inexploitable   | **readable QSO** (5NN 185 … DA1R … CQ DA1A TEST … 5NN 1186 … DA1R) | readable QSO, F4HYY/DA1A slightly crisper, more trailing garbage |
| cwcww5 | `COT` (hallu)   | **`CQ`** ✓ + `F4HYY588` clean       | `CMT` (hallu) + `HHA7A7A` doubled — **regresses** |
| cwcww11| `5NN T11 … OM7M`| `5NN T11 … OM7M` ✓                   | `5NN T11 … OM7M` ✓ (≈ tied)    |

bw=100 recovers the DA1A QSO that motivated this work **and** beats bw=60 on
cwcww5 (where 60 re-hallucinates `CMT`).

## Decision

**`contest` preset → bandwidth_hz = 100.0** (was 60.0). Strictly dominates:
- vs bw=200: same clean CER, better WER, recovers the contest files.
- vs bw=60: same contest recovery, none of the websdr regression, better cwcww5.

`live`/`prose`/`conservative` stay at 200 Hz (the global gain is within noise;
don't move the universal default for one clip). The `--bandwidth` override
(CLI flag + GUI BW field) keeps 60 Hz reachable for very dense pile-ups, where
the tighter band buys DA1A callsign crispness at the cost of some accuracy on
isolated stations.

Tests: `test_contest_preset_uses_narrow_bandwidth` now asserts 100.0; 22/22 CLI
tests pass.
