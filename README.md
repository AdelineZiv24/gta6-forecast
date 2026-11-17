# Clik GTA VI Twitch CCU forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AdelineZiv24/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/AdelineZiv24/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Built for the make.fun bounty [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

`main.py` (Python 3.13, standard library only, deterministic, no network) reads
one case file — everything recorded up to the end of release day's first hour —
and writes, for each of hours 2 to 24 (UTC), five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's concurrent viewers in Twitch's "Grand Theft Auto VI"
category.

## The model

A launch day's curve is far more regular relative to its own first hour than in
absolute terms, so the program works with log-ratios
`r_h = ln(viewers_h / viewers_1)`:

- Every past launch in the case's `comparables` contributes its ratio curve.
- Each comparable is weighted by similarity to this case on two axes: the size
  of its hour-1 print, and its release-eve ratio (the mean of the six hours
  before day start over hour 1) — the fingerprint of a rolling midnight unlock
  like GTA VI's, as opposed to a single global unlock, whose day looks entirely
  different from hour 1.
- Each hour's five quantiles are the average, in log space, of a weighted
  normal fit (blended toward the pooled moments when the weighted sample runs
  thin) and the weighted empirical quantiles, with the spread pulled 10%
  toward the median.
- Anything unusable (no comparables, no hour-1 print) falls back to a plain
  persistence curve, so the output is always a valid forecast.

## Local evaluation

There is no server-side practice answer key in the kit, so the model was
validated by simulation: each of the 37 comparable launches with an hour-1
print was turned into a case (its own curve hidden from its pool) and scored
with the contest's exact pinball loss on the log scale, leave-one-out:

| forecaster | all 37 | hour-1 ≥ 100k | hour-1 ≥ 300k |
|---|---|---|---|
| this entry | **0.458** | **0.222** | **0.155** |
| house baseline (persistence) | 0.930 | 0.261 | 0.226 |

A single case runs in about 0.06 s locally (the cap is 8 s), and two runs of
any case produce byte-identical output.

## v2 (round 2 retune)

Same architecture, constants re-chosen by a second leave-one-out pass that
also tracked the rolling-midnight-unlock family (GTA VI's family) as its own
segment. Every constant sits in the middle of a plateau that beats v1 on all
five segments:

| segment | v1 | v2 |
|---|---|---|
| all 37 | 0.4575 | **0.4046** |
| rolling-unlock family (12) | 0.2732 | **0.2631** |
| hour-1 ≥ 100k | 0.2218 | **0.2135** |
| hour-1 ≥ 300k | 0.1548 | **0.1370** |
| family & ≥ 100k | 0.2180 | **0.2031** |

Changes: size-kernel bandwidth 2.5 → 1.5, eve-kernel bandwidth 0.8 → 3.0,
normal-fit share 0.5 → 0.65, effective-sample blending threshold 10 → 5.
Extra similarity features (eve-ramp slope, 24 h pre-launch heat, viewers per
channel), a skewed two-piece fit, hour smoothing and a spread floor were all
tested against the same cross-validation and rejected — each hurt a primary
segment or bought nothing. Anchor check: at a synthetic hour-1 print of
0.8M–2.5M against the full pool, the kernel's effective sample stays ≈12–6
and the implied curve keeps the classic dip-then-evening-peak shape.

## v3 (round 3 self-calibration)

A third pass decomposed v2's loss by hour and quantile: calibrated on
average, but conditionally biased — the largest launches' days land
systematically below a local-constant kernel's forecast (a top-of-range
target can only borrow smaller launches' curves), increasingly late in the
day. v3 adds a calibration layer that is estimated from the case itself at
run time: every comparable is forecast from the others, the per-launch
median residuals (clipped at ±1 log) are regressed on log hour-1 size, and
the target's quantiles shift by half the fitted residual at its own size,
ramped in over hours 5–8 and capped at ±0.6 log.

| segment | v2 | v3 |
|---|---|---|
| all 37 | 0.4046 | **0.3940** |
| rolling-unlock family (12) | 0.2631 | **0.2614** |
| hour-1 ≥ 100k | 0.2135 | **0.2077** |
| hour-1 ≥ 300k | 0.1370 | **0.1251** |
| family & ≥ 100k | **0.2031** | 0.2050 |

The damping factor (0.5) is the middle of the range that improves both
primary segments (family, ≥ 100k); family & ≥ 100k moves +0.0019, inside
the noise band, and is the only segment that pays. Also tested and
rejected in round 3: quantile-averaging ensembles of v2 with its plateau
neighbours and with v1 (no ensemble beat v2 on both primaries), hour-block
stretch / normal-share schedules (within ±0.002 throughout), and
kernel-weighted local residual fits (same big-launch gains, worse on
family & ≥ 100k). A full case now runs in about 0.1 s.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```
