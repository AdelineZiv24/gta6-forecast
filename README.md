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

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```
