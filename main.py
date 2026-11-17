"""gta6-twitch-ccu-forecast-01 entry, v2: a similarity-weighted comparable-launch forecaster.

The idea, in one paragraph. A launch day's Twitch curve, hour by hour, is hard to
predict in absolute terms but much more regular relative to its own first hour:
most big console launches dip through the UTC morning, then climb as North
America wakes up, peaking in the 19:00-23:00 UTC overlap. So this program treats
every past launch in the case's `comparables` as a sample of the log-ratio
curve r_h = ln(viewers_h / viewers_1), weights each sample by how similar that
launch is to this case (size of its hour-1 print, and how strong its release-eve
ramp was relative to hour 1 - the fingerprint of a rolling midnight unlock),
and reads the five scored quantiles of hour h off the weighted distribution:
a weighted normal fit mixed with the weighted empirical quantiles, with the
spread pulled 10% toward the median. The forecast is hour 1's print times exp
of those quantiles.

What v2 changes over v1 (same architecture, retuned constants). Round 2
re-ran the leave-one-launch-out cross-validation over the 37 comparable
launches with an hour-1 print, scoring with the contest's exact pinball loss,
and chose every constant from the middle of a plateau that beats v1 on all
five tracked segments (all launches; the rolling-midnight-unlock family;
hour-1 >= 100k; hour-1 >= 300k; family and >= 100k):

    segment            v1       v2
    all 37             0.4575   0.4046
    rolling family     0.2732   0.2631
    hour-1 >= 100k     0.2218   0.2135
    hour-1 >= 300k     0.1548   0.1370
    family & >= 100k   0.2180   0.2031

The size kernel is narrower (2.5 -> 1.5 log units: a launch's own size class
says more about its curve than v1 let it), the eve kernel is wider (0.8 ->
3.0: the eve ratio still separates the extreme unlock shapes, but at v1's
width it was throwing away curve information from launches whose fingerprint
merely differed), the normal fit's share of the mix rises (0.5 -> 0.65: the
weighted empirical quantiles of a concentrated sample are jumpy), and the
effective-sample blending toward the pooled moments now starts below 5
instead of 10 (at the chosen bandwidths the weighting rarely concentrates
enough to need it sooner). Tested and rejected in round 2, with numbers in
the round-2 notes: extra kernel features (eve-ramp slope, 24 h pre-launch
heat, viewers per channel), a two-piece (skewed) normal fit, smoothing the
per-hour moments across neighbouring hours, shrinking the mean toward the
pooled mean, and a floor under the spread tied to the pooled spread - each
either hurt a primary segment or bought nothing.

Anchor behaviour, checked on synthetic targets against the full pool: at an
hour-1 print of 0.8M-2.5M the kernel's effective sample stays between about
12 and 6 (no collapse onto one or two comparables), and the implied median
curve keeps the classic shape (dip to ~0.6x by midday UTC, peak ~1.0-1.2x
at hours 19-21). The 90% band narrows as the print leaves the pool's range;
that is the estimator the cross-validation prefers, and the mega-launch
segment is where it gains most over v1.

Runs as `python3 -P -s /bot/main.py` (Python 3.13, standard library only, no
network). Deterministic: a pure function of the case file. If anything about a
case is unusable (no comparables, no hour-1 print), it falls back to the plain
persistence curve so the output is always a valid forecast.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
Z = [_NORM.inv_cdf(t) for t in LEVELS]

# Similarity kernel bandwidths, in log-print units and in eve-ratio units.
BW_SIZE = 1.5
BW_EVE = 3.0
# The normal half of each hour's distribution is blended toward the pooled
# (unweighted) moments when the weighted sample's effective size falls below
# this, so one dominant comparable cannot dictate the spread alone.
MIN_EFFECTIVE = 5.0
# Share of the normal fit in the final quantile (the rest is empirical).
NORMAL_SHARE = 0.65
# The mixed quantiles are pulled this far toward the median (1 = untouched).
STRETCH = 0.9


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def parse_ts(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def hourly_map(points):
    """{datetime: viewers} from a case's hourly-style point list."""
    out = {}
    for p in points or []:
        try:
            out[parse_ts(p["from"])] = p["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    return out


def eve_ratio(hour_map, day_start, p1):
    """Mean viewers over the 6 hours before day_start, relative to hour 1's print.

    A rolling midnight unlock (New Zealand first) leaves a strong ramp in those
    hours; a single global unlock leaves almost nothing. Returns None when the
    window is too sparse to say.
    """
    vals = []
    for k in range(1, 7):
        v = hour_map.get(day_start - timedelta(hours=k))
        if v is not None:
            vals.append(v)
    if len(vals) < 4:
        return None
    return (sum(vals) / len(vals)) / p1


def launch_features(day_start, hour_map):
    """(p1, log p1, eve ratio, {hour: log ratio}) for one launch's curve, or None."""
    p1 = hour_map.get(day_start)
    if not p1 or p1 < 1:
        return None
    ratios = {}
    for h in range(2, 25):
        v = hour_map.get(day_start + timedelta(hours=h - 1))
        if v:
            ratios[h] = math.log(v / p1)
    if len(ratios) < 12:
        return None
    return p1, math.log(p1), eve_ratio(hour_map, day_start, p1), ratios


def weighted_quantile(pairs, tau):
    """The tau quantile of (value, weight) pairs, linearly interpolated."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc / total >= tau:
            if i == 0:
                return v
            prev_v = pairs[i - 1][0]
            frac = (tau * total - (acc - w)) / w if w > 0 else 0.0
            return prev_v + frac * (v - prev_v)
    return pairs[-1][0]


def weighted_moments(pairs):
    total = sum(w for _, w in pairs)
    mu = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / total
    return mu, math.sqrt(var)


def hour_log_quantiles(pool, weights, hour, levels):
    """The mixed normal/empirical log-ratio quantiles for one hour."""
    pairs = [(f[3][hour], w) for f, w in zip(pool, weights) if hour in f[3]]
    if not pairs:
        return None
    if sum(w for _, w in pairs) <= 0:
        pairs = [(v, 1.0) for v, _ in pairs]
    mu, sd = weighted_moments(pairs)
    eff = (sum(w for _, w in pairs) ** 2) / sum(w * w for _, w in pairs)
    if eff < MIN_EFFECTIVE:
        mu0, sd0 = weighted_moments([(v, 1.0) for v, _ in pairs])
        a = eff / MIN_EFFECTIVE
        mu = a * mu + (1 - a) * mu0
        sd = a * sd + (1 - a) * sd0
    # A weighted sample's extreme order statistics are jumpy: keep the empirical
    # quantiles within three (blended) standard deviations of the fitted centre.
    lo, hi = mu - 3 * sd, mu + 3 * sd
    empirical = [min(max(weighted_quantile(pairs, t), lo), hi) for t in levels]
    zs = [_NORM.inv_cdf(t) for t in levels]
    normal = [mu + z * sd for z in zs]
    mixed = [NORMAL_SHARE * n + (1 - NORMAL_SHARE) * e for n, e in zip(normal, empirical)]
    med = mixed[len(levels) // 2] if len(levels) % 2 == 1 else 0.5 * (mixed[len(levels) // 2 - 1] + mixed[len(levels) // 2])
    return [med + STRETCH * (v - med) for v in mixed]


def persistence_row(p1, hour, levels):
    """The fallback: every hour stays at hour 1's print, spread widening slowly."""
    return [p1 * math.exp(_NORM.inv_cdf(t) * 0.15 * math.sqrt(hour - 1)) for t in levels]


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    p1 = max(1, known["viewers"]) if known and known.get("viewers") else 1
    day_start = parse_ts(case["day_start"])

    # The target's own eve ratio, from its whole-hour history before day_start.
    target_map = hourly_map(case.get("history_hourly"))
    target_eve = eve_ratio(target_map, day_start, p1)

    pool = []
    for comp in case.get("comparables") or []:
        try:
            f = launch_features(parse_ts(comp["day_start"]), hourly_map(comp.get("hourly")))
        except (KeyError, TypeError, ValueError):
            f = None
        if f is not None:
            pool.append(f)
    if target_eve is None and pool:
        eves = sorted(f[2] for f in pool if f[2] is not None)
        target_eve = eves[len(eves) // 2] if eves else 1.0
    if target_eve is None:
        target_eve = 1.0

    weights = []
    for f in pool:
        d_size = (f[1] - math.log(p1)) / BW_SIZE
        d_eve = ((f[2] if f[2] is not None else target_eve) - target_eve) / BW_EVE
        # The floor keeps a far-away pool member at a whisper of weight instead
        # of letting every weight underflow to exactly zero.
        weights.append(max(math.exp(-(d_size * d_size + d_eve * d_eve)), 1e-12))

    rows = []
    for spec in case.get("forecast", []):
        h = spec["hour"]
        log_qs = hour_log_quantiles(pool, weights, h, levels) if pool else None
        if log_qs is None:
            qs = persistence_row(p1, h, levels)
        else:
            qs = [p1 * math.exp(v) for v in log_qs]
        # Enforce the output contract: finite, in (0, 1e9], non-decreasing.
        clean = []
        for q in qs:
            q = min(max(q, 1e-6), 1e9)
            if clean and q < clean[-1]:
                q = clean[-1]
            clean.append(round2(q))
        rows.append({"hour": h, "q": clean})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
