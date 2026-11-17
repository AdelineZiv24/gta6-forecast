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

What v3 adds over v2: a self-calibration layer, chosen by round 3's loss
decomposition. The decomposition (leave-one-out, by hour and by quantile)
showed v2 is well calibrated on average but carries a conditional level
bias: the biggest launches' days systematically come in below what a
local-constant kernel predicts for them (their hour-1 spike is
disproportionate to their day, and a target at the top of the size range
can only borrow curves from smaller launches), increasingly so as the day
wears on, while mid-size launches run slightly hot. Round 3 therefore lets
the case grade its own estimator: every comparable in the pool is
forecast from the others with the same machinery, the per-launch median
residuals (clipped at +/-1 log, averaged over hours 5+ on a ramp) are
regressed on log hour-1 size, and the target's five quantiles are shifted
in log space by half (lambda = 0.5) of the fitted residual at the target's
size, ramped in over hours 5-8 and capped at +/-0.6 log. Nothing about the
target's identity is used - the correction is estimated from the case's
own comparables at run time. Leave-one-out over the same 37 launches:

    segment            v2       v3
    all 37             0.4046   0.3940
    rolling family     0.2631   0.2614
    hour-1 >= 100k     0.2135   0.2077
    hour-1 >= 300k     0.1370   0.1251
    family & >= 100k   0.2031   0.2050

Lambda was picked from the middle of the interval that improves both
primary segments (family, >= 100k) without moving any guardrail segment
beyond noise: the response is smooth and monotone from 0.25 to 0.6, and
the one segment that pays for the correction, family & >= 100k (+0.0019,
inside the +/-0.003 noise band), pays more only past 0.6. Also tested in
round 3 and rejected with numbers: quantile-averaging ensembles of v2
with its plateau neighbours and with v1 (members too correlated; no
ensemble beat v2 on both primary segments), hour-block stretch and
normal-share schedules (all within +/-0.002 of v2), kernel-weighted
local residual fits (same gains as the global fit on the big launches,
worse on family & >= 100k), and an eve-ratio gate toward family-like
comparables (the eve ratio does not separate the rolling-unlock family:
family and non-family eve ranges overlap almost completely).

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

# Self-calibration (v3): within the case's own pool, each comparable is
# forecast from the others and its median residuals are regressed on its
# log hour-1 size; the target's log quantiles shift by CALIB_LAMBDA times
# the fitted residual at the target's size, ramped in over hours 5-8.
CALIB_LAMBDA = 0.5
CALIB_CLIP = 1.0
CALIB_MAX_SHIFT = 0.6


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


def kernel_weights(pool, lp_t, eve_t):
    """Gaussian similarity weights of the pool members for one target."""
    weights = []
    for f in pool:
        d_size = (f[1] - lp_t) / BW_SIZE
        d_eve = ((f[2] if f[2] is not None else eve_t) - eve_t) / BW_EVE
        # The floor keeps a far-away pool member at a whisper of weight instead
        # of letting every weight underflow to exactly zero.
        weights.append(max(math.exp(-(d_size * d_size + d_eve * d_eve)), 1e-12))
    return weights


def _calib_ramp(h):
    """How much of the calibration shift applies at hour h (0 to hour 4,
    ramping over hours 5-8, full from hour 8)."""
    return min(1.0, max(0.0, (h - 4) / 4.0))


def _member_residuals(pool, eve_t):
    """(pool index, log p1, mean clipped median residual) per pool member.

    Each member is forecast from the rest of the pool with the same kernel
    and quantile machinery, centred on the member itself; the residual at
    hour h is the member's actual log ratio minus the forecast median log
    ratio, clipped and averaged with the calibration ramp as weights.
    """
    out = []
    for i, f in enumerate(pool):
        sub = pool[:i] + pool[i + 1:]
        centre_eve = f[2] if f[2] is not None else eve_t
        w = kernel_weights(sub, f[1], centre_eve)
        num, den = 0.0, 0.0
        for h, r in f[3].items():
            g = _calib_ramp(h)
            if g <= 0:
                continue
            qs = hour_log_quantiles(sub, w, h, LEVELS)
            if qs is None:
                continue
            e = max(-CALIB_CLIP, min(CALIB_CLIP, r - qs[2]))
            num += g * e
            den += g
        if den > 0:
            out.append((f[1], num / den))
    return out


def calibration_shift(pool, lp_t, eve_t):
    """The fitted median residual (log units, before lambda) at the target.

    Regresses the members' residuals on their log hour-1 sizes and reads
    the fit at the target's size. Returns 0 when the pool is too small to
    say anything.
    """
    if len(pool) < 8:
        return 0.0
    res = _member_residuals(pool, eve_t)
    if len(res) < 8:
        return 0.0
    lp_bar = sum(f[1] for f in pool) / len(pool)
    xs = [lp - lp_bar for lp, _ in res]
    ys = [e for _, e in res]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    return a + b * (lp_t - lp_bar)


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

    weights = kernel_weights(pool, math.log(p1), target_eve)
    shift = 0.0
    if pool:
        shift = calibration_shift(pool, math.log(p1), target_eve)
        shift = max(-CALIB_MAX_SHIFT, min(CALIB_MAX_SHIFT, CALIB_LAMBDA * shift))

    rows = []
    for spec in case.get("forecast", []):
        h = spec["hour"]
        log_qs = hour_log_quantiles(pool, weights, h, levels) if pool else None
        if log_qs is None:
            qs = persistence_row(p1, h, levels)
        else:
            s = shift * _calib_ramp(h)
            qs = [p1 * math.exp(v + s) for v in log_qs]
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
