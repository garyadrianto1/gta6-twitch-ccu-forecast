"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts, for hours 2..24 of Grand Theft Auto VI's release day (UTC), five
quantiles (5/25/50/75/95%) of the Twitch category's concurrent viewers, from
the case's hour-1 print and the launch-day curves of past big launches
(the case's `comparables`).

Model: a local linear regression in ln(hour 1), fitted per hour over the
comparables. For each comparable launch with a usable release-day curve,
the log-ratio r_h = ln(viewers_h / viewers_hour1). Three facts drive the
design, all measured on the kit's practice data:

1. The afternoon rise is almost entirely explained by hour-1 size
   (correlation -0.90 between ln(hour 1) and the median rise over hours
   15-20): a small hour 1 means the game unlocked later in the day, so the
   afternoon explodes; a big hour 1 means it unlocked at midnight and the
   afternoon stays flat. So the centre of each hour is a least-squares line
   in ln(hour 1), r_h = a_h + b_h * ln(hour1). Early hours have slope ~0 (a
   plain dip); afternoon hours have slope -0.55 to -0.94 with R2 0.57-0.88.
2. The residual spread is heteroscedastic: launches near the target's
   hour-1 size have a much tighter spread (residual SD 0.56 for big
   launches) than far ones (SD 1.42 for small ones). So both the centre line
   and the residual quantiles are computed locally: only comparables within
   |ln(hour1) - ln(target hour1)| <= 2.0 are used to fit the centre line,
   and only those within <= 1.5 for the residual quantiles, each falling
   back to all comparables when too few are near.
3. Two band-gated second regressors, each fitted as a two-feature plane
   (ln hour 1 + feature) where the feature's signal lives:
   - Big targets (hour 1 > 100k, the GTA VI analogues): the unlock hour
     from the case's notes / the comparables' unlock research notes —
     rolling midnight local (0.0) vs a single global unlock time. It
     separates flat rolling launches from sharp simultaneous ones, which
     ln(hour 1) alone cannot time. Only ~9 comparables carry a parseable
     unlock, so the plane needs just 8 points.
   - Small targets (hour 1 < 10k): the 7-day pre-launch level relative to
     hour 1 (log ratio), from the case's history / the comparables' hourly
     data. It separates explosive small launches (quiet before, huge
     afternoon) from merely small ones; the plane needs 10 points.
   Mid targets use the one-feature line. A missing feature (unparseable
   unlock, no pre-launch data) falls back to the one-feature line.

The quantiles are the empirical quantiles of those local residuals,
widened by 1.1 in log units (in-sample residuals underestimate the true
spread), so the spread is calibrated to what similar launches actually
did instead of a fixed sigma. Everything is computed at run time from the case the grader
gives, so the entry adapts to the final case's comparables. Deterministic:
sorted iteration, no randomness, no network, no wall-clock use.

Validation (leave-one-launch-out over the kit's 37 usable comparables,
exact pinball loss on the log scale): mean loss 0.333, against 0.385 for
the previous entry (E7) and 0.930 for the house baseline (starter) — a
64% improvement over the baseline; on big launches (hour 1 above 100k,
the GTA VI analogues) 0.188 against the baseline's 0.261 and E7's 0.210.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history", "history_hourly",
    "streams", "categories_hourly", "comparables", "notes"}, where `known`
    holds hour 1's print {hour: 1, from, until, viewers, channels, reads}
    and `forecast` lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2,
    "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24
    each once, no other key, each q five finite numbers above 0 and at most
    1e9 (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (fallback spread only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Fallback spread (house baseline's), in natural-log units per sqrt hour.
SIGMA = 0.15
# A comparable counts as usable when hour 1 is present and at least this
# many of hours 2..24 have points.
MIN_DAY_HOURS = 18
# An hour needs at least this many (ln(hour1), log-ratio) points for its own
# regression; otherwise the median curve is used for that hour.
MIN_REGRESSION_POINTS = 8
# An hour needs at least this many residual values for its own empirical
# quantiles; otherwise the nearest usable hour's spread is borrowed.
MIN_RESIDUALS = 5
# Local window (in ln(hour1)) for fitting the centre line.
CENTRE_WINDOW = 2.0
# Local window (in ln(hour1)) for the residual quantiles.
SPREAD_WINDOW = 1.5

# The residual quantiles are empirical quantiles of in-sample residuals,
# which underestimate the true spread; widen them by 1.1 in log units.
# LOLO: 1.0 -> 0.3361, 1.1 -> 0.3335, 1.2 -> 0.3327, 1.3 -> 0.3331, but
# the big-launch band starts degrading past 1.1, so 1.1 keeps it at its
# best (0.1882) while fixing most of the narrow tails.
SPREAD_MULT = 1.1

# Band boundaries (natural log of hour-1 viewers): the same bands the
# ledger reports. Big launches (hour 1 > 100k) are the GTA VI analogues;
# small launches (hour 1 < 10k) are the explosive ones.
BIG_BAND = 11.512925  # ln(100000)
SMALL_BAND = 9.210340  # ln(10000)


def parse_unlock(text):
    """Unlock hour (hours after day_start 00:00Z) from free text, or None.

    Rolling midnight local -> 0.0 (already unlocking by 00:00Z in the
    earliest zones). A specific global unlock time (ISO timestamp) -> its
    UTC hour. Early-access-contaminated or unverified -> None (no signal).
    Checks rolling before any timestamp, because a rolling note also names
    the first (New Zealand) unlock time.
    """
    if not text or not isinstance(text, str):
        return None
    t = text.lower()
    if "early access" in t or "early-access" in t:
        return None
    if "rolling" in t or "midnight local" in t:
        return 0.0
    m = re.search(r"\d{4}-\d{2}-\d{2}T(\d{2}):(\d{2})Z", text)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0
    return None


def prelaunch_ratio(day_start, points, hours):
    """Mean viewers in the `hours` before day_start, divided by hour-1 viewers.

    `points` is a list of {"at"/"from": iso, "viewers": n}. Returns None when
    there are no usable pre-launch points. The caller divides by h1; this
    returns the raw mean so the caller can take the log-ratio.
    """
    if day_start is None:
        return None
    vals = []
    for p in points or []:
        at = p.get("at") or p.get("from")
        try:
            t = datetime.fromisoformat(str(at).replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            continue
        v = p.get("viewers")
        if v is None or v < 1:
            continue
        if day_start - timedelta(hours=hours) <= t < day_start:
            vals.append(v)
    if not vals:
        return None
    return sum(vals) / len(vals)


def fit_line2(points, min_points=8):
    """Least-squares plane r = a + b*x + c*u over (x, u, r) points, or None.

    Solved by Gaussian elimination on the 3x3 normal equations. Needs at
    least min_points points (3 parameters, and the local window thins the
    data).
    """
    n = len(points)
    if n < min_points:
        return None
    # Normal equations: M @ [a, b, c] = v, with rows [1, x, u].
    M = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    for x, u, r in points:
        row = [1.0, x, u]
        for i in range(3):
            v[i] += row[i] * r
            for j in range(3):
                M[i][j] += row[i] * row[j]
    # Gaussian elimination with partial pivoting.
    A = [M[i] + [v[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda k: abs(A[k][col]))
        if abs(A[piv][col]) < 1e-12:
            return None
        A[col], A[piv] = A[piv], A[col]
        for k in range(col + 1, 3):
            f = A[k][col] / A[col][col]
            for j in range(col, 4):
                A[k][j] -= f * A[col][j]
    sol = [0.0] * 3
    for i in (2, 1, 0):
        s = A[i][3] - sum(A[i][j] * sol[j] for j in range(i + 1, 3))
        sol[i] = s / A[i][i]
    return tuple(sol)


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def quantile(sorted_values, tau):
    """Linear-interpolation quantile (type 7) of an already-sorted list."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("quantile of empty list")
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_values[-1]
    return sorted_values[lo] * (1 - frac) + sorted_values[lo + 1] * frac


def day_curve(comp):
    """Hourly viewers for the 24h of the launch day (UTC), index 0 = hour 1.

    Returns a list of 24 values (None where the source has no point).
    """
    try:
        ds = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        return [None] * 24
    by_from = {}
    for h in comp.get("hourly") or []:
        try:
            by_from[h["from"]] = h["viewers"]
        except (KeyError, TypeError):
            continue
    out = []
    for h in range(24):
        key = (ds + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z")
        out.append(by_from.get(key))
    return out


def regression_points(comparables):
    """Per hour, the (ln(hour1), log-ratio) points of usable comparables.

    Returns {hour: [(x, r), ...]} for hours 2..24.
    """
    points = {h: [] for h in range(2, 25)}
    for comp in comparables or []:
        curve = day_curve(comp)
        h1 = curve[0]
        if not h1 or h1 < 1:
            continue
        present = sum(1 for v in curve[1:24] if v is not None)
        if present < MIN_DAY_HOURS:
            continue
        x = math.log(h1)
        for h in range(2, 25):
            v = curve[h - 1]
            if v is not None and v >= 1:
                points[h].append((x, math.log(v / h1)))
    return points


def fit_line(points):
    """Least-squares line r = a + b*x over (x, r) points, or None if too few."""
    n = len(points)
    if n < MIN_REGRESSION_POINTS:
        return None
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 0:
        return None
    b = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
    return (my - b * mx, b)


def hour_quantiles(p1, centre, spreads):
    """Five quantiles for one hour: p1 * exp(centre + spread_tau).

    Enforces non-decreasing order, > 0 and <= 1e9 after rounding.
    """
    qs = []
    for s in spreads:
        q = p1 * math.exp(centre + s)
        q = round2(q)
        if q <= 0:
            q = 0.01
        if q > 1e9:
            q = 1e9
        qs.append(q)
    # Enforce non-decreasing (rounding/clamping must not break the order).
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def persistence_quantiles(p1, h):
    """House-baseline fallback: every hour stays at hour 1's print, spread
    widening like a random walk's in log space."""
    return hour_quantiles(p1, 0.0, [z * SIGMA * math.sqrt(h - 1) for z in Z])


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    x_target = math.log(p1)

    points = regression_points(case.get("comparables"))
    # Two second-regressor features, one per band where it helps:
    #   big targets (hour 1 > 100k): the unlock hour — rolling midnight
    #     (0.0) vs a single global unlock time. It separates the flat
    #     rolling launches from the sharp simultaneous ones, which ln(hour 1)
    #     alone cannot time.
    #   small targets (hour 1 < 10k): the 7-day pre-launch level relative
    #     to hour 1 (log ratio) — it separates explosive small launches
    #     (quiet before, huge afternoon) from merely small ones.
    # Mid targets use the E7 one-feature line. Either feature falling back
    # to None (unparseable / no pre-launch data) also falls back to E7.
    u_unlock_target = parse_unlock(case.get("notes"))
    try:
        ds_target = datetime.fromisoformat(case["day_start"].replace("Z", "+00:00"))
    except (KeyError, ValueError, AttributeError):
        ds_target = None
    pre_target = prelaunch_ratio(ds_target, case.get("history"), 24 * 7)
    u_pre_target = (math.log(pre_target / p1)
                    if pre_target is not None and pre_target > 0 else None)
    # Per-comparable feature values, computed once.
    u_unlock_comp = {}
    u_pre_comp = {}
    for comp in case.get("comparables") or []:
        u_unlock_comp[id(comp)] = parse_unlock(comp.get("unlock"))
        try:
            ds = datetime.fromisoformat(comp["day_start"].replace("Z", "+00:00"))
        except (KeyError, ValueError, AttributeError):
            ds = None
        curve = day_curve(comp)
        h1c = curve[0]
        pre = prelaunch_ratio(ds, comp.get("hourly"), 24 * 7)
        u_pre_comp[id(comp)] = (math.log(pre / h1c)
                                if pre and h1c and pre > 0 else None)

    def two_feature_points(u_comp):
        """Per hour, (ln h1, feature, log-ratio) for comparables that have
        the feature; same usability filter as regression_points."""
        out = {h: [] for h in range(2, 25)}
        for comp in case.get("comparables") or []:
            u = u_comp.get(id(comp))
            if u is None:
                continue
            curve = day_curve(comp)
            h1c = curve[0]
            if not h1c or h1c < 1:
                continue
            if sum(1 for v in curve[1:24] if v is not None) < MIN_DAY_HOURS:
                continue
            x = math.log(h1c)
            for h in range(2, 25):
                v = curve[h - 1]
                if v is not None and v >= 1:
                    out[h].append((x, u, math.log(v / h1c)))
        return out

    points2u = two_feature_points(u_unlock_comp)
    points2p = two_feature_points(u_pre_comp)
    # Centre and residual spreads per hour.
    centre = {}
    spreads = {}
    for h in range(2, 25):
        pts = points[h]
        if len(pts) < MIN_RESIDUALS:
            continue
        # Local centre: fit the line on launches near the target's hour-1
        # size; fall back to all points when too few are near.
        fit_pts = pts
        near_centre = [p for p in pts if abs(p[0] - x_target) <= CENTRE_WINDOW]
        if len(near_centre) >= MIN_REGRESSION_POINTS:
            fit_pts = near_centre
        line = fit_line(fit_pts)
        # Band-gated second regressor: a two-feature plane (ln h1 +
        # feature) where the feature is available for the target and the
        # band; otherwise the one-feature line.
        line2 = None
        pts2 = None
        u_t = None
        min_pts = 8
        if x_target >= BIG_BAND and u_unlock_target is not None:
            pts2 = points2u[h]
            u_t = u_unlock_target
            min_pts = 8  # only ~9 classified launches exist; 8 is the floor
        elif x_target < SMALL_BAND and u_pre_target is not None:
            pts2 = points2p[h]
            u_t = u_pre_target
            min_pts = 10  # pre-launch data exists for all launches; prefer
            # the global fit over a thin local one
        if pts2 is not None:
            near2 = [p for p in pts2 if abs(p[0] - x_target) <= CENTRE_WINDOW]
            cand = near2 if len(near2) >= min_pts else pts2
            line2 = fit_line2(cand, min_pts)
        if line2 is not None:
            a, b, c = line2
            rs = sorted(r for _, r in pts)
            centre[h] = min(max(a + b * x_target + c * u_t, rs[0]), rs[-1])
            residuals = sorted(r - (a + b * x + c * u) for x, u, r in pts2)
            # Heteroscedastic spread: launches near the target's hour-1
            # size have a much tighter residual spread than far ones.
            near = sorted(r - (a + b * x + c * u) for x, u, r in pts2
                          if abs(x - x_target) <= SPREAD_WINDOW)
            if len(near) >= MIN_RESIDUALS:
                residuals = near
        elif line is None:
            # Too few points for a line: fall back to the median curve.
            med = quantile(sorted(r for _, r in pts), 0.5)
            centre[h] = med
            residuals = sorted(r - med for _, r in pts)
        else:
            a, b = line
            rs = sorted(r for _, r in pts)
            centre[h] = min(max(a + b * x_target, rs[0]), rs[-1])
            residuals = sorted(r - (a + b * x) for x, r in pts)
            # Heteroscedastic spread: launches near the target's hour-1
            # size have a much tighter residual spread than far ones.
            near = sorted(r - (a + b * x) for x, r in pts
                          if abs(x - x_target) <= SPREAD_WINDOW)
            if len(near) >= MIN_RESIDUALS:
                residuals = near
        spreads[h] = [quantile(residuals, tau) * SPREAD_MULT for tau in LEVELS]
    if not centre:
        # No usable comparables at all: fall back to the house baseline.
        return {"hours": [{"hour": row["hour"], "q": persistence_quantiles(p1, row["hour"])}
                         for row in case["forecast"]]}
    # Hours without enough points borrow the nearest usable hour's centre
    # (recomputed from that hour's own line) and spread.
    usable_hours = sorted(spreads)
    for h in range(2, 25):
        if h not in centre:
            nearest = min(usable_hours, key=lambda u: (abs(u - h), u))
            pts = points[nearest]
            fit_pts = pts
            near_centre = [p for p in pts if abs(p[0] - x_target) <= CENTRE_WINDOW]
            if len(near_centre) >= MIN_REGRESSION_POINTS:
                fit_pts = near_centre
            line = fit_line(fit_pts)
            if line is None:
                centre[h] = quantile(sorted(r for _, r in pts), 0.5)
            else:
                a, b = line
                rs = sorted(r for _, r in pts)
                centre[h] = min(max(a + b * x_target, rs[0]), rs[-1])
            spreads[h] = spreads[nearest]
    return {"hours": [{"hour": row["hour"],
                       "q": hour_quantiles(p1, centre[row["hour"]], spreads[row["hour"]])}
                      for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
