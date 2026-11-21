"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will have earned it at the case's target, as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95). Reads every <id>.in in the input
directory and writes <id>.out to the output directory.

Method, per trophy, on the logit scale:

  1. Regime break. Exophase refreshes its counts in steps, and the launch-night
     page described a different, hardcore population (638 tracked players) from
     the one that follows: at the first refresh the tracked base jumped ~6x and
     nearly every rate fell. The largest players-tracked jump in the history
     marks that break; reads before it are dropped, so the trend is fitted on
     the representative population only.
  2. Trend. logit(rate) is fitted by least squares against ln(1 + hours since
     launch / 24), weighted toward recent reads (24-hour half-life), on the
     case's change points (consecutive identical reads carry one observation).
     The slope is shrunk toward zero by n / (n + 2): trophies with a clear
     trend keep it, trophies with one or two noisy points are pulled to flat.
  3. Damped extrapolation. Growth in a launch week decelerates (the player
     base saturates, the early cohort finishes the easy content), so the fitted
     slope is damped over the horizon: slope * (1 - exp(-h/2)) / (h/2) for a
     horizon of h days (about 0.4 over the final's 4.5 days, near 1 over the
     practice's 90 minutes). The median's move from the last rate is capped
     at 2.5 logits.
  4. Spread. The quantile spread on the logit scale combines the trophy's fit
     residual, a random-walk term growing with the horizon, and an allowance
     for Exophase's discrete refreshes (the answering read is a stale snapshot
     of the target, not the target itself).

Standard library only, no network, deterministic: the same case always writes
the same bytes. Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread terms, in logit units.
RESID_FLOOR = 0.25      # floor on a trophy's fit-residual spread
RESID_CAP = 1.20        # cap on a trophy's fit-residual spread
HORIZON_SIGMA = 0.30    # random-walk spread per sqrt(day) of horizon
REFRESH_SIGMA = 0.30    # allowance for Exophase's discrete refresh jumps
SPREAD_CAP = 2.00
MAX_LOGIT_MOVE = 2.5    # cap on the median's move from the last rate, logits

SHRINK_K = 2.0          # slope shrinks toward zero as n / (n + SHRINK_K)
SLOPE_MIN = -3.0
SLOPE_MAX = 3.0
RECENCY_HALF_LIFE_H = 24.0
BREAK_JUMP = 1.8         # players-tracked jump that marks a regime break


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit_pct(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def weighted_least_squares(xs, ys, ws):
    """Weighted slope and intercept of y ~ a + b*x; (None, None) if degenerate."""
    sw = sum(ws)
    if sw <= 0:
        return None, None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None, None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return b, my - b * mx


def change_points(case):
    """Compress history to reads where players or any rate changed.

    Exophase refreshes its counts in steps, so long flat stretches carry one
    observation, not dozens.
    """
    points = []
    prev = None
    for read in case["history"]:
        key = (read.get("players_tracked"), tuple(read["rates"]))
        if key != prev:
            points.append(read)
            prev = key
    return points


def after_break(points):
    """Drop reads before the largest players-tracked jump, if it is a break."""
    best_i, best_ratio = None, 1.0
    for i in range(1, len(points)):
        prev_p = points[i - 1].get("players_tracked") or 0
        cur_p = points[i].get("players_tracked") or 0
        if prev_p > 0 and cur_p > 0:
            ratio = cur_p / prev_p
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i
    if best_i is not None and best_ratio >= BREAK_JUMP and len(points) - best_i >= 2:
        return points[best_i:]
    return points


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    t_target_h = max(0.0, (target - release).total_seconds() / 3600.0)
    x_target = math.log1p(t_target_h / 24.0)

    points = after_break(change_points(case))
    t_last = max(0.0, (parse_time(points[-1]["at"]) - release).total_seconds() / 3600.0)
    x_last = math.log1p(t_last / 24.0)

    # Damping: the fitted trend continues at a fraction of its strength that
    # falls as the horizon grows (launch-week growth decelerates).
    if horizon_days > 0:
        damp = (1.0 - math.exp(-horizon_days / 2.0)) / (horizon_days / 2.0)
    else:
        damp = 1.0
    spread_floor = max(0.50, 0.35 * math.sqrt(horizon_days))
    spread_hor = HORIZON_SIGMA * math.sqrt(horizon_days)

    last = case["history"][-1]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys, ws = [], [], []
        for r in points:
            rate = r["rates"][i]
            if rate is None:
                continue
            t = max(0.0, (parse_time(r["at"]) - release).total_seconds() / 3600.0)
            xs.append(math.log1p(t / 24.0))
            ys.append(logit(rate))
            ws.append(0.5 ** ((t_last - t) / RECENCY_HALF_LIFE_H))
        n = len(xs)
        b, a = weighted_least_squares(xs, ys, ws)
        if b is None:
            slope = 0.0
            resid = 0.45
        else:
            slope = min(SLOPE_MAX, max(SLOPE_MIN, b)) * n / (n + SHRINK_K)
            res = [y - (a + b * x) for x, y in zip(xs, ys)]
            if len(res) >= 2:
                mean_res = sum(w * r for w, r in zip(ws, res)) / sum(ws)
                var = sum(w * (r - mean_res) ** 2 for w, r in zip(ws, res)) / sum(ws)
                resid = min(RESID_CAP, max(RESID_FLOOR, math.sqrt(var)))
            else:
                resid = 0.45

        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = 0.01
        if last_rate >= 99.99:
            # Everybody tracked already holds it; it stays earned by all.
            med = 100.0
            spread = 0.10
        else:
            move = slope * damp * (x_target - x_last)
            move = min(MAX_LOGIT_MOVE, max(-MAX_LOGIT_MOVE, move))
            med = clamp(inv_logit_pct(logit(last_rate) + move))
            spread = math.sqrt(resid ** 2 + spread_hor ** 2 + REFRESH_SIGMA ** 2)
            spread = min(SPREAD_CAP, max(spread_floor, spread))

        lg = logit(med)
        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit_pct(lg + z * spread))
            q = min(100, max(0.01, float(q)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over the 65,536 byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
