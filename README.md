# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chenchendashu/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/chenchendashu/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A per-trophy quantile forecast of how many GTA VI players earn each trophy in week one. Given a case file with the Exophase trophy rates observed up to a checkpoint (players tracked and the percentage who earned each of the 55 trophies, sampled over time), the program forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of each trophy's earn rate at the target time.

## How it works

Every case carries the trophy-rate history as a sequence of reads. The program:

1. Detects the launch-night regime break: Exophase's tracked-player count jumps in steps as the player base grows, so the rate series is diluted. Reads before the largest tracked-player jump (≥1.8×) are dropped when at least two reads remain.
2. Fits, per trophy, a weighted least-squares line of `logit(rate)` against `ln(1 + hours-since-launch / 24)`, on change points only (consecutive identical reads deduplicated), with recency weights (24 h half-life). Trophy rates move roughly linearly in log-time as the player base accumulates.
3. Shrinks each slope toward zero by `n / (n + 2)` (fewer change points → more shrinkage) and clamps it to ±3 logits per log-day, then extrapolates with damping — the slope is multiplied by `(1 − exp(−h/2)) / (h/2)` for a horizon of `h` days (≈0.4 over the final's 4.5 days), because earn-rate growth mean-reverts as the population saturates. The median move is capped at ±2.5 logits.
4. Sets the spread from the fit residuals plus horizon uncertainty (`0.30·√days`) and a 0.30-logit floor, floored at `max(0.50, 0.35·√days)` and capped at 2.0 logits, so long horizons get honestly wide intervals. A trophy already at ≥99.99% is special-cased (median 100, spread 0.10).
5. Falls back to the last observed rate (persistence) whenever a trophy has too little usable history, so every case always produces a valid forecast.

The program is deterministic (byte-identical output across runs), uses only the Python standard library, and runs in ~0.06 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` next to the requested output directory, in the contest's `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}` format.

---

<sub>Entered on [make.fun](https://make.fun/?ref=chenchendashu&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
