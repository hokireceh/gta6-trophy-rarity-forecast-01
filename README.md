# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hokireceh/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hokireceh/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, five quantiles
(5%, 25%, 50%, 75%, 95%) of its Exophase earn rate at 2026-11-26 00:00 UTC.
`main.ts` reads each case in the input directory (a history of Exophase
trophy-page reads) and writes one forecast per case to the output directory.

How it forecasts: per trophy, a saturating adoption curve in logit space —
the rate follows `R * sigmoid(a + b * ln(1 + hours since launch))`, with the
ceiling `R` chosen by grid search — blended with a counts model that fits
earned counts and players tracked separately, so a growing player base
dilutes rates honestly. Medians borrow strength from their group and obey
the story chain's order (chapter 1 >= chapter 2 >= ... >= final mission >=
100% completion >= platinum). Spreads come from fit residuals, slope
uncertainty, and a horizon random-walk term. See MODEL.md for the full
reasoning, the backtests against the house baselines, and the sources.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output node main.ts
```

Requires Node.js 22 or later. No dependencies, no network, no wall clock:
two runs on the same case write the same forecast.

---

<sub>Entered on [make.fun](https://make.fun/?ref=jjqnkn84&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
