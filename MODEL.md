# How this forecaster works

## The problem

Grand Theft Auto VI launched on 19 November 2026. Exophase tracks, for each
of the game's 55 PlayStation 5 trophies, the percent of the players it
tracks who have earned it. The house reads that page on a fixed clock and
keeps every read. This program gets one case — every read before a
checkpoint — and must forecast, for every trophy, five quantiles (5%, 25%,
50%, 75%, 95%) of its earn rate at the target, 2026-11-26 00:00 UTC, about
four and a half days after the final checkpoint.

The score is the mean pinball loss on the logit scale over every trophy and
quantile. On that scale a miss from 1% to 2% costs about what a miss from
33% to 50% does, so rare trophies count as much as common ones, and a
forecast whose five quantiles are equal scores exactly its mean absolute
logit error. The entry must also beat the best house entry (a persistence
baseline, a trend baseline, and a house AI model forecast) to take any paid
place.

## What the data actually looks like

Two facts from the case data and the public page drive the whole design:

1. **The launch-night page is a stub, not a signal.** The first reads
   (18 Nov 23:50 UTC, 638 players tracked) show many trophies at exactly
   1.10% — 7 of 638 players — and the whole history of the practice cases
   is byte-identical across reads. Nothing moves until Exophase refreshes.
   Any fit that trusts the whole history equally learns the stub, not the
   game. So the program only looks at the last 48 hours of reads, weighted
   toward the most recent (12-hour half-life), and widens its spread when a
   trophy's history is completely flat, because a refresh can jump it.
2. **By day two the picture is already extreme.** The live page on
   21 Nov ~01:14 UTC shows Chapter 1 at 11.98% and Chapter 2 at 1.71%,
   while Chapter 3, Chapter 4 and the final mission sit at 0.02–0.03% and
   most trophies sit at the 0.01% floor. One trophy, Gone Fishing, reads
   100% — every tracked player has it, at 638 players and still at tens of
   thousands. The distribution is a steep story chain on top of a flat sea
   of un-earned grind trophies, and the player base is growing fast, which
   dilutes every rate even as more people earn the trophy.

## The model

**1. A saturating adoption curve, per trophy, in logit space.**
The rate is modelled as `r(t) = R * sigmoid(a + b * ln(1 + hours since
launch))`, with the ceiling `R` chosen by a grid search that minimises the
fit's error measured in logit(rate) space, so ceilings compare fairly.
Adoption saturates — the first players race through the story, then the
base broadens and growth slows — so a straight line in log time (the house
trend baseline's shape) keeps climbing past where the truth bends over.
The ceiling stops that: a trophy whose recent reads are still accelerating
gets a high ceiling, one that is already bending gets a low one, and a flat
history falls back to persistence (`R` = last rate, slope 0). A larger
ceiling only wins the grid search on a clear (>0.5%) improvement, which
prefers the saturating explanation.

**2. A counts model, blended in at 0.35 weight and clamped.**
Earned counts (`rate x players tracked`) and players tracked are fitted
separately in log space and recombined into a rate. This is the dilution
correction: when the player base grows, the rate can fall while the number
of earners rises, and the rate-only fit reads that as a decline. The
counts-implied rate is clamped to within 1 logit of the adoption fit,
because extrapolating two log-linear fits independently can run past 100%
— that exact failure showed up in testing, and the clamp kills it.

**3. Borrowing strength and the story chain.**
Each median is shrunk 10% toward its group's median (story chain, side
activities, universal opener), because a single trophy's fit is noisy.
Then the chain's order is enforced by isotonic pooling: chapter 1 >=
chapter 2 >= chapter 3 >= chapter 4 >= final mission >= 100% completion
>= platinum. You cannot finish chapter 2 without chapter 1, and you
cannot hold the platinum without everything else, so any fit that violates
that order is wrong on its face.

**4. An honest spread.**
The spread combines the fit's residual, the slope's standard error times
the distance to the target, a horizon random-walk term (`0.5 * sqrt(days)`,
as the house baseline uses), and extra width for flat histories. Near the
0% and 100% edges the spread is widened by 25%, because on the logit
scale a miss there is brutally expensive. The quantiles use the exact
normal z-scores for the case's levels.

## Backtests

No local grade exists for this competition, so the model was tested on
synthetic cases built to match the real data: histories generated from
Weibull adoption curves anchored to the live 21 Nov snapshot (Chapter 1
11.98%, Chapter 2 1.71% at ~49 hours), with the launch-night stub, six-hour
Exophase refresh jumps, and a player base growing from 638 to 250,000.
Four scenarios (base, slow growth, fast growth, growth stalling after day
3) were scored with the spec's pinball loss, against reimplementations of
the two house baselines from their published descriptions.

Mean pinball loss, lower is better:

| scenario | this entry | house trend | house persistence |
|----------|-----------:|------------:|------------------:|
| base     |     0.3550 |      0.4093 |            0.4970 |
| slow     |     0.2590 |      0.2779 |            0.3260 |
| fast     |     0.3658 |      0.3884 |            0.5462 |
| stalled  |     0.2476 |      0.3729 |            0.2022 |

The entry beats the house trend baseline in all four scenarios and the
persistence baseline in three; persistence only wins the stalled scenario,
where nothing changes after the checkpoint and any trend forecast
overshoots by construction. The first version of this program (log-time
linear fit, 48-hour window, unbounded counts model) lost to the house
trend baseline on the same scenarios — the saturating ceiling and the
clamped counts blend are what turned it around.

## Limits

- The practice cases (flat stub histories) cannot validate a trend model:
  they only check that the program reads a case and writes a valid
  forecast, which it does — deterministically, in ~0.2 s per case, with
  outputs validated against the required schema.
- The final answer is a single read at the target; if Exophase's page is
  stale or the launch-week surge breaks pattern, every forecast suffers
  together. The spread is the honest hedge against that.

## Sources

- Exophase, Grand Theft Auto VI trophies (PS5):
  https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/
- make.fun bounty page and brief:
  https://make.fun/bounty/gta6-trophy-rarity-forecast-01
- Competition kit rules (input/output format, scoring, baselines):
  `gta6-trophy-rarity-forecast-01` kit, RULES.txt
