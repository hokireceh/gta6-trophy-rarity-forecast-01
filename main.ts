// GTA VI trophy rarity forecaster.
//
// Reads every case file (*.in) in the input directory and writes one forecast
// file (*.out) per case to the output directory. Each case is a history of
// Exophase earn-rate reads for the game's PlayStation 5 trophies; the program
// forecasts, for every trophy, five quantiles (5/25/50/75/95%) of its earn
// rate at the case's target time.
//
// Method, in brief (see MODEL.md for the reasoning):
//   1. Work in logit space, where pinball loss is scored.
//   2. Per trophy, fit a saturating adoption curve: the rate follows
//      r(t) = R * sigmoid(a + b * ln(1 + hours since launch)), with the
//      ceiling R chosen by a grid search that minimises the fit's error in
//      logit space. Adoption saturates, so a straight line in log time would
//      overshoot; the ceiling keeps the extrapolation honest. A flat history
//      falls back to persistence (R = last rate, slope 0).
//   3. A counts model (earned counts and players tracked fitted separately,
//      so a growing player base dilutes rates honestly) is blended in at
//      0.35 weight, clamped to within 1 logit of the adoption fit, so it can
//      correct the trend but never run away past 100%.
//   4. Medians are shrunk 10% toward their group's median, and the story
//      chain's order (chapter 1 >= chapter 2 >= ... >= final mission >=
//      100% completion >= platinum) is enforced by isotonic pooling.
//   5. Spread comes from the fit's residual, the slope's uncertainty, a
//      horizon random-walk term, and extra width when a trophy's history is
//      flat (an Exophase refresh can jump it).
// No network, no wall clock, no randomness: two runs on the same case write
// the same forecast.

import * as fs from "node:fs";
import * as path from "node:path";

type Trophy = {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
};

type HistoryRead = {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
};

type CaseFile = {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: HistoryRead[];
};

const LAUNCH_MS = Date.parse("2026-11-19T00:00:00Z");
const HOUR_MS = 3600_000;
const RATE_LO = 0.01;
const RATE_HI = 99.99;

// Story chain, hardest last: chapter 1..4, final mission, 100% completion,
// platinum. A later trophy in this chain cannot be earned more often than
// an earlier one, so the medians must not increase along it.
const CHAIN_IDS = [3, 4, 5, 6, 7, 14, 1];

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const e = Math.exp(-x);
    return 1 / (1 + e);
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

function logitOfRate(ratePct: number): number {
  const p = clamp(ratePct, RATE_LO, RATE_HI) / 100;
  return Math.log(p / (1 - p));
}

function rateOfLogit(l: number): number {
  return clamp(100 * sigmoid(l), RATE_LO, RATE_HI);
}

// Inverse standard normal CDF (Acklam's rational approximation, good to
// ~1e-9). Turns a quantile level into its z-score.
function normInv(p: number): number {
  const q = clamp(p, 0.000001, 0.999999);
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239e0];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let x: number;
  if (q < plow) {
    const t = Math.sqrt(-2 * Math.log(q));
    x = (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) /
      ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1);
  } else if (q <= phigh) {
    const t = q - 0.5;
    const u = t * t;
    x = (((((a[0] * u + a[1]) * u + a[2]) * u + a[3]) * u + a[4]) * u + a[5]) * t /
      (((((b[0] * u + b[1]) * u + b[2]) * u + b[3]) * u + b[4]) * u + 1);
  } else {
    const t = Math.sqrt(-2 * Math.log(1 - q));
    x = -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) /
      ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1);
  }
  return x;
}

type Point = { x: number; y: number; w: number };

type LinFit = {
  intercept: number;
  slope: number;
  residStd: number;
  slopeSe: number;
  xMean: number;
  n: number;
};

// Recency-weighted least squares of y on x. Falls back to a flat fit
// (slope 0) when there are too few points or no spread in x.
function weightedFit(points: Point[]): LinFit {
  const flat = (intercept: number, xMean: number): LinFit => ({
    intercept, slope: 0, residStd: 0.35, slopeSe: 0, xMean, n: points.length,
  });
  if (points.length === 0) return flat(0, 0);
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  if (sw <= 0) return flat(0, 0);
  const xMean = sx / sw;
  const yMean = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    const dx = p.x - xMean;
    sxx += p.w * dx * dx;
    sxy += p.w * dx * (p.y - yMean);
  }
  if (points.length < 3 || sxx < 1e-9) return flat(yMean, xMean);
  const slope = sxy / sxx;
  const intercept = yMean - slope * xMean;
  let rss = 0;
  for (const p of points) {
    const r = p.y - (intercept + slope * p.x);
    rss += p.w * r * r;
  }
  const dof = Math.max(points.length - 2, 1);
  const s2 = rss / dof;
  const residStd = Math.sqrt(Math.max(s2, 0.0004));
  const slopeSe = Math.sqrt(Math.max(s2 / sxx, 0));
  return { intercept, slope, residStd, slopeSe, xMean, n: points.length };
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

// Pool-adjacent-violators: make `values` non-increasing, by averaging
// adjacent blocks that violate the order.
function isotonicDecreasing(values: number[]): number[] {
  const blocks: { sum: number; count: number }[] = [];
  for (const v of values) {
    blocks.push({ sum: v, count: 1 });
    while (blocks.length >= 2) {
      const n = blocks.length;
      const prev = blocks[n - 2];
      const last = blocks[n - 1];
      if (prev.sum / prev.count >= last.sum / last.count) break;
      blocks.splice(n - 2, 2, {
        sum: prev.sum + last.sum,
        count: prev.count + last.count,
      });
    }
  }
  const result: number[] = [];
  for (const b of blocks) {
    for (let i = 0; i < b.count; i++) result.push(b.sum / b.count);
  }
  return result;
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

type AdoptionFit = {
  logitMedian: number;
  residStd: number;
  slopeSe: number;
  xMean: number;
  distinctY: number;
  lastLogit: number;
};

// Saturating adoption fit for one trophy's rate series.
// points: (x = ln(1 + hours since launch), ratePct, weight), oldest first.
// The ceiling R is grid-searched; for each R the model is
// r(t) = R * sigmoid(a + b*x), fitted by least squares on logit(r/R), and
// scored by its weighted error in logit(rate) space so ceilings compare
// fairly. A larger ceiling replaces the best only on a clear improvement,
// which prefers the saturating (smaller-R) explanation.
function adoptionFit(
  points: { x: number; rate: number; w: number }[],
  xTarget: number,
  lastLogit: number,
): AdoptionFit {
  const base: AdoptionFit = {
    logitMedian: lastLogit, residStd: 0.35, slopeSe: 0, xMean: 0,
    distinctY: 0, lastLogit,
  };
  if (points.length === 0) return base;
  const distinctY = new Set(points.map((p) => Math.round(p.rate * 100))).size;
  base.distinctY = distinctY;
  const lastRate = points[points.length - 1].rate;
  if (points.length < 3 || distinctY < 2) {
    base.xMean = points[points.length - 1].x;
    return base;
  }
  const ladder = [1.0, 1.25, 1.6, 2, 2.5, 3.2, 4, 5, 6.5, 8, 10, 13, 16,
    20, 26, 33, 42, 55, 70, 90];
  let best: {
    sse: number; intercept: number; slope: number; slopeSe: number;
    xMean: number; residStd: number; ceiling: number;
  } | null = null;
  for (const mult of ladder) {
    const ceiling = Math.min(lastRate * mult, RATE_HI);
    if (ceiling <= lastRate * 0.999 && mult > 1.0) continue;
    const ys: Point[] = [];
    let usable = true;
    for (const p of points) {
      const frac = clamp(p.rate, RATE_LO, ceiling * 0.999) / ceiling;
      if (!(frac > 0 && frac < 1)) { usable = false; break; }
      ys.push({ x: p.x, y: Math.log(frac / (1 - frac)), w: p.w });
    }
    if (!usable) continue;
    const fit = weightedFit(ys);
    // Score in logit(rate) space so every ceiling is judged alike.
    let sse = 0;
    let sw = 0;
    for (let i = 0; i < points.length; i++) {
      const pred = ceiling * sigmoid(fit.intercept + fit.slope * points[i].x);
      const e = logitOfRate(points[i].rate) - logitOfRate(pred);
      sse += points[i].w * e * e;
      sw += points[i].w;
    }
    const meanSse = sw > 0 ? sse / sw : Infinity;
    if (best === null || meanSse < best.sse * 0.995) {
      best = {
        sse: meanSse, intercept: fit.intercept, slope: fit.slope,
        slopeSe: fit.slopeSe, xMean: fit.xMean,
        residStd: Math.sqrt(Math.max(meanSse, 0.01)), ceiling,
      };
    }
  }
  if (best === null) {
    base.xMean = points[points.length - 1].x;
    return base;
  }
  const logitMedian = logitOfRate(best.ceiling * sigmoid(best.intercept + best.slope * xTarget));
  return {
    logitMedian, residStd: best.residStd, slopeSe: best.slopeSe,
    xMean: best.xMean, distinctY, lastLogit,
  };
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const trophies = c.trophies;
  const nT = trophies.length;
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(c.horizon_minutes, 0) / 1440;
  const xTarget = Math.log(1 + Math.max((targetMs - LAUNCH_MS) / HOUR_MS, 0));

  // Recent reads only (48-hour lookback), heaviest on the most recent, so
  // a stale launch-night page cannot drag the fit.
  const windowMs = 48 * HOUR_MS;
  const recent = c.history.filter((r) => {
    const t = Date.parse(r.at);
    return Number.isFinite(t) && t <= checkpointMs && t >= checkpointMs - windowMs;
  });
  const useReads = recent.length >= 2
    ? recent
    : c.history.filter((r) => Date.parse(r.at) <= checkpointMs);
  const lastRead = useReads.length > 0 ? useReads[useReads.length - 1] : null;
  const lastMs = lastRead ? Date.parse(lastRead.at) : checkpointMs;

  type PerTrophy = { logitMedian: number; spread: number; group: number };
  const per: PerTrophy[] = [];

  const groupOf = (t: Trophy): number => {
    if (CHAIN_IDS.includes(t.id)) return 0; // story chain
    if (t.id === 2) return 2; // universal opener
    return 1; // side activities, collections, grinds
  };

  for (let i = 0; i < nT; i++) {
    const t = trophies[i];
    const ratePts: { x: number; rate: number; w: number }[] = [];
    const countPts: Point[] = [];
    const nPts: Point[] = [];
    for (const r of useReads) {
      const ms = Date.parse(r.at);
      if (!Number.isFinite(ms)) continue;
      const hours = Math.max((ms - LAUNCH_MS) / HOUR_MS, 0);
      const x = Math.log(1 + hours);
      const w = Math.pow(0.5, Math.max(lastMs - ms, 0) / (12 * HOUR_MS));
      const rate = r.rates[i];
      if (typeof rate === "number" && Number.isFinite(rate)) {
        ratePts.push({ x, rate, w });
        if (typeof r.players_tracked === "number" && r.players_tracked > 0) {
          const earned = Math.max((rate / 100) * r.players_tracked, 0.01);
          countPts.push({ x, y: Math.log(earned), w });
        }
      }
      if (typeof r.players_tracked === "number" && r.players_tracked > 0) {
        nPts.push({ x, y: Math.log(r.players_tracked), w });
      }
    }
    const nPtsDedup: Point[] = [];
    for (const p of nPts) {
      const prev = nPtsDedup[nPtsDedup.length - 1];
      if (!prev || Math.abs(prev.y - p.y) > 1e-9 || Math.abs(prev.x - p.x) > 1e-9) {
        nPtsDedup.push(p);
      }
    }

    const lastRate = lastRead && typeof lastRead.rates[i] === "number"
      ? (lastRead.rates[i] as number)
      : null;
    const lastLogit = lastRate !== null ? logitOfRate(lastRate) : logitOfRate(0.5);

    // --- adoption model ----------------------------------------------------
    const fit = adoptionFit(ratePts, xTarget, lastLogit);
    let logitMedian = fit.logitMedian;

    // --- counts model, bounded ---------------------------------------------
    // Earned counts and players tracked extrapolated separately; the implied
    // rate is clamped to within 1 logit of the adoption fit so a runaway
    // extrapolation can never push a rate past 100% on its own.
    const countFit = weightedFit(countPts);
    const nFit = weightedFit(nPtsDedup);
    if (countFit.n >= 3 && nFit.n >= 2) {
      const logE = countFit.intercept + clamp(countFit.slope, -1.5, 4) * xTarget;
      const logN = nFit.intercept + clamp(nFit.slope, -1.5, 4) * xTarget;
      const pCount = clamp(Math.exp(logE - logN), 0.0001, 0.9999);
      const logitCounts = clamp(
        Math.log(pCount / (1 - pCount)),
        logitMedian - 1.0,
        logitMedian + 1.0,
      );
      logitMedian = 0.65 * logitMedian + 0.35 * logitCounts;
    }

    // --- spread --------------------------------------------------------------
    const flat = fit.distinctY < 2;
    const horizonSpread = 0.5 * Math.sqrt(horizonDays);
    let spread = Math.sqrt(
      fit.residStd * fit.residStd +
      Math.pow(fit.slopeSe * Math.max(xTarget - fit.xMean, 0), 2) +
      horizonSpread * horizonSpread +
      (flat ? 0.9 * 0.9 : 0),
    );
    spread = clamp(spread, 0.35, 2.6);
    const medRate = rateOfLogit(logitMedian);
    if (medRate > 90 || medRate < 0.05) spread = Math.min(spread * 1.25, 2.6);

    per.push({ logitMedian, spread, group: groupOf(t) });
  }

  // Shrink each median 10% toward its group's median: noisy single-trophy
  // fits borrow strength from their group.
  const groupMedians = new Map<number, number>();
  for (let g = 0; g <= 2; g++) {
    const vals = per.filter((p) => p.group === g).map((p) => p.logitMedian);
    if (vals.length > 0) groupMedians.set(g, median(vals));
  }
  for (const p of per) {
    const gMed = groupMedians.get(p.group);
    if (gMed !== undefined) {
      p.logitMedian = clamp(
        0.9 * p.logitMedian + 0.1 * gMed,
        logitOfRate(RATE_LO), logitOfRate(RATE_HI),
      );
    }
  }

  // Story chain: medians must not increase along the chain.
  const chainIdx = CHAIN_IDS
    .map((id) => trophies.findIndex((t) => t.id === id))
    .filter((ix) => ix >= 0);
  if (chainIdx.length >= 2) {
    const vals = chainIdx.map((ix) => per[ix].logitMedian);
    const fixed = isotonicDecreasing(vals);
    chainIdx.forEach((ix, k) => { per[ix].logitMedian = fixed[k]; });
  }

  const levels = c.levels && c.levels.length === 5 ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const zs = levels.map((lv) => normInv(lv));
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < nT; i++) {
    const p = per[i];
    const qs: number[] = [];
    let prev = RATE_LO;
    for (let k = 0; k < levels.length; k++) {
      let v = round4(rateOfLogit(p.logitMedian + zs[k] * p.spread));
      if (v < prev) v = prev;
      v = clamp(v, RATE_LO, RATE_HI);
      qs.push(v);
      prev = v;
    }
    rows.push({ id: trophies[i].id, q: qs });
  }
  rows.sort((a, b) => a.id - b.id);
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });
  const files = fs.readdirSync(inputDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const file of files) {
    const raw = fs.readFileSync(path.join(inputDir, file), "utf8");
    const c = JSON.parse(raw) as CaseFile;
    const rows = forecastCase(c);
    const out = { trophies: rows };
    const outPath = path.join(outputDir, file.replace(/\.in$/, ".out"));
    fs.writeFileSync(outPath, JSON.stringify(out) + "\n", "utf8");
  }
}

main();
