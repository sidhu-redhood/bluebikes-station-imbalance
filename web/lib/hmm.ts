/**
 * Station hidden Markov model for the web app. Pure: no React.
 *
 * Hidden state: b = bikes docked (0..C), which evolves by the same hourly birth-death chain as
 * lib/markov.ts (the transition model). Observation: the 10-minute poll, which is missing with
 * probability `dropRate` and otherwise reads the count with small error (`kernel`). The forward
 * filter turns the polls seen so far into a belief over b. Forecasts start from that belief
 * instead of from a known count.
 *
 * This is a separate model from pipeline/src/coolboston/station_hmm.py (which fits rates and van
 * moves from trips). It only reuses the transition matrices compiled by lib/markov.ts, and does
 * not change them, so the Markov parity rule is unaffected.
 */
import { HOURS, rng, type Compiled } from "@/lib/markov";

export interface HmmOptions {
  /** Probability that a poll returns nothing. */
  dropRate: number;
  /** kernel[k] = weight of reading the true count off by exactly k (either sign). */
  kernel: number[];
}

export const DEFAULT_HMM: HmmOptions = { dropRate: 0.3, kernel: [0.6, 0.15, 0.05] };

/** Belief over the hidden count at every poll step: beliefs[k] is for minute start + k * stepMin. */
export interface Filtered {
  beliefs: Float64Array[];
  /** What the poller saw at each step: a count, or null when the poll was dropped. */
  obs: (number | null)[];
}

export interface BeliefSummary { mean: number; lo: number; hi: number }

type Mat = Compiled["P"][number];

const hourAt = (startMin: number, k: number, stepMin: number) => Math.floor((startMin + k * stepMin) / 60) % HOURS;

/** Normalizer so the reading distribution given a true count b sums to 1 over readings in 0..C. */
function readingNorm(cap: number, kernel: number[]): Float64Array {
  const z = new Float64Array(cap + 1);
  for (let b = 0; b <= cap; b++)
    for (let y = 0; y <= cap; y++) z[b] += kernel[Math.abs(y - b)] ?? 0;
  return z;
}

/** P(reading y | true count b). Zero when the reading is more than kernel.length - 1 away. */
export function emission(y: number, b: number, norm: Float64Array, kernel: number[]): number {
  return (kernel[Math.abs(y - b)] ?? 0) / norm[b];
}

/** Draw one poll: null if dropped, otherwise a reading in 0..cap drawn from the emission model. */
export function observe(b: number, cap: number, random: () => number, opts: HmmOptions = DEFAULT_HMM): number | null {
  if (random() < opts.dropRate) return null;
  const norm = readingNorm(cap, opts.kernel);
  let u = random(), last = b;
  for (let y = 0; y <= cap; y++) {
    const w = emission(y, b, norm, opts.kernel);
    if (w === 0) continue;
    last = y;
    u -= w;
    if (u <= 0) return y;
  }
  return last;
}

/** Observe a station's true counts at every poll step. Uses its own rng, so the sim is untouched. */
export function observeDay(bikes: ArrayLike<number>, cap: number, stepMin: number, seed: number, opts: HmmOptions = DEFAULT_HMM) {
  const random = rng(seed), steps = Math.floor((bikes.length - 1) / stepMin), obs: (number | null)[] = [];
  for (let k = 0; k <= steps; k++) obs.push(observe(bikes[k * stepMin], cap, random, opts));
  return obs;
}

/** Prior over the starting count: peaked at the nominal initialBikes, because that is an assumption. */
export function startPrior(cap: number, initialBikes: number): Float64Array {
  const p = new Float64Array(cap + 1);
  let s = 0;
  for (let b = 0; b <= cap; b++) s += p[b] = 0.5 ** Math.abs(b - initialBikes);
  for (let b = 0; b <= cap; b++) p[b] /= s;
  return p;
}

function step(belief: Float64Array, m: Mat, n: number): Float64Array {
  const next = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = belief[i];
    if (v === 0) continue;
    for (let j = 0; j < n; j++) next[j] += v * m[i * n + j];
  }
  return next;
}

function update(prior: Float64Array, y: number | null, norm: Float64Array, kernel: number[]): Float64Array {
  if (y === null) return prior;
  const post = new Float64Array(prior.length);
  let s = 0;
  for (let b = 0; b < prior.length; b++) s += post[b] = prior[b] * emission(y, b, norm, kernel);
  if (s <= 0) return prior; // impossible reading under the model: keep the prediction
  for (let b = 0; b < post.length; b++) post[b] /= s;
  return post;
}

/** Forward filter: beliefs[k] = P(count at step k | polls 0..k). */
export function filter(
  c: Compiled,
  startMin: number,
  prior: Float64Array,
  obs: (number | null)[],
  opts: HmmOptions = DEFAULT_HMM,
): Filtered {
  const n = c.n, norm = readingNorm(n - 1, opts.kernel);
  const beliefs: Float64Array[] = [update(prior, obs[0] ?? null, norm, opts.kernel)];
  for (let k = 1; k < obs.length; k++) {
    const predicted = step(beliefs[k - 1], c.P[hourAt(startMin, k - 1, c.stepMin)], n);
    beliefs.push(update(predicted, obs[k], norm, opts.kernel));
  }
  return { beliefs, obs };
}

/** Probability of having hit the absorbing state (index `target`) within `steps` polls, from a belief. */
export function hitFromBelief(mats: Mat[], n: number, belief: Float64Array, startMin: number, steps: number, stepMin: number, target: number): number {
  let b = belief;
  for (let t = 0; t < steps; t++) b = step(b, mats[hourAt(startMin, t, stepMin)], n);
  return b[target];
}

export const hmmHitEmpty = (c: Compiled, belief: Float64Array, startMin: number, steps: number) =>
  hitFromBelief(c.Pe, c.n, belief, startMin, steps, c.stepMin, 0);
export const hmmHitFull = (c: Compiled, belief: Float64Array, startMin: number, steps: number) =>
  hitFromBelief(c.Pf, c.n, belief, startMin, steps, c.stepMin, c.n - 1);

/** Mean count and the central 80% interval of a belief. */
export function summarize(belief: Float64Array): BeliefSummary {
  let mean = 0, cum = 0, lo = -1, hi = belief.length - 1;
  for (let b = 0; b < belief.length; b++) mean += b * belief[b];
  for (let b = 0; b < belief.length; b++) {
    cum += belief[b];
    if (lo < 0 && cum >= 0.1) lo = b;
    if (cum >= 0.9) { hi = b; break; }
  }
  return { mean, lo: Math.max(lo, 0), hi };
}
