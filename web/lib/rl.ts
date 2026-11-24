/**
 * Operator rebalancing with reinforcement learning. Pure: no React, no storage.
 *
 * Problem: one van can service a few stations every 30 minutes. At a station it either TAKES bikes
 * away (relieves a full station, loads the van) or GIVES bikes (fills an empty one, unloads the van).
 * Riders keep arriving by the same hourly Markov chain as lib/markov.ts, so the environment is the
 * existing Gillespie simulator, run in 30-minute chunks.
 *
 * Learner: tabular Q-learning with parameters shared across stations. A station's state is its fill
 * level (6 buckets) and the direction of its expected flow this hour (5 buckets), so 30 states and
 * 3 actions. The van's limited visits couple the stations: each step the agent ranks stations by the
 * gain of acting over leaving and services the best few that the van can physically do. Each station
 * learns from its own reward: minus the riders it turned away in the next 30 min, minus a move cost.
 *
 * Not modelled: travel time, several vans, rider incentives or nudges, any effect of moves on demand.
 */
import type { BikeStation } from "@/lib/bikes";
import { hitEmpty, hitFull, rng, simulate, type Compiled } from "@/lib/markov";

export type RlStation = Pick<BikeStation, "capacity" | "ratesIn" | "ratesOut" | "initialBikes">;

export interface RlConfig {
  stepMin: number; startMin: number; endMin: number;
  vanCap: number; startLoad: number;
  /** Most bikes moved at one station in one visit. */
  moveMax: number;
  /** Most stations the van services per step. */
  visits: number;
  /** Cost per bike moved, in units of one turned-away rider. */
  moveCost: number;
  alpha: number; gamma: number;
}

export const DEFAULT_RL: RlConfig = {
  stepMin: 30, startMin: 300, endMin: 24 * 60,
  vanCap: 20, startLoad: 10, moveMax: 6, visits: 2, moveCost: 0.05, alpha: 0.1, gamma: 0.95,
};

export const LEVELS = 6, TRENDS = 5, ACTIONS = 3, STATES = LEVELS * TRENDS;
export const LEAVE = 0, TAKE = 1, GIVE = 2;
export const LEVEL_NAMES = ["empty", "≤25%", "≤50%", "≤75%", "<full", "full"];
export const TREND_NAMES = ["draining fast", "draining", "balanced", "filling", "filling fast"];
/** Greedy play ignores a (state, action) tried fewer times than this. */
export const MIN_VISITS = 10;

export function levelOf(b: number, cap: number): number {
  if (b <= 0) return 0;
  if (b >= cap) return 5;
  const f = b / cap;
  return f <= 0.25 ? 1 : f <= 0.5 ? 2 : f <= 0.75 ? 3 : 4;
}

/** Expected net flow this hour as a fraction of capacity, bucketed. */
export function trendOf(st: RlStation, minute: number): number {
  const h = Math.floor(minute / 60) % 24, net = (st.ratesIn[h] - st.ratesOut[h]) / st.capacity;
  return net <= -0.1 ? 0 : net < -0.03 ? 1 : net <= 0.03 ? 2 : net < 0.1 ? 3 : 4;
}

export const stateOf = (st: RlStation, b: number, minute: number) => levelOf(b, st.capacity) * TRENDS + trendOf(st, minute);

export interface Move { station: number; action: typeof TAKE | typeof GIVE; score?: number }
export interface Obs { stations: RlStation[]; bikes: number[]; minute: number; load: number; cfg: RlConfig }
/** A policy returns candidate moves, best first. The van does the first few it can physically do. */
export type Policy = (o: Obs) => Move[];

export const nonePolicy: Policy = () => [];

/** Rule of thumb: relieve stations that are nearly full, refill ones that are nearly empty. */
export const reactivePolicy: Policy = ({ stations, bikes }) => {
  const out: Move[] = [];
  stations.forEach((st, i) => {
    const f = bikes[i] / st.capacity;
    if (f <= 0.15) out.push({ station: i, action: GIVE, score: 0.16 - f });
    else if (f >= 0.85) out.push({ station: i, action: TAKE, score: f - 0.84 });
  });
  return out.sort((a, b) => b.score! - a.score!);
};

/** Uses the Markov forecast: act where the chance of running out / filling in 30 min is high. */
export const forecastPolicy = (compiled: Compiled[], threshold = 0.35): Policy => ({ bikes, minute, cfg }) => {
  const out: Move[] = [];
  compiled.forEach((c, i) => {
    const steps = Math.round(cfg.stepMin / c.stepMin), pE = hitEmpty(c, bikes[i], minute, steps), pF = hitFull(c, bikes[i], minute, steps);
    if (pE >= threshold && pE >= pF) out.push({ station: i, action: GIVE, score: pE });
    else if (pF >= threshold) out.push({ station: i, action: TAKE, score: pF });
  });
  return out.sort((a, b) => b.score! - a.score!);
};

/** Greedy from a Q table: rank stations by Q(act) - Q(leave) and keep positive gains. */
export const qPolicy = (Q: Float64Array, N: Uint32Array, minVisits = MIN_VISITS): Policy => ({ stations, bikes, minute }) => {
  const out: Move[] = [];
  stations.forEach((st, i) => {
    const s = stateOf(st, bikes[i], minute) * ACTIONS, base = Q[s + LEAVE];
    for (const a of [TAKE, GIVE] as const) {
      const gain = Q[s + a] - base;
      if (gain > 1e-9 && N[s + a] >= minVisits) out.push({ station: i, action: a, score: gain });
    }
  });
  return out.sort((a, b) => b.score! - a.score!);
};

/** Do the first `visits` feasible moves. Returns what actually happened, per station. */
export function execute(stations: RlStation[], bikes: number[], load: number, ranked: Move[], cfg: RlConfig) {
  const n = stations.length, moved = new Array<number>(n).fill(0), actions = new Array<number>(n).fill(LEAVE);
  let done = 0;
  for (const m of ranked) {
    if (done >= cfg.visits) break;
    if (actions[m.station] !== LEAVE) continue;
    const b = bikes[m.station], cap = stations[m.station].capacity;
    const q = m.action === TAKE ? Math.min(cfg.moveMax, b, cfg.vanCap - load) : Math.min(cfg.moveMax, load, cap - b);
    if (q <= 0) continue;
    bikes[m.station] += m.action === TAKE ? -q : q;
    load += m.action === TAKE ? q : -q;
    moved[m.station] = q;
    actions[m.station] = m.action;
    done++;
  }
  return { load, moved, actions, visits: done };
}

export interface DayStats { unmet: number; missedP: number; missedR: number; emptyH: number; fullH: number; moved: number; visits: number }
export interface Learner { Q: Float64Array; N: Uint32Array; eps: number; random: () => number }

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
/** Seed per (day, station, chunk), so every policy faces the same rider randomness on a given day. */
const chunkSeed = (day: number, i: number, k: number) => (Math.imul(day + 1, 2654435761) ^ Math.imul(i + 1, 40503) ^ Math.imul(k + 1, 69069)) >>> 0;

function randomMoves(n: number, random: () => number): Move[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  return idx.map((station) => ({ station, action: random() < 0.5 ? TAKE : GIVE }));
}

/** Play one day. With `learn`, also does epsilon-greedy exploration and Q-learning updates. */
export function runDay(stations: RlStation[], policy: Policy, cfg: RlConfig, day: number, learn?: Learner): DayStats {
  const n = stations.length, jitter = rng(day ^ 0x9e3779b9);
  const bikes = stations.map((st) => clamp(st.initialBikes + Math.round((jitter() - 0.5) * 4), 0, st.capacity));
  let load = cfg.startLoad;
  const stats: DayStats = { unmet: 0, missedP: 0, missedR: 0, emptyH: 0, fullH: 0, moved: 0, visits: 0 };
  let emptyMin = 0, fullMin = 0;
  for (let minute = cfg.startMin, k = 0; minute < cfg.endMin; minute += cfg.stepMin, k++) {
    const s0 = learn ? stations.map((st, i) => stateOf(st, bikes[i], minute)) : null;
    const ranked = learn && learn.random() < learn.eps ? randomMoves(n, learn.random) : policy({ stations, bikes, minute, load, cfg });
    const ex = execute(stations, bikes, load, ranked, cfg);
    load = ex.load;
    stats.visits += ex.visits;
    for (let i = 0; i < n; i++) {
      stats.moved += ex.moved[i];
      const tr = simulate(stations[i], bikes[i], rng(chunkSeed(day, i, k)), minute, cfg.stepMin);
      let p = 0, r = 0;
      for (let j = 0; j < tr.missedP.length; j++) { p += tr.missedP[j]; r += tr.missedR[j]; }
      for (let j = 0; j < cfg.stepMin; j++) { if (tr.bikes[j] === 0) emptyMin++; else if (tr.bikes[j] === stations[i].capacity) fullMin++; }
      stats.missedP += p; stats.missedR += r;
      bikes[i] = tr.bikes[cfg.stepMin];
      if (learn && s0) {
        const next = minute + cfg.stepMin, last = next >= cfg.endMin, s1 = stateOf(stations[i], bikes[i], next) * ACTIONS;
        const target = -(p + r) - cfg.moveCost * ex.moved[i] + (last ? 0 : cfg.gamma * Math.max(learn.Q[s1], learn.Q[s1 + 1], learn.Q[s1 + 2]));
        const at = s0[i] * ACTIONS + ex.actions[i];
        learn.Q[at] += cfg.alpha * (target - learn.Q[at]);
        learn.N[at]++;
      }
    }
  }
  stats.unmet = stats.missedP + stats.missedR;
  stats.emptyH = emptyMin / 60; stats.fullH = fullMin / 60;
  return stats;
}

export interface Evaluation { mean: DayStats; unmetByDay: number[]; score: number }

/** Average a policy over `days` days. Same seeds for every policy, so results are paired. */
export function evaluate(stations: RlStation[], policy: Policy, cfg: RlConfig, days: number, baseSeed: number): Evaluation {
  const sum: DayStats = { unmet: 0, missedP: 0, missedR: 0, emptyH: 0, fullH: 0, moved: 0, visits: 0 }, unmetByDay: number[] = [];
  for (let d = 0; d < days; d++) {
    const s = runDay(stations, policy, cfg, baseSeed + d);
    unmetByDay.push(s.unmet);
    for (const key of Object.keys(sum) as (keyof DayStats)[]) sum[key] += s[key] / days;
  }
  return { mean: sum, unmetByDay, score: sum.unmet + cfg.moveCost * sum.moved };
}

export function meanSe(x: number[]): { mean: number; se: number } {
  const n = x.length, mean = x.reduce((s, v) => s + v, 0) / n;
  const v = n > 1 ? x.reduce((s, v2) => s + (v2 - mean) ** 2, 0) / (n - 1) : 0;
  return { mean, se: Math.sqrt(v / n) };
}

/** Per-day difference a - b, as mean and standard error. Negative means a turned away fewer riders. */
export const pairedDiff = (a: number[], b: number[]) => meanSe(a.map((v, i) => v - b[i]));

export interface Trained { Q: Float64Array; N: Uint32Array; curve: number[]; stopped: boolean }

/** Train by playing `episodes` simulated days. Yields to the event loop so a page stays responsive. */
export async function trainQ(
  stations: RlStation[],
  cfg: RlConfig,
  opts: { episodes: number; seed: number; onProgress?: (done: number, curve: number[]) => void; shouldStop?: () => boolean },
): Promise<Trained> {
  const Q = new Float64Array(STATES * ACTIONS), N = new Uint32Array(STATES * ACTIONS), curve: number[] = [], random = rng(opts.seed ^ 0xabcdef);
  const greedy = qPolicy(Q, N, 0); // minVisits 0: untried actions look good (Q starts at 0, rewards are negative), which explores
  let stopped = false;
  for (let e = 0; e < opts.episodes; e++) {
    const eps = 0.5 + (0.05 - 0.5) * (e / Math.max(1, opts.episodes - 1));
    curve.push(runDay(stations, greedy, cfg, opts.seed + e, { Q, N, eps, random }).unmet);
    if (e % 5 === 4 || e === opts.episodes - 1) {
      opts.onProgress?.(e + 1, curve);
      await new Promise((r) => setTimeout(r, 0));
      if (opts.shouldStop?.()) { stopped = true; break; }
    }
  }
  return { Q, N, curve, stopped };
}

/** For display: the move the trained policy prefers in each (level, trend) cell. */
export function policyGrid(Q: Float64Array, N: Uint32Array): { action: number; seen: boolean }[][] {
  return Array.from({ length: LEVELS }, (_, lv) =>
    Array.from({ length: TRENDS }, (_, tr) => {
      const s = (lv * TRENDS + tr) * ACTIONS;
      let best = LEAVE, gain = 1e-9;
      for (const a of [TAKE, GIVE]) if (N[s + a] >= MIN_VISITS && Q[s + a] - Q[s + LEAVE] > gain) { gain = Q[s + a] - Q[s + LEAVE]; best = a; }
      return { action: best, seen: N[s] + N[s + 1] + N[s + 2] >= MIN_VISITS };
    }),
  );
}
