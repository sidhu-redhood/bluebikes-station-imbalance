import { describe, expect, it } from "vitest";
import { compile } from "@/lib/markov";
import {
  DEFAULT_RL, GIVE, LEAVE, TAKE, evaluate, execute, forecastPolicy, levelOf, nonePolicy, pairedDiff, policyGrid,
  qPolicy, reactivePolicy, runDay, trainQ, trendOf, type RlConfig, type RlStation,
} from "@/lib/rl";

// A has a steady drain, B a steady fill: moving bikes from B to A is clearly the right thing to do.
const A: RlStation = { capacity: 10, initialBikes: 5, ratesIn: Array(24).fill(0), ratesOut: Array(24).fill(8) };
const B: RlStation = { capacity: 10, initialBikes: 5, ratesIn: Array(24).fill(8), ratesOut: Array(24).fill(0) };
const cfg: RlConfig = { ...DEFAULT_RL, visits: 2, moveMax: 5, startLoad: 5 };

describe("rebalancing environment", () => {
  it("buckets the fill level at the edges", () => {
    expect([0, 1, 5, 6, 8, 9, 10].map((b) => levelOf(b, 10))).toEqual([0, 1, 2, 3, 4, 4, 5]);
  });

  it("buckets the flow direction", () => {
    expect([trendOf(A, 600), trendOf(B, 600)]).toEqual([0, 4]);
  });

  it("conserves bikes and respects the van's size when it moves them", () => {
    const bikes = [0, 10];
    const out = execute([A, B], bikes, 18, [{ station: 1, action: TAKE }, { station: 0, action: GIVE }], { ...cfg, vanCap: 20 });
    expect(bikes[0] + bikes[1] + out.load).toBe(0 + 10 + 18);
    expect(out.load).toBeLessThanOrEqual(20);
    expect(out.moved).toEqual([5, 2]); // van had room for 2 from B, then dropped 5 at A
    expect(out.actions).toEqual([GIVE, TAKE]);
  });

  it("skips moves that cannot be done and only counts the ones that happen", () => {
    const bikes = [3, 4];
    const out = execute([A, B], bikes, 0, [{ station: 0, action: GIVE }], cfg); // empty van cannot give
    expect(out.visits).toBe(0);
    expect(out.actions).toEqual([LEAVE, LEAVE]);
    expect(bikes).toEqual([3, 4]);
  });

  it("serves every policy the same riders on a given day", () => {
    const a = runDay([A, B], nonePolicy, cfg, 11), b = runDay([A, B], nonePolicy, cfg, 11);
    expect(a).toEqual(b);
    expect(a.moved).toBe(0);
  });

  it("shows the problem is real: with no van, riders are turned away", () => {
    expect(evaluate([A, B], nonePolicy, cfg, 5, 100).mean.unmet).toBeGreaterThan(20);
  });
});

describe("policies", () => {
  it("the reactive rule beats doing nothing on the drain/fill example", () => {
    const none = evaluate([A, B], nonePolicy, cfg, 20, 500), rule = evaluate([A, B], reactivePolicy, cfg, 20, 500);
    expect(rule.mean.unmet).toBeLessThan(none.mean.unmet);
    expect(pairedDiff(rule.unmetByDay, none.unmetByDay).mean).toBeLessThan(0);
  });

  it("the forecast rule acts on the same example", () => {
    const rule = evaluate([A, B], forecastPolicy([A, B].map((s) => compile(s, 10))), cfg, 10, 500);
    expect(rule.mean.visits).toBeGreaterThan(0);
  });
});

describe("Q-learning", () => {
  it("learns to beat doing nothing, and learns to relieve the full station", async () => {
    const t = await trainQ([A, B], cfg, { episodes: 150, seed: 1000 });
    expect(t.curve).toHaveLength(150);
    const none = evaluate([A, B], nonePolicy, cfg, 20, 90000);
    const learned = evaluate([A, B], qPolicy(t.Q, t.N), cfg, 20, 90000);
    expect(learned.mean.unmet).toBeLessThan(none.mean.unmet * 0.9);
    const grid = policyGrid(t.Q, t.N);
    // Draining station (trend 0) at the empty level: if the policy has an opinion there, it must be to give.
    const cell = grid[0][0];
    if (cell.seen && cell.action !== LEAVE) expect(cell.action).toBe(GIVE);
  }, 60_000);

  it("stops when asked", async () => {
    const t = await trainQ([A, B], cfg, { episodes: 100, seed: 1, shouldStop: () => true });
    expect(t.stopped).toBe(true);
    expect(t.curve.length).toBeLessThan(100);
  });
});
