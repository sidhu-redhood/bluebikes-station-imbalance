import { describe, expect, it } from "vitest";
import { compile, hitEmpty, rng, simulate } from "@/lib/markov";
import { DEFAULT_HMM, filter, hmmHitEmpty, observeDay, startPrior, summarize, type HmmOptions } from "@/lib/hmm";

const st = { capacity: 10, ratesIn: Array(24).fill(6), ratesOut: Array(24).fill(9) };
const perfect: HmmOptions = { dropRate: 0, kernel: [1] };
const point = (cap: number, b: number) => Float64Array.from({ length: cap + 1 }, (_, i) => (i === b ? 1 : 0));

describe("station HMM", () => {
  const c = compile(st, 10);

  it("keeps every belief a probability distribution", () => {
    const day = simulate(st, 5, rng(1), 300, 600);
    const obs = observeDay(day.bikes, 10, 10, 2);
    const { beliefs } = filter(c, 300, startPrior(10, 5), obs);
    expect(beliefs).toHaveLength(obs.length);
    for (const b of beliefs) expect(b.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
  });

  it("with perfect polls, the belief collapses onto the true count", () => {
    const day = simulate(st, 5, rng(3), 300, 600);
    const obs = observeDay(day.bikes, 10, 10, 4, perfect);
    const { beliefs } = filter(c, 300, startPrior(10, 5), obs, perfect);
    beliefs.forEach((b, k) => expect(b[day.bikes[k * 10]]).toBeCloseTo(1, 10));
  });

  it("with every poll dropped, the belief is the plain Markov forecast from the prior", () => {
    const drop: HmmOptions = { dropRate: 1, kernel: [1] };
    const obs = Array<number | null>(7).fill(null);
    const { beliefs } = filter(c, 300, point(10, 4), obs, drop);
    // Point-mass prior at 4: step k must equal the 10-minute chain applied k times.
    const hit = hmmHitEmpty(c, beliefs[0], 300, 3);
    expect(hit).toBeCloseTo(hitEmpty(c, 4, 300, 3), 12);
  });

  it("a low reading raises the run-out forecast above the no-reading forecast", () => {
    const prior = startPrior(10, 5);
    const seen = filter(c, 300, prior, [1], DEFAULT_HMM).beliefs[0];
    const unseen = filter(c, 300, prior, [null], DEFAULT_HMM).beliefs[0];
    expect(hmmHitEmpty(c, seen, 300, 3)).toBeGreaterThan(hmmHitEmpty(c, unseen, 300, 3));
  });

  it("filtering narrows the belief compared with no polls at all", () => {
    const day = simulate(st, 5, rng(7), 300, 600);
    const width = (obs: (number | null)[]) => {
      const s = summarize(filter(c, 300, startPrior(10, 5), obs).beliefs.at(-1)!);
      return s.hi - s.lo;
    };
    const all = observeDay(day.bikes, 10, 10, 8, { dropRate: 0, kernel: DEFAULT_HMM.kernel });
    expect(width(all)).toBeLessThan(width(Array<number | null>(all.length).fill(null)));
  });

  it("summarize returns the mean and an 80% interval that contains it", () => {
    const s = summarize(Float64Array.from([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]));
    expect(s.mean).toBeCloseTo(3, 10);
    expect(s.lo).toBeLessThanOrEqual(s.mean);
    expect(s.hi).toBeGreaterThanOrEqual(s.mean);
  });
});
