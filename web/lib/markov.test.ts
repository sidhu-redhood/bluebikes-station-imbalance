import { describe, expect, it } from "vitest";
import { compile, forecast, generator, hitEmpty, hitFull, rng, simulate, stationary, transition } from "@/lib/markov";

const flat = (a: number, d: number) => ({ capacity: 10, ratesIn: Array(24).fill(a), ratesOut: Array(24).fill(d) });

describe("station Markov model", () => {
  it("builds a stochastic 10-minute transition matrix", () => {
    const n = 9, p = transition(generator(8, 9, 4), n, 10);
    for (let i = 0; i < n; i++) expect(p.slice(i * n, i * n + n).reduce((s, v) => s + v, 0)).toBeCloseTo(1, 12);
  });

  it("two 5-minute steps equal one 10-minute step", () => {
    const n = 13, q = generator(12, 20, 7), half = transition(q, n, 5), whole = transition(q, n, 10);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += half[i * n + k] * half[k * n + j];
        expect(s).toBeCloseTo(whole[i * n + j], 10);
      }
  });

  it("forecasts converge to the stationary distribution", () => {
    const st = flat(8, 4), rows = forecast(compile(st, 10), 3, 0, 400), pi = stationary(st, 0);
    rows[400].forEach((v, b) => expect(v).toBeCloseTo(pi[b], 6));
  });

  it("run-out probability grows with the horizon and beats the poll-only chance", () => {
    const c = compile({ capacity: 6, ratesIn: Array(24).fill(2), ratesOut: Array(24).fill(10) }, 10);
    const h3 = hitEmpty(c, 2, 0, 3), h6 = hitEmpty(c, 2, 0, 6);
    expect(h6).toBeGreaterThanOrEqual(h3);
    expect(h6).toBeGreaterThanOrEqual(forecast(c, 2, 0, 6)[6][0] - 1e-12);
  });

  it("simulated days average out to the forecast mean", () => {
    const st = { capacity: 15, ratesIn: Array(24).fill(12), ratesOut: Array(24).fill(6) }, r = rng(1);
    let sum = 0;
    for (let k = 0; k < 2000; k++) sum += simulate(st, 5, r, 0, 60).bikes[60];
    const mean = forecast(compile(st, 10), 5, 0, 6)[6].reduce((s, v, b) => s + v * b, 0);
    expect(sum / 2000).toBeCloseTo(mean, 0);
  });
});

import parity from "@/lib/__fixtures__/markov-parity.json";

describe("parity with the Python model (pipeline/src/coolboston/bike_markov.py)", () => {
  it.each(parity.cases.map((c, k) => [k, c] as const))("case %i matches", (_k, c) => {
    const comp = compile(c.station, 10);
    const last = forecast(comp, c.bikes, c.startMin, c.steps)[c.steps];
    c.forecastLast.forEach((v, b) => expect(Math.abs(last[b] - v)).toBeLessThan(1e-9));
    expect(Math.abs(hitEmpty(comp, c.bikes, c.startMin, c.steps) - c.hitEmpty)).toBeLessThan(1e-9);
    expect(Math.abs(hitFull(comp, c.bikes, c.startMin, c.steps) - c.hitFull)).toBeLessThan(1e-9);
  });
});
