// Station Markov model, browser mirror of pipeline/src/coolboston/bike_markov.py.
// State b = bikes docked (0..C). Per clock hour: CTMC with returns at rate lamIn, pickups at lamOut
// (bikes/hour). One poll step (10 min) is P = expm(Q dt), by uniformization.
const Markov = (() => {
  const HOURS = 24;

  function generator(cap, lamIn, lamOut, absorbing = -1) {
    const n = cap + 1, q = new Float64Array(n * n), a = lamIn / 60, d = lamOut / 60;
    for (let b = 0; b < n; b++) {
      if (b === absorbing) continue;
      let out = 0;
      if (b < cap) { q[b * n + b + 1] = a; out += a; }
      if (b > 0) { q[b * n + b - 1] = d; out += d; }
      q[b * n + b] = -out;
    }
    return q;
  }

  function matmul(x, y, n) {
    const z = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) {
        const v = x[i * n + k];
        if (v === 0) continue;
        for (let j = 0; j < n; j++) z[i * n + j] += v * y[k * n + j];
      }
    return z;
  }


  function transition(q, n, minutes) {
    let rate = 0;
    for (let b = 0; b < n; b++) rate = Math.max(rate, -q[b * n + b]);
    const p = new Float64Array(n * n);
    if (rate === 0) { for (let b = 0; b < n; b++) p[b * n + b] = 1; return p; }
    const u = new Float64Array(n * n);
    for (let i = 0; i < n * n; i++) u[i] = q[i] / rate;
    for (let b = 0; b < n; b++) u[b * n + b] += 1;
    const lam = rate * minutes, kMax = Math.ceil(lam + 10 * Math.sqrt(lam) + 20);
    let term = new Float64Array(n * n), total = 0, logW = -lam;
    for (let b = 0; b < n; b++) term[b * n + b] = 1;
    for (let k = 0; k <= kMax; k++) {
      if (k > 0) logW += Math.log(lam) - Math.log(k); // log Poisson(k; lam), built up term by term
    const w = Math.exp(logW);
      for (let i = 0; i < n * n; i++) p[i] += w * term[i];
      total += w;
      if (1 - total < 1e-12 && k > lam) break;
      term = matmul(term, u, n);
    }
    for (let b = 0; b < n; b++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += p[b * n + j];
      for (let j = 0; j < n; j++) p[b * n + j] /= s;
    }
    return p;
  }

  /** Precompute the 24 hourly poll matrices, plus absorbing-empty and absorbing-full versions. */
  function compile(st, stepMin) {
    const n = st.capacity + 1, mk = (abs) => Array.from({ length: HOURS }, (_, h) =>
      transition(generator(st.capacity, st.ratesIn[h], st.ratesOut[h], abs), n, stepMin));
    return { n, stepMin, P: mk(-1), Pe: mk(0), Pf: mk(st.capacity) };
  }

  const hourAt = (startMin, step, stepMin) => Math.floor((startMin + step * stepMin) / 60) % HOURS;

  function propagate(mats, n, bikes, startMin, steps, stepMin) {
    const rows = [new Float64Array(n)];
    rows[0][bikes] = 1;
    for (let t = 0; t < steps; t++) {
      const m = mats[hourAt(startMin, t, stepMin)], prev = rows[t], next = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const v = prev[i];
        if (v === 0) continue;
        for (let j = 0; j < n; j++) next[j] += v * m[i * n + j];
      }
      rows.push(next);
    }
    return rows;
  }

  const forecast = (c, bikes, startMin, steps) => propagate(c.P, c.n, bikes, startMin, steps, c.stepMin);
  const hitEmpty = (c, bikes, startMin, steps) => propagate(c.Pe, c.n, bikes, startMin, steps, c.stepMin)[steps][0];
  const hitFull = (c, bikes, startMin, steps) => propagate(c.Pf, c.n, bikes, startMin, steps, c.stepMin)[steps][c.n - 1];

  function rng(seed) { // mulberry32
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Exact event-by-event day (Gillespie). bikes[k] = count at minute startMin + k. */
  function simulate(st, bikes0, random, startMin, minutes) {
    const cap = st.capacity, bikes = new Int16Array(minutes + 1);
    const missedP = new Uint16Array(minutes + 1), missedR = new Uint16Array(minutes + 1);
    let b = bikes0, t = startMin, filled = 0;
    const end = startMin + minutes;
    const recordUntil = (tm) => {
      const stop = Math.min(Math.floor(tm - startMin), minutes);
      while (filled <= stop) bikes[filled++] = b;
    };
    while (t < end) {
      const h = Math.floor(t / 60) % HOURS, segEnd = Math.min(end, (Math.floor(t / 60) + 1) * 60);
      const a = st.ratesIn[h] / 60, d = st.ratesOut[h] / 60, total = a + d;
      for (;;) {
        const tEv = total > 0 ? t - Math.log(1 - random()) / total : Infinity;
        recordUntil(Math.min(tEv, segEnd) - 1e-9);
        if (tEv >= segEnd) { t = segEnd; break; }
        t = tEv;
        const k = Math.min(Math.floor(t - startMin) + 1, minutes);
        if (random() < a / total) { if (b < cap) b++; else missedR[k]++; }
        else { if (b > 0) b--; else missedP[k]++; }
      }
    }
    recordUntil(end);
    return { bikes, missedP, missedR };
  }

  /** Long-run occupancy if one hour's rates held forever (truncated geometric). */
  function stationary(st, h) {
    const n = st.capacity + 1, a = st.ratesIn[h], d = st.ratesOut[h], pi = new Float64Array(n);
    let s = 0;
    for (let b = 0; b < n; b++) { pi[b] = d === 0 ? (b === n - 1 ? 1 : 0) : (a / d) ** b; s += pi[b]; }
    for (let b = 0; b < n; b++) pi[b] /= s;
    return pi;
  }

  return { HOURS, generator, transition, compile, forecast, hitEmpty, hitFull, rng, simulate, stationary };
})();
if (typeof module !== "undefined") module.exports = Markov;
