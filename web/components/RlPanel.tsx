"use client";

import { useEffect, useRef, useState } from "react";
import type { Compiled } from "@/lib/markov";
import {
  DEFAULT_RL, GIVE, LEVELS, LEVEL_NAMES, TAKE, TREND_NAMES, TRENDS, evaluate, forecastPolicy, nonePolicy, pairedDiff, policyGrid,
  qPolicy, reactivePolicy, trainQ, type Evaluation, type RlStation, type Trained,
} from "@/lib/rl";

const EPISODES = 600, EVAL_DAYS = 40, TRAIN_SEED = 1000, EVAL_SEED = 900_000, SMOOTH = 20;
const cfg = DEFAULT_RL;

interface Result { rows: { key: string; label: string; ev: Evaluation }[]; trained: Trained }

const smooth = (x: number[]) => x.map((_, i) => { const w = x.slice(Math.max(0, i - SMOOTH + 1), i + 1); return w.reduce((s, v) => s + v, 0) / w.length; });
const f1 = (x: number) => x.toFixed(1);

/** Training curve (turned-away riders per simulated day) with the other policies as flat reference lines. */
function Curve({ curve, refs }: { curve: number[]; refs: { label: string; value: number; color: string }[] }) {
  const W = 420, H = 140, P = 28, ys = smooth(curve), top = Math.max(...ys, ...refs.map((r) => r.value), 1) * 1.05;
  const x = (i: number) => P + (i / Math.max(1, EPISODES - 1)) * (W - P - 6), y = (v: number) => H - 18 - (v / top) * (H - 26);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Training curve: turned-away riders per day by training episode">
      <line x1={P} y1={H - 18} x2={W - 6} y2={H - 18} stroke="currentColor" opacity={0.25} />
      <line x1={P} y1={8} x2={P} y2={H - 18} stroke="currentColor" opacity={0.25} />
      {refs.map((r) => (
        <g key={r.label}>
          <line x1={P} x2={W - 6} y1={y(r.value)} y2={y(r.value)} stroke={r.color} strokeDasharray="4 3" />
          <text x={W - 8} y={y(r.value) - 3} textAnchor="end" fontSize={9} fill={r.color}>{r.label}</text>
        </g>
      ))}
      <polyline fill="none" stroke="rgb(31,99,180)" strokeWidth={1.8} points={ys.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
      <text x={P - 4} y={y(top / 1.05)} textAnchor="end" fontSize={9} fill="currentColor">{Math.round(top / 1.05)}</text>
      <text x={P - 4} y={H - 18} textAnchor="end" fontSize={9} fill="currentColor">0</text>
      <text x={(P + W) / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="currentColor">training episode (simulated days)</text>
    </svg>
  );
}

/** What the trained agent does in each situation: rows are fill level, columns are flow direction. */
function PolicyGrid({ t }: { t: Trained }) {
  const grid = policyGrid(t.Q, t.N);
  const look = (c: { action: number; seen: boolean }) =>
    !c.seen ? { text: "?", cls: "bg-muted text-muted-foreground" }
    : c.action === GIVE ? { text: "give", cls: "bg-[rgb(38,160,85)]/20 font-semibold" }
    : c.action === TAKE ? { text: "take", cls: "bg-[rgb(200,36,59)]/20 font-semibold" }
    : { text: "–", cls: "" };
  return (
    <table className="w-full border-collapse text-center text-[11px]">
      <thead>
        <tr><th className="p-1 text-left font-normal text-muted-foreground">fill ↓ / flow →</th>{TREND_NAMES.map((n) => <th key={n} className="p-1 font-normal text-muted-foreground">{n}</th>)}</tr>
      </thead>
      <tbody>
        {Array.from({ length: LEVELS }, (_, lv) => (
          <tr key={lv}>
            <th className="p-1 text-left font-normal text-muted-foreground">{LEVEL_NAMES[lv]}</th>
            {Array.from({ length: TRENDS }, (_, tr) => { const l = look(grid[lv][tr]); return <td key={tr} className={`border border-border p-1 ${l.cls}`}>{l.text}</td>; })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Operator rebalancing analysis: trains a Q-learning agent in the browser and compares it with simple rules. */
export default function RlPanel({ stations, compiled, onClose }: { stations: RlStation[]; compiled: Compiled[]; onClose: () => void }) {
  const [phase, setPhase] = useState<"idle" | "training" | "evaluating" | "done">("idle");
  const [done, setDone] = useState(0);
  const [live, setLive] = useState<number[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const stop = useRef(false);
  useEffect(() => () => { stop.current = true; }, []);

  async function run() {
    stop.current = false;
    setResult(null); setDone(0); setLive([]); setPhase("training");
    const trained = await trainQ(stations, cfg, {
      episodes: EPISODES, seed: TRAIN_SEED, shouldStop: () => stop.current,
      onProgress: (n, curve) => { setDone(n); setLive(curve.slice()); },
    });
    if (stop.current) { setPhase("idle"); return; }
    setPhase("evaluating");
    await new Promise((r) => setTimeout(r, 30)); // let the "evaluating" label paint before the sync loop
    const policies = [
      { key: "none", label: "Do nothing", p: nonePolicy },
      { key: "reactive", label: "Rule: act when ≤15% or ≥85% full", p: reactivePolicy },
      { key: "forecast", label: "Rule: act on Markov forecast (≥35% in 30 min)", p: forecastPolicy(compiled) },
      { key: "q", label: "Q-learning agent", p: qPolicy(trained.Q, trained.N) },
    ];
    const rows = policies.map(({ key, label, p }) => ({ key, label, ev: evaluate(stations, p, cfg, EVAL_DAYS, EVAL_SEED) }));
    setResult({ rows, trained });
    setPhase("done");
  }

  const none = result?.rows[0].ev, q = result?.rows[3].ev;
  const best = result ? result.rows.slice(1).reduce((a, b) => (b.ev.score < a.ev.score ? b : a)) : null;
  const curve = result?.trained.curve ?? live;

  return (
    <div role="dialog" aria-label="Reinforcement learning analysis" className="absolute inset-x-3 bottom-3 top-14 z-20 mx-auto flex max-w-3xl flex-col overflow-hidden rounded-lg border border-border bg-card text-sm shadow-lg">
      <div className="flex items-start justify-between gap-3 border-b border-border p-3">
        <div>
          <h2 className="font-semibold">Reinforcement learning: where should the van go?</h2>
          <p className="text-xs text-muted-foreground">Q-learning agent vs simple rules, on {stations.length} stations, simulated with the Markov model.</p>
        </div>
        <button type="button" aria-label="Close" className="text-lg leading-none text-muted-foreground" onClick={onClose}>×</button>
      </div>

      <div className="space-y-4 overflow-y-auto p-3">
        <section className="space-y-1 text-xs leading-snug text-muted-foreground">
          <p>
            One van services up to {cfg.visits} stations every {cfg.stepMin} min from {Math.floor(cfg.startMin / 60)}:00 to midnight. At a station it takes up to {cfg.moveMax} bikes
            (relieves a full station) or gives up to {cfg.moveMax} (fills an empty one). The van holds {cfg.vanCap} bikes and starts with {cfg.startLoad}.
          </p>
          <p>
            The agent sees each station&apos;s fill level and whether it is draining or filling this hour. It is rewarded by the riders it does <em>not</em> turn
            away, minus {cfg.moveCost} of a rider per bike moved. It learns by playing {EPISODES} simulated days, then every policy is scored on the same {EVAL_DAYS} new days.
          </p>
        </section>

        <div className="flex items-center gap-3">
          {phase === "training" || phase === "evaluating" ? (
            <button type="button" className="rounded-md border border-border px-3 py-1.5" onClick={() => { stop.current = true; }}>Stop</button>
          ) : (
            <button type="button" className="rounded-md bg-foreground px-3 py-1.5 text-background" onClick={() => void run()}>
              {result ? "Train again" : "Train and compare"}
            </button>
          )}
          {phase === "training" && (
            <div className="flex-1">
              <div className="h-2 overflow-hidden rounded-full bg-muted"><div className="h-full bg-[rgb(31,99,180)] transition-all" style={{ width: `${(done / EPISODES) * 100}%` }} /></div>
              <p className="mt-0.5 text-xs text-muted-foreground">Training: day {done} of {EPISODES}</p>
            </div>
          )}
          {phase === "evaluating" && <p className="text-xs text-muted-foreground">Scoring all policies on {EVAL_DAYS} new days…</p>}
        </div>

        {curve.length > 0 && (
          <section>
            <h3 className="text-xs font-semibold">Riders turned away per day while learning</h3>
            <Curve
              curve={curve}
              refs={result ? result.rows.slice(0, 3).map((r, i) => ({ label: r.key === "none" ? "do nothing" : r.key === "reactive" ? "rule" : "forecast rule", value: r.ev.mean.unmet, color: ["rgb(200,36,59)", "rgb(120,120,120)", "rgb(38,160,85)"][i] })) : []}
            />
          </section>
        )}

        {result && none && q && best && (
          <>
            <section>
              <h3 className="text-xs font-semibold">Results, average per day over {EVAL_DAYS} unseen days</h3>
              <div className="overflow-x-auto">
                <table className="mt-1 w-full text-xs">
                  <thead>
                    <tr className="border-b border-border text-left text-muted-foreground">
                      <th className="py-1 pr-2 font-normal">Policy</th>
                      <th className="px-2 font-normal">Riders turned away</th>
                      <th className="px-2 font-normal">vs do nothing</th>
                      <th className="px-2 font-normal">Empty h</th>
                      <th className="px-2 font-normal">Full h</th>
                      <th className="px-2 font-normal">Bikes moved</th>
                      <th className="px-2 font-normal">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map(({ key, label, ev }) => {
                      const d = key === "none" ? null : pairedDiff(ev.unmetByDay, none.unmetByDay);
                      return (
                        <tr key={key} className={`border-b border-border/60 ${key === "q" ? "font-semibold" : ""}`}>
                          <td className="py-1 pr-2">{label}</td>
                          <td className="px-2 tabular-nums">{f1(ev.mean.unmet)}</td>
                          <td className="px-2 tabular-nums">{d ? `${f1(d.mean)} ± ${f1(d.se)}` : "–"}</td>
                          <td className="px-2 tabular-nums">{f1(ev.mean.emptyH)}</td>
                          <td className="px-2 tabular-nums">{f1(ev.mean.fullH)}</td>
                          <td className="px-2 tabular-nums">{Math.round(ev.mean.moved)}</td>
                          <td className="px-2 tabular-nums">{f1(ev.score)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                Empty h and Full h are station-hours a day. Score is riders turned away plus {cfg.moveCost} per bike moved (lower is better). &quot;vs do nothing&quot; is the
                per-day difference with its standard error; every policy sees the same rider randomness each day.
              </p>
              <p className="mt-2 text-xs leading-snug">
                The agent turns away {Math.round((1 - q.mean.unmet / none.mean.unmet) * 100)}% fewer riders than doing nothing. The best policy by score here is{" "}
                <span className="font-semibold">{best.label.toLowerCase()}</span>
                {best.key === "q" ? "." : `, which is ${f1(q.score - best.ev.score)} score points ahead of the agent. A learned policy is not automatically better than a good rule.`}
              </p>
            </section>

            <section>
              <h3 className="text-xs font-semibold">What the agent learned</h3>
              <PolicyGrid t={result.trained} />
              <p className="mt-1 text-[11px] leading-snug text-muted-foreground">
                Green: give bikes. Red: take bikes. –: leave it alone. ?: the agent rarely saw this situation. The pattern to expect is give at empty stations, take at full ones.
              </p>
            </section>
          </>
        )}

        <section className="rounded-md bg-muted/50 p-2 text-[11px] leading-snug text-muted-foreground">
          Limits: demand is the August weekday average, so the same pattern repeats every day. There is one van and no travel time. Rider incentives and app nudges are
          not modelled, and moving bikes does not change demand. Results show how the methods compare in this simulator, not what would happen on real streets.
        </section>
      </div>
    </div>
  );
}
