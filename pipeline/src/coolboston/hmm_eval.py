"""Score the station HMM on simulated stations where the truth is known (hmm_sim.py).

The simulated stations use the cluster's real capacities and the August weekday rates from bikes.json,
plus a planted van schedule. Each setting refits the HMM and compares it with two baselines:

  assumption replay   what the repo does today: start each day at 05:00 at build_bikes' half-full
                      ASSUMPTION, then add and remove the recorded trips (no van moves)
  last poll + trips   (settings with polls) start from the most recent poll and add the trips since

Run from the repo root (each setting takes about a minute):
    PYTHONPATH=pipeline/src python3 -m coolboston.hmm_eval
    PYTHONPATH=pipeline/src python3 -m coolboston.hmm_eval --settings trips polls_hourly --days 21
Results go to outputs/hmm/eval/<setting>.json and outputs/hmm/eval/eval.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.build_bikes import initial_bikes
from coolboston.hmm_data import SlotData
from coolboston.hmm_sim import NetworkTruth, simulate_network, thin_polls
from coolboston.station_hmm import WEEKDAY, HMMConfig, _Prep, count_mean, count_quantiles, fit

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "hmm" / "eval"
START = pd.Timestamp("2026-08-03")   # a Monday

SETTINGS = {
    # name: (description, sim options, poll thinning (keep 1 in k, held-out share) or None, HMM options)
    "trips": ("trips only", {}, None, {}),
    "trips_no_demand": ("trips only, demand evidence off (ablation)", {}, None, {"use_demand": False}),
    "polls_hourly": ("trips + one poll an hour, 20% of those held out", {}, (6, 0.2), {}),
    "polls_10min": ("trips + a poll every 10 min, 20% held out", {}, (1, 0.2), {}),
    "bursty_days": ("trips only, whole-cluster demand swings 30% day to day (weather)",
                    {"day_noise": 0.3}, None, {}),
    "bursty_no_day_term": ("same, with the per-day multiplier switched off (ablation)",
                           {"day_noise": 0.3}, None, {"prior_day": 1e9}),
    "event_days": ("trips only, each station has its own 30% day swings (model has no term)",
                   {"station_day_noise": 0.3}, None, {}),
    "vans_mid_slot": ("trips only, vans arrive mid-slot (model assumes slot start)",
                      {"van_inside": True}, None, {}),
}


def load_rates(path: Path):
    doc = json.loads(path.read_text())
    st = doc["stations"]
    cap = np.array([s["capacity"] for s in st])
    out = np.array([s["ratesOut"] for s in st], dtype=float)
    inn = np.array([s["ratesIn"] for s in st], dtype=float)
    lam = np.stack([out, 0.8 * out], axis=1)    # weekends: 80% of weekday demand
    mu = np.stack([inn, 0.8 * inn], axis=1)
    return cap, lam, mu


def _replay(data: SlotData, starts: dict[int, np.ndarray], first: np.ndarray) -> np.ndarray:
    """Counts at slot starts from replaying trips, reset to `starts[t]` at the slots listed."""
    T, S = data.n_slots, data.n_stations
    out = np.zeros((T, S), dtype=int)
    b = first.copy()
    for t in range(T):
        if t in starts:
            b = starts[t].copy()
        out[t] = b
        b = np.clip(b + data.net[t], 0, data.capacity)
    return out


def assumption_replay(data: SlotData) -> np.ndarray:
    wk = ~data.weekend
    days = wk.sum() * data.step_min / 1440
    r_out = np.stack([np.bincount(data.hour[wk], weights=data.dep[wk, s], minlength=24) for s in range(data.n_stations)]) / days
    r_in = np.stack([np.bincount(data.hour[wk], weights=data.arr[wk, s], minlength=24) for s in range(data.n_stations)]) / days
    b5 = np.array([initial_bikes(int(c), r_in[s], r_out[s]) for s, c in enumerate(data.capacity)])
    per_day = 1440 // data.step_min
    starts = {t: b5 for t in range(0, data.n_slots) if t % per_day == 300 // data.step_min}
    return _replay(data, starts, b5)


def last_poll_replay(data: SlotData) -> np.ndarray:
    """Most recent poll (minus the trips between slot start and poll) plus trips since; assumption before."""
    base = assumption_replay(data)
    T, S = data.n_slots, data.n_stations
    out = base.copy()
    for s in range(S):
        sel = np.flatnonzero(data.poll_s == s)
        if not len(sel):
            continue
        order = sel[np.argsort(data.poll_t[sel], kind="stable")]
        anchor = {}
        for i in order:                      # last poll in each slot wins
            anchor[int(data.poll_t[i])] = int(data.poll_y[i]) - int(data.poll_offset[i])
        b, have = None, False
        for t in range(T):
            if have:
                out[t, s] = b
            if t in anchor:
                b, have = int(np.clip(anchor[t], 0, data.capacity[s])), True
            if have:
                b = int(np.clip(b + data.net[t, s], 0, data.capacity[s]))
    return out


def _empty_minutes(prep: _Prep, counts: np.ndarray) -> np.ndarray:
    t = np.arange(prep.T)[:, None]
    s = np.arange(prep.S)[None, :]
    return prep.step - prep.A[t, s, counts]


def score(name: str, data: SlotData, truth: NetworkTruth, held, cfg: HMMConfig) -> dict:
    t0 = time.time()
    res = fit(data, cfg)
    seconds = time.time() - t0
    post = res.posterior
    prep = _Prep(data, cfg)
    days = data.n_slots * data.step_min / 1440
    mean = count_mean(post.gamma)
    lo, hi = count_quantiles(post.gamma, (0.1, 0.9))
    base = assumption_replay(data)
    out = {
        "setting": name, "description": SETTINGS[name][0], "stations": data.n_stations, "days": days,
        "iterations": len(res.history), "seconds": round(seconds, 1),
        "em_monotone": bool(np.all(np.diff(res.history) >= -1e-9 * abs(res.history[-1]))),
        "count_mae": {"hmm": float(np.abs(mean - truth.count).mean()),
                      "assumption_replay": float(np.abs(base - truth.count).mean())},
        "count_cover80": float(((truth.count >= lo) & (truth.count <= hi)).mean()),
        "count_width80": float((hi - lo).mean()),
    }
    emp_true = truth.empty_min.sum(axis=0) / 60 / days
    emp_hmm = post.empty_min.sum(axis=0) / 60 / days
    emp_base = _empty_minutes(prep, base).sum(axis=0) / 60 / days
    out["empty_h_per_day"] = {"true": float(emp_true.mean()), "hmm": float(emp_hmm.mean()),
                              "assumption_replay": float(emp_base.mean())}
    out["empty_h_per_day_abs_err"] = {"hmm": float(np.abs(emp_hmm - emp_true).mean()),
                                      "assumption_replay": float(np.abs(emp_base - emp_true).mean())}
    wk = ~data.weekend
    wk_days = wk.sum() * data.step_min / 1440
    true_d = truth.demand(False, data)[0].sum(axis=1)
    raw_d = data.dep[wk].sum(axis=0) / wk_days
    hmm_d = res.params.demand(data)[0][:, WEEKDAY].sum(axis=1)
    out["weekday_pickup_demand_err_pct"] = {"raw_trips": float(100 * np.mean(np.abs(raw_d - true_d) / true_d)),
                                            "hmm": float(100 * np.mean(np.abs(hmm_d - true_d) / true_d))}
    out["weekday_pickup_demand_total"] = {"true": float(true_d.sum()), "raw_trips": float(raw_d.sum()),
                                          "hmm": float(hmm_d.sum())}
    moved_true = (truth.added + truth.removed).sum(axis=0) / days
    moved_hmm = (post.added + post.removed).sum(axis=0) / days
    out["bikes_moved_per_station_day"] = {"true": float(moved_true.mean()), "hmm": float(moved_hmm.mean()),
                                          "abs_err": float(np.abs(moved_hmm - moved_true).mean())}
    if data.n_polls:
        lp = last_poll_replay(data)
        out["count_mae"]["last_poll_replay"] = float(np.abs(lp - truth.count).mean())
    if held is not None and len(held[0]):
        t, s, off, y = held
        C = data.capacity[s]
        g_after = post.gamma_after[t, s].astype(float)                        # (P, N)
        m = np.clip(np.arange(g_after.shape[1])[None, :] + off[:, None], 0, C[:, None])
        pred = (g_after * m).sum(axis=1)
        lp = last_poll_replay(data)
        lp_pred = np.clip(lp[t, s] + off, 0, C)                               # slot start + trips to poll
        out["heldout_poll_mae"] = {"hmm": float(np.abs(pred - y).mean()),
                                   "last_poll_replay": float(np.abs(lp_pred - y).mean()), "n": int(len(y))}
    return out


def run(names, days: int, seed: int, rates: Path, stations: int | None) -> list[dict]:
    cap, lam, mu = load_rates(rates)
    if stations:
        cap, lam, mu = cap[:stations], lam[:stations], mu[:stations]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        desc, sim_kw, thin, hmm_kw = SETTINGS[name]
        polls = thin is not None
        data, truth = simulate_network(cap, lam, mu, START, days * 144, seed=seed,
                                       poll_every=10.0 if polls else None, **sim_kw)
        held = None
        if polls:
            data, held = thin_polls(data, keep_every=thin[0], holdout=thin[1], seed=seed)
        cfg = HMMConfig(**hmm_kw)
        r = score(name, data, truth, held, cfg)
        (OUT / f"{name}.json").write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=2))
        results.append(r)
    return results


def report(path: Path = OUT) -> str:
    rows = [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]
    order = {k: i for i, k in enumerate(SETTINGS)}
    rows.sort(key=lambda r: order.get(r["setting"], 99))
    lines = ["| setting | count MAE: HMM | baseline | 80% range covers | empty h/day: true | HMM | baseline "
             "| demand error: raw | HMM | bikes moved/day: true | HMM |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        base = r["count_mae"].get("last_poll_replay", r["count_mae"]["assumption_replay"])
        e, d, m = r["empty_h_per_day"], r["weekday_pickup_demand_err_pct"], r["bikes_moved_per_station_day"]
        lines.append(f"| {r['description']} | {r['count_mae']['hmm']:.2f} | {base:.2f} | {100 * r['count_cover80']:.0f}% "
                     f"| {e['true']:.2f} | {e['hmm']:.2f} | {e['assumption_replay']:.2f} | {d['raw_trips']:.1f}% "
                     f"| {d['hmm']:.1f}% | {m['true']:.1f} | {m['hmm']:.1f} |")
    held = [r for r in rows if "heldout_poll_mae" in r]
    if held:
        lines += ["", "| setting | held-out polls | MAE: HMM | last poll + trips |", "|---|---|---|---|"]
        for r in held:
            h = r["heldout_poll_mae"]
            lines.append(f"| {r['description']} | {h['n']:,} | {h['hmm']:.2f} | {h['last_poll_replay']:.2f} |")
    first = rows[0] if rows else {}
    lines += ["", f"{first.get('stations', '?')} simulated stations, {first.get('days', '?'):.0f} days each. "
              "Baseline for count MAE: last poll + trips where polls exist, else the assumption replay."]
    text = "\n".join(lines) + "\n"
    (path / "eval.md").write_text(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="Planted-truth evaluation of the station HMM.")
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS), choices=list(SETTINGS))
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stations", type=int, default=None, help="use only the first N stations")
    ap.add_argument("--rates", type=Path, default=ROOT / "web" / "data" / "bikes.json")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)
    if not a.report_only:
        run(a.settings, a.days, a.seed, a.rates, a.stations)
    print(report())


if __name__ == "__main__":
    main()
