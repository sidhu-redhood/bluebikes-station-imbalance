"""Fit the station HMM on one month and write what it found.

Run from the repo root (inputs are Saad's processed files from code/load_data.py):
    PYTHONPATH=pipeline/src python3 -m coolboston.run_hmm --month 2026-08
    PYTHONPATH=pipeline/src python3 -m coolboston.run_hmm --month 2026-08 --polls polling/bluebikes.db
    PYTHONPATH=pipeline/src python3 -m coolboston.run_hmm --month 2026-08 --plots 4

Inputs   data/processed/neu_trips_<month>.csv, data/processed/neu_stations.csv, optional poller database
Outputs  outputs/hmm/<month>/ (gitignored):
  stations.csv      per station: empty and full hours a day, missed riders, van moves, certainty
  rates.csv         per station, day type and hour: raw and corrected pickup and return rates, van visits
  counts.csv.gz     per station and 10-minute slot: estimated bikes (mean, 10/50/90%), P(empty), P(full)
  bikes_hmm.json    bikes.json (schema v1) with corrected rates and estimated 05:00 counts. Load it in the
                    demo with "Load bikes.json", or copy it over web/data/bikes.json to use it in the app.
  days.csv          per calendar day: trips and the learned demand multiplier (1 = an average day)
  params.npz        learned parameters and the EM objective per iteration
  summary.md        the headline numbers
  plots/            with --plots N: timelines for the N most-censored stations, demand by hour
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.build_bikes import am_net_flow, make_doc, write_bikes
from coolboston.hmm_data import SlotData, load_month, load_stations_csv
from coolboston.station_hmm import (
    WEEKDAY, WEEKEND, FitResult, HMMConfig, count_mean, count_quantiles, fit,
)

ROOT = Path(__file__).resolve().parents[3]
START_MINUTE = 5 * 60   # bikes.json simulates from 05:00


def infeasible_days(data: SlotData) -> np.ndarray:
    """(S,) station-days whose trips are impossible without unrecorded moves: the day's net flow
    swings by more than the station has docks. Model-free; shows why the hidden moves are needed."""
    per_day = int(round(24 * 60 / data.step_min))
    T = data.n_slots - data.n_slots % per_day
    net = data.net[:T].reshape(-1, per_day, data.n_stations)
    start = np.cumsum(net, axis=1) - net                     # count change since midnight, slot start
    low = np.minimum((start + data.lo[:T].reshape(net.shape)).min(axis=1), 0)
    high = np.maximum((start + data.hi[:T].reshape(net.shape)).max(axis=1), 0)
    return ((high - low) > data.capacity[None, :]).sum(axis=0)


def station_table(data: SlotData, res: FitResult) -> pd.DataFrame:
    post, par = res.posterior, res.params
    days = data.n_slots * data.step_min / (24 * 60)
    wk = ~data.weekend
    wk_days = wk.sum() * data.step_min / (24 * 60)
    lo, hi = count_quantiles(post.gamma, (0.1, 0.9))
    raw_out = data.dep[wk].sum(axis=0) / wk_days
    raw_in = data.arr[wk].sum(axis=0) / wk_days
    lam, mu = par.demand(data)
    hmm_out = lam[:, WEEKDAY, :].sum(axis=1)
    hmm_in = mu[:, WEEKDAY, :].sum(axis=1)
    return pd.DataFrame({
        "station_id": data.station_ids,
        "name": data.names,
        "capacity": data.capacity,
        "empty_h_per_day": post.empty_min.sum(axis=0) / 60 / days,
        "full_h_per_day": post.full_min.sum(axis=0) / 60 / days,
        "weekday_pickups_seen": raw_out,
        "weekday_pickups_wanted": hmm_out,
        "missed_pickups_pct": 100 * (1 - raw_out / np.maximum(hmm_out, 1e-9)),
        "weekday_returns_seen": raw_in,
        "weekday_returns_wanted": hmm_in,
        "missed_returns_pct": 100 * (1 - raw_in / np.maximum(hmm_in, 1e-9)),
        "van_visits_per_day": post.p_visit.sum(axis=0) / days,
        "bikes_added_per_day": post.added.sum(axis=0) / days,
        "bikes_removed_per_day": post.removed.sum(axis=0) / days,
        "infeasible_days": infeasible_days(data),
        "count_80pct_width": (hi - lo).mean(axis=0),
    })


def rates_table(data: SlotData, res: FitResult) -> pd.DataFrame:
    hours = np.zeros((2, 24))
    np.add.at(hours, (data.weekend.astype(int), data.hour), data.step_min / 60)
    dep = np.zeros((data.n_stations, 2, 24))
    arr = np.zeros_like(dep)
    np.add.at(dep, (slice(None), data.weekend.astype(int), data.hour), data.dep.T)
    np.add.at(arr, (slice(None), data.weekend.astype(int), data.hour), data.arr.T)
    safe = np.where(hours > 0, hours, np.nan)
    visits = np.zeros((data.n_stations, 24))
    np.add.at(visits, (slice(None), data.hour), res.posterior.p_visit.T)
    n_days = data.n_slots * data.step_min / (24 * 60)
    lam, mu = res.params.demand(data)
    rows = []
    for s, sid in enumerate(data.station_ids):
        for d, label in ((WEEKDAY, "weekday"), (WEEKEND, "weekend")):
            for h in range(24):
                rows.append((sid, label, h, dep[s, d, h] / safe[d, h], lam[s, d, h],
                             arr[s, d, h] / safe[d, h], mu[s, d, h], visits[s, h] / n_days))
    return pd.DataFrame(rows, columns=["station_id", "day_type", "hour", "pickups_per_h_seen",
                                       "pickups_per_h_wanted", "returns_per_h_seen", "returns_per_h_wanted",
                                       "van_visits_per_day_this_hour"])


def counts_table(data: SlotData, res: FitResult) -> pd.DataFrame:
    post = res.posterior
    T, S = data.n_slots, data.n_stations
    q = count_quantiles(post.gamma, (0.1, 0.5, 0.9))
    rows = np.arange(S)
    return pd.DataFrame({
        "slot_start": np.repeat(data.slot_times(), S),
        "station_id": np.tile(data.station_ids, T),
        "bikes_mean": count_mean(post.gamma).ravel().round(2),
        "bikes_p10": q[0].ravel(), "bikes_p50": q[1].ravel(), "bikes_p90": q[2].ravel(),
        "p_empty": post.gamma[:, :, 0].ravel().round(4),
        "p_full": post.gamma[:, rows, data.capacity].ravel().round(4),
        "empty_min": post.empty_min.ravel().round(2),
        "full_min": post.full_min.ravel().round(2),
        "p_van_visit": post.p_visit.ravel().round(4),
    })


def bikes_doc(data: SlotData, res: FitResult, stations: pd.DataFrame, month: str) -> dict:
    """bikes.json with the HMM's corrected weekday rates, and the estimated weekday 05:00 count in place
    of the half-full ASSUMPTION used by build_bikes."""
    from coolboston.neighborhoods import Neighborhoods

    per_day = int(round(24 * 60 / data.step_min))
    at_five = (np.arange(data.n_slots) % per_day == START_MINUTE // data.step_min) & ~data.weekend
    start = count_mean(res.posterior.gamma[at_five]).mean(axis=0)
    hoods = Neighborhoods.load()
    nu = ["Fenway", "Back Bay", "Mission Hill", "South End", "Roxbury", "Jamaica Plain"]
    info = stations.set_index("station_id")
    lam, mu = res.params.demand(data)
    records = []
    for s, sid in enumerate(data.station_ids):
        cap = int(data.capacity[s])
        r_in, r_out = mu[s, WEEKDAY], lam[s, WEEKDAY]
        rec = {
            "id": sid, "name": data.names[s],
            "lat": round(float(info.at[sid, "lat"]), 6), "lon": round(float(info.at[sid, "lon"]), 6),
            "capacity": cap, "initialBikes": int(np.clip(round(float(start[s])), 0, cap)),
            "ratesIn": [round(float(x), 3) for x in r_in], "ratesOut": [round(float(x), 3) for x in r_out],
        }
        hood = hoods.name_for(rec["lon"], rec["lat"], nu)
        if hood:
            rec["neighborhood"] = hood
        flow = am_net_flow(r_in, r_out)
        rec["amNetFlow"] = round(flow, 3)
        rec["isAmDestination"] = bool(flow > 0)
        records.append(rec)
    source = {"kind": "hmm-corrected", "month": month, "weekdays": int(at_five.sum()),
              "base": "saad-neu-cluster", "stations": len(records)}
    return make_doc(records, source)


def days_table(data: SlotData, res: FitResult) -> pd.DataFrame:
    D = int(data.day.max()) + 1
    trips = np.bincount(data.day, weights=(data.dep + data.arr).sum(axis=1), minlength=D)
    weekend = np.zeros(D, dtype=bool)
    weekend[data.day] = data.weekend
    m = res.params.day_scale
    rel = m / np.array([m[weekend == w].mean() for w in weekend])
    dates = pd.date_range(data.start.normalize(), periods=D, freq="D")
    return pd.DataFrame({"date": dates.date, "day_type": np.where(weekend, "weekend", "weekday"),
                         "trip_events": trips.astype(int), "demand_vs_average_day_of_type": rel})


def summary_md(table: pd.DataFrame, data: SlotData, res: FitResult, month: str, seconds: float) -> str:
    t = table
    days = days_table(data, res)
    quiet = days.sort_values("demand_vs_average_day_of_type").head(3)
    worst = t.sort_values("missed_pickups_pct", ascending=False).head(5)
    vans = t.sort_values("bikes_removed_per_day", ascending=False).head(5)
    lines = [
        f"# Station HMM, {month}",
        "",
        f"{data.n_stations} stations, {data.n_slots} slots of {data.step_min} min, "
        f"{int(data.dep.sum()):,} pickups and {int(data.arr.sum()):,} returns, {data.n_polls:,} poll snapshots. "
        f"EM: {len(res.history)} iterations, {seconds:.0f} s.",
        "",
        f"- Station-days that trips alone cannot explain (unrecorded moves are certain): "
        f"{int(t.infeasible_days.sum())} of {data.n_stations * int(data.n_slots * data.step_min // 1440)}.",
        f"- Average station: empty {t.empty_h_per_day.mean():.1f} h and full {t.full_h_per_day.mean():.1f} h a day.",
        f"- Weekday pickups seen {t.weekday_pickups_seen.sum():,.0f} a day across the cluster; riders who wanted "
        f"one {t.weekday_pickups_wanted.sum():,.0f} "
        f"({100 * (1 - t.weekday_pickups_seen.sum() / t.weekday_pickups_wanted.sum()):.1f}% missed).",
        f"- Unrecorded moves: {t.bikes_added_per_day.sum():,.0f} bikes added and "
        f"{t.bikes_removed_per_day.sum():,.0f} removed a day, about {t.van_visits_per_day.sum():.0f} van visits.",
        "- Quietest days (demand vs an average day of the same type): "
        + ", ".join(f"{r.date} ({r.demand_vs_average_day_of_type:.2f})" for r in quiet.itertuples()) + ".",
        "",
        "Most missed pickups:",
        "",
        "| station | docks | empty h/day | seen/day | wanted/day | missed |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in worst.iterrows():
        lines.append(f"| {r['name']} | {r.capacity} | {r.empty_h_per_day:.1f} | {r.weekday_pickups_seen:.0f} | "
                     f"{r.weekday_pickups_wanted:.0f} | {r.missed_pickups_pct:.0f}% |")
    lines += ["", "Most bikes taken away by unrecorded moves:", "",
              "| station | docks | removed/day | added/day | visits/day |", "|---|---|---|---|---|"]
    for _, r in vans.iterrows():
        lines.append(f"| {r['name']} | {r.capacity} | {r.bikes_removed_per_day:.1f} | "
                     f"{r.bikes_added_per_day:.1f} | {r.van_visits_per_day:.1f} |")
    lines += ["", "Without poll data the absolute count is only pinned down where a station often hits 0 or "
              "capacity; `count_80pct_width` in stations.csv says how sure each estimate is."]
    return "\n".join(lines) + "\n"


def plot_station(data: SlotData, res: FitResult, s: int, path: Path, day0: int = 0, days: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_day = int(round(24 * 60 / data.step_min))
    sl = slice(day0 * per_day, (day0 + days) * per_day)
    post = res.posterior
    times = data.slot_times()[sl]
    lo, mid, hi = count_quantiles(post.gamma[sl, s][:, None, :], (0.1, 0.5, 0.9))[:, :, 0]
    mean = count_mean(post.gamma[sl, s])
    C = int(data.capacity[s])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.fill_between(times, lo, hi, step="post", alpha=0.3, label="80% range")
    ax.plot(times, mean, lw=1.4, label="estimated bikes")
    ax.axhline(C, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    visit = post.p_visit[sl, s] > 0.5
    ax.scatter(times[visit], mean[visit], marker="v", color="crimson", zorder=3, label="van visit (p > 0.5)")
    sel = (data.poll_s == s) & (data.poll_t >= sl.start) & (data.poll_t < sl.stop)
    if sel.any():
        pt = data.slot_times()[data.poll_t[sel]]
        ax.scatter(pt, data.poll_y[sel], s=6, color="black", label="poll")
    ax.set_ylabel("bikes")
    ax.set_ylim(-0.5, C + 0.5)
    ax.set_title(f"{data.names[s]} ({data.station_ids[s]}, {C} docks)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax2.fill_between(times, 0, post.empty_min[sl, s] / data.step_min, step="post", color="tab:red", alpha=0.6,
                     label="share of slot empty")
    ax2.fill_between(times, 0, -post.full_min[sl, s] / data.step_min, step="post", color="tab:blue", alpha=0.6,
                     label="share of slot full")
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_yticks([-1, 0, 1], ["full", "", "empty"])
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_demand(data: SlotData, res: FitResult, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wk = ~data.weekend
    hours = np.bincount(data.hour[wk], minlength=24) * data.step_min / 60
    seen = np.bincount(data.hour[wk], weights=data.dep[wk].sum(axis=1), minlength=24) / hours
    wanted = res.params.demand(data)[0][:, WEEKDAY, :].sum(axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(np.arange(24) - 0.2, seen, 0.4, label="pickups in the trip data")
    ax.bar(np.arange(24) + 0.2, wanted, 0.4, label="riders who wanted a bike (HMM)")
    ax.set_xlabel("hour (weekdays)")
    ax.set_ylabel("per hour, whole cluster")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit the station HMM on one month of trips (and polls).")
    ap.add_argument("--month", default="2026-08", help="YYYY-MM (default 2026-08)")
    ap.add_argument("--trips", type=Path, help="default data/processed/neu_trips_<month>.csv")
    ap.add_argument("--stations", type=Path, default=ROOT / "data" / "processed" / "neu_stations.csv")
    ap.add_argument("--polls", type=Path, help="poller SQLite database (polling/bluebikes.db)")
    ap.add_argument("--out", type=Path, help="default outputs/hmm/<month>")
    ap.add_argument("--max-iter", type=int, default=HMMConfig.max_iter)
    ap.add_argument("--no-demand", action="store_true", help="ignore the demand evidence (ablation)")
    ap.add_argument("--plots", type=int, default=0, help="timelines for the N most-censored stations")
    a = ap.parse_args(argv)

    trips = a.trips or ROOT / "data" / "processed" / f"neu_trips_{a.month}.csv"
    out = a.out or ROOT / "outputs" / "hmm" / a.month
    for p in (trips, a.stations) + ((a.polls,) if a.polls else ()):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")
    out.mkdir(parents=True, exist_ok=True)

    data = load_month(trips, a.stations, a.month, polls_db=a.polls)
    print(f"[hmm] {data.n_stations} stations, {data.n_slots} slots, {int(data.dep.sum()):,} pickups, "
          f"{int(data.arr.sum()):,} returns, {data.n_polls:,} polls")
    config = HMMConfig(max_iter=a.max_iter, use_demand=not a.no_demand)
    t0 = time.time()
    res = fit(data, config, verbose=True)
    seconds = time.time() - t0

    table = station_table(data, res)
    table.round(3).to_csv(out / "stations.csv", index=False)
    rates_table(data, res).round(4).to_csv(out / "rates.csv", index=False)
    days_table(data, res).round(3).to_csv(out / "days.csv", index=False)
    counts_table(data, res).to_csv(out / "counts.csv.gz", index=False)
    np.savez(out / "params.npz", p=res.params.p, g=res.params.g, u=res.params.u, lam=res.params.lam,
             mu=res.params.mu, day_scale=res.params.day_scale, history=np.array(res.history),
             station_ids=np.array(data.station_ids))
    if config.use_demand:
        write_bikes(bikes_doc(data, res, load_stations_csv(a.stations), a.month), out / "bikes_hmm.json")
    (out / "summary.md").write_text(summary_md(table, data, res, a.month, seconds))
    (out / "run.json").write_text(json.dumps({
        "month": a.month, "trips": str(trips), "stations": str(a.stations), "polls": str(a.polls) if a.polls else None,
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "iterations": len(res.history), "seconds": round(seconds, 1), "generated": dt.datetime.now().isoformat(),
    }, indent=2, default=str))

    if a.plots:
        (out / "plots").mkdir(exist_ok=True)
        plot_demand(data, res, out / "plots" / "demand_by_hour.png")
        first_weekday = int(np.argmax(~data.weekend)) // int(round(24 * 60 / data.step_min))
        for s in table.sort_values("missed_pickups_pct", ascending=False).index[: a.plots]:
            plot_station(data, res, int(s), out / "plots" / f"station_{data.station_ids[s]}.png", day0=first_weekday)

    print((out / "summary.md").read_text())
    print(f"[hmm] wrote {out}")


if __name__ == "__main__":
    main()
