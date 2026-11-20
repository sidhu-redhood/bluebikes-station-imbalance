"""Correct the demand censoring in bikes.json for every station, using the poller data and the HMM.

Reads a bikes.json (trip-based rates, censored), runs coolboston.hmm on each station's polled bike counts,
and writes a new bikes.json with the corrected ratesIn/ratesOut. Everything else (ids, capacity, schema)
is unchanged, so the web app and demo read the result exactly like the original.

Two modes:
  default  heuristic: HMM posterior P(empty)/P(full) per hour, then rate / (1 - P)  (hmm.corrected_rates)
  --fit    maximum likelihood: learn a level multiplier for ratesIn and ratesOut from the polled counts
           (hmm_fit.fit_station). The chain already blocks at empty/full, so the fitted rates are the
           uncensored demand and no division is applied.

Stations with fewer than --min-days days of poller data keep their original rates.

    export PYTHONPATH=$PWD/pipeline/src
    python -m coolboston.build_bikes_hmm --db path/to/bluebikes.db --start 2026-11-03 --days 14
    python -m coolboston.build_bikes_hmm --db ... --start ... --report data/processed/hmm_report.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from coolboston.bike_markov import HOURS, StationModel, hourly_transitions
from coolboston.build_bikes import validate
from coolboston.hmm import corrected_rates, hourly_empty_full, load_day, posterior
from coolboston.hmm_fit import fit_station

ROOT = Path(__file__).resolve().parents[3]
IN_PATH = ROOT / "web" / "data" / "bikes.json"
OUT_PATH = ROOT / "web" / "data" / "bikes.hmm.json"


def station_model(s: dict) -> StationModel:
    return StationModel(s["capacity"], np.array(s["ratesIn"], dtype=float), np.array(s["ratesOut"], dtype=float))


def censoring_profile(model: StationModel, day_obs: list[list], eps: float | None = None):
    """Average hourly P(empty), P(full) over the days that have any poller data.
    Returns (p_empty, p_full, n_days); n_days == 0 when every day is blank."""
    kwargs = {} if eps is None else {"eps": eps}
    empty_sum, full_sum, n = np.zeros(HOURS), np.zeros(HOURS), 0
    for obs in day_obs:
        if all(o is None for o in obs):
            continue
        gamma, _ = posterior(model, obs, **kwargs)
        e, f = hourly_empty_full(gamma)
        empty_sum += np.nan_to_num(e)
        full_sum += np.nan_to_num(f)
        n += 1
    if n == 0:
        return np.zeros(HOURS), np.zeros(HOURS), 0
    return empty_sum / n, full_sum / n, n


def correct_doc(doc: dict, db: str | Path, first: date, days: int, min_days: int = 2, fit: bool = False):
    """Return (new_doc, report_rows). new_doc is a deep copy; the input is not modified."""
    out = copy.deepcopy(doc)
    report = []
    for s in out["stations"]:
        model = station_model(s)
        day_obs = [load_day(db, s["id"], first + timedelta(days=i)) for i in range(days)]
        p_empty, p_full, n = censoring_profile(model, day_obs)
        row = {"id": s["id"], "name": s["name"], "days": n, "corrected": False,
               "rate_out_before": float(model.rates_out.sum()), "rate_out_after": float(model.rates_out.sum()),
               "rate_in_before": float(model.rates_in.sum()), "rate_in_after": float(model.rates_in.sum()),
               "mean_p_empty": float(p_empty.mean()), "mean_p_full": float(p_full.mean())}
        if n >= min_days:
            if fit:
                fitted, _, s_in, s_out, _ = fit_station(model, day_obs)
                rin, rout = fitted.rates_in, fitted.rates_out
                row.update(scale_in=s_in, scale_out=s_out)
            else:
                rin, rout = corrected_rates(model, p_empty, p_full)
            s["ratesIn"] = [round(float(x), 3) for x in rin]
            s["ratesOut"] = [round(float(x), 3) for x in rout]
            row.update(corrected=True, rate_out_after=float(rout.sum()), rate_in_after=float(rin.sum()))
        report.append(row)
    done = sum(r["corrected"] for r in report)
    out["source"] = {
        **doc["source"],
        "kind": doc["source"]["kind"] + "+hmm",
        "hmm": {"pollerStart": first.isoformat(), "pollerDays": days, "minDays": min_days,
                "stationsCorrected": done, "stationsTotal": len(report), "method": "mle" if fit else "posterior-divide"},
    }
    return validate(out), report


def print_report(report: list[dict]):
    print(f"{'station':<38}{'days':>5}{'P(empty)':>9}{'P(full)':>9}{'out/day':>16}{'in/day':>16}")
    for r in sorted(report, key=lambda r: r["rate_out_after"] - r["rate_out_before"], reverse=True):
        tag = "" if r["corrected"] else "  (kept: not enough poller data)"
        print(f"{r['name'][:36]:<38}{r['days']:>5}{r['mean_p_empty']:>9.3f}{r['mean_p_full']:>9.3f}"
              f"{r['rate_out_before']:>8.0f}->{r['rate_out_after']:<6.0f}{r['rate_in_before']:>8.0f}->{r['rate_in_after']:<6.0f}{tag}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", required=True, help="poller SQLite database (opened read-only)")
    p.add_argument("--start", required=True, help="first poller day, YYYY-MM-DD (Boston time)")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--min-days", type=int, default=2, help="fewest poller days needed to correct a station")
    p.add_argument("--fit", action="store_true", help="learn the rate level by maximum likelihood instead of the posterior divide")
    p.add_argument("--in", dest="src", type=Path, default=IN_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--report", type=Path, help="also write the per-station before/after table as CSV")
    args = p.parse_args(argv)

    if not Path(args.db).exists():
        p.error(f"poller database not found: {args.db}")
    doc = json.loads(args.src.read_text())
    new_doc, report = correct_doc(doc, args.db, date.fromisoformat(args.start), args.days, args.min_days, args.fit)
    done = sum(r["corrected"] for r in report)
    if done == 0:
        print("no station had enough poller data; nothing written. Check --start/--days and the station ids.")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(new_doc, allow_nan=False, separators=(",", ":")))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in report for k in r)))
            w.writeheader()
            w.writerows(report)
    print_report(report)
    print(f"\nwrote {args.out} ({done}/{len(report)} stations corrected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
