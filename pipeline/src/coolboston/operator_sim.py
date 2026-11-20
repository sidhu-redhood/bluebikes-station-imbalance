"""Operator-response simulator: how many riders go unserved with and without van rebalancing?

Each station is the birth-death chain from bike_markov (exact event-by-event run). All stations share one
clock. Once per `interval` minutes the operator's policy looks at the bike counts and sends vans:
a van takes bikes from one station to another (capacity-limited, travel time ignored). Between moves the
stations run independently, as in the rest of the model.

Policies (each returns a list of Move):
  NoOp          do nothing (baseline).
  Threshold     rule of thumb: fullest stations -> emptiest stations, by b / capacity.
  Forecast      uses the Markov model: receivers = highest P(run out within the next hour),
                donors = highest P(fill up within the next hour), with a reserve left at the donor.

Metrics per simulated day, summed over stations:
  missed_pickups   riders who found the station empty        missed_returns   riders who found it full
  empty_hours / full_hours   station-hours spent empty / full        moved   bikes carried by vans (cost proxy)

Known limits (same as the model): stations are independent, so a rider turned away does not go to a
neighbouring station and total bikes are not conserved by trips. Vans have no travel time.

    python -m coolboston.operator_sim --bikes web/data/bikes.json --days 100 --vans 2 --van-capacity 12
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coolboston.bike_markov import StationModel, forecast, hourly_transitions, simulate_day, STEP_MIN

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Move:
    src: int
    dst: int
    bikes: int


@dataclass
class DayResult:
    missed_pickups: int = 0
    missed_returns: int = 0
    empty_hours: float = 0.0
    full_hours: float = 0.0
    moved: int = 0

    @property
    def unmet(self) -> int:
        return self.missed_pickups + self.missed_returns


# ---------- policies ----------

class NoOp:
    name = "no-op"

    def moves(self, models, bikes, minute):
        return []


class Threshold:
    """Pair the K fullest stations with the K emptiest, moving up to a van load but never past the middle."""
    name = "threshold"

    def __init__(self, vans: int, van_capacity: int, gap: float = 0.4):
        self.vans, self.van_capacity, self.gap = vans, van_capacity, gap

    def moves(self, models, bikes, minute):
        frac = np.array([b / m.capacity for b, m in zip(bikes, models)])
        donors = [i for i in np.argsort(-frac)[: self.vans]]
        receivers = [i for i in np.argsort(frac)[: self.vans]]
        out, used = [], set()
        for d, r in zip(donors, receivers):
            if d == r or d in used or r in used or frac[d] - frac[r] < self.gap:
                continue
            n = _load(models, bikes, d, r, self.van_capacity, reserve=0)
            if n > 0:
                out.append(Move(int(d), int(r), n))
                used.update((d, r))
        return out


class Forecast:
    """Send bikes where the Markov model says a run-out is likely, from where a fill-up is likely."""
    name = "forecast"

    def __init__(self, models, vans: int, van_capacity: int, horizon_min: int = 60, p_min: float = 0.2):
        self.vans, self.van_capacity, self.p_min = vans, van_capacity, p_min
        self.steps = max(1, horizon_min // STEP_MIN)
        # Absorbing-chain matrices once per station: P(hit empty / full at any moment within the horizon).
        self.to_empty = [hourly_transitions(m, absorbing=(0,)) for m in models]
        self.to_full = [hourly_transitions(m, absorbing=(m.capacity,)) for m in models]

    def probabilities(self, models, bikes, minute):
        p_e = np.array([forecast(m, b, minute, self.steps, mats=self.to_empty[i])[-1, 0]
                        for i, (m, b) in enumerate(zip(models, bikes))])
        p_f = np.array([forecast(m, b, minute, self.steps, mats=self.to_full[i])[-1, m.capacity]
                        for i, (m, b) in enumerate(zip(models, bikes))])
        return p_e, p_f

    def moves(self, models, bikes, minute):
        p_e, p_f = self.probabilities(models, bikes, minute)
        receivers = [i for i in np.argsort(-p_e) if p_e[i] >= self.p_min][: self.vans]
        out, used = [], set(receivers)
        for r in receivers:
            # best donor: likely to fill, has bikes to spare, not already involved this round
            order = [i for i in np.argsort(-(p_f - p_e)) if i not in used and bikes[i] > _reserve(models[i])]
            if not order:
                break
            d = int(order[0])
            n = _load(models, bikes, d, int(r), self.van_capacity, reserve=_reserve(models[d]))
            if n > 0:
                out.append(Move(d, int(r), n))
                used.add(d)
        return out


def _reserve(m: StationModel) -> int:
    return max(2, m.capacity // 5)


def _load(models, bikes, src, dst, van_capacity, reserve):
    return int(max(0, min(van_capacity, bikes[src] - reserve, models[dst].capacity - bikes[dst] - reserve)))


# ---------- simulation ----------

def apply_moves(models, bikes, moves):
    """Apply vans' moves in place, clipped to bikes available and docks free. Returns bikes carried."""
    carried = 0
    for mv in moves:
        n = min(mv.bikes, bikes[mv.src], models[mv.dst].capacity - bikes[mv.dst])
        if n > 0:
            bikes[mv.src] -= n
            bikes[mv.dst] += n
            carried += n
    return carried


def run_day(models, bikes0, policy, rng, start_min=300, interval=60, minutes=24 * 60) -> DayResult:
    bikes = list(bikes0)
    res = DayResult()
    t = start_min
    end = start_min + minutes
    while t < end:
        res.moved += apply_moves(models, bikes, policy.moves(models, bikes, t))
        span = min(interval, end - t)
        for i, m in enumerate(models):
            tr = simulate_day(m, bikes[i], rng, start_min=t, minutes=span)
            bikes[i] = int(tr.bikes[-1])
            res.missed_pickups += tr.missed_pickups
            res.missed_returns += tr.missed_returns
            res.empty_hours += tr.empty_minutes / 60
            res.full_hours += tr.full_minutes / 60
        t += span
    return res


def evaluate(models, bikes0, policy, days, seed=0, **kw):
    """Run `days` simulated days. Same seeds for every policy (common random numbers) so differences are
    mostly the policy, not luck."""
    return [run_day(models, bikes0, policy, np.random.default_rng(seed + d), **kw) for d in range(days)]


def summarize(results: list[DayResult]) -> dict:
    n = len(results)
    out = {"days": n}
    for key in ("missed_pickups", "missed_returns", "unmet", "empty_hours", "full_hours", "moved"):
        v = np.array([getattr(r, key) for r in results], dtype=float)
        out[key] = v.mean()
        out[key + "_se"] = v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return out


def load_cluster(path: Path):
    doc = json.loads(Path(path).read_text())
    models = [StationModel(s["capacity"], np.array(s["ratesIn"], float), np.array(s["ratesOut"], float))
              for s in doc["stations"]]
    return doc, models, [s["initialBikes"] for s in doc["stations"]]


def compare(models, bikes0, vans, van_capacity, days, seed=0, interval=60):
    policies = [NoOp(), Threshold(vans, van_capacity), Forecast(models, vans, van_capacity)]
    return {p.name: summarize(evaluate(models, bikes0, p, days, seed, interval=interval)) for p in policies}


def print_table(rows: dict):
    base = rows["no-op"]["unmet"]
    print(f"{'policy':<11}{'unmet/day':>11}{'  (+/- se)':>10}{'vs no-op':>10}{'missed pickups':>16}{'missed returns':>16}"
          f"{'empty st-h':>12}{'full st-h':>11}{'bikes moved':>13}")
    for name, r in rows.items():
        rel = "" if name == "no-op" else f"{(r['unmet'] / base - 1) * 100:+.0f}%"
        print(f"{name:<11}{r['unmet']:>11.1f}{r['unmet_se']:>10.1f}{rel:>10}{r['missed_pickups']:>16.1f}"
              f"{r['missed_returns']:>16.1f}{r['empty_hours']:>12.1f}{r['full_hours']:>11.1f}{r['moved']:>13.0f}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare operator policies on a bikes.json cluster")
    p.add_argument("--bikes", type=Path, default=ROOT / "web" / "data" / "bikes.json")
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--vans", type=int, default=2)
    p.add_argument("--van-capacity", type=int, default=12)
    p.add_argument("--interval", type=int, default=60, help="minutes between van dispatches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", type=Path, help="also write the summary table as CSV")
    args = p.parse_args(argv)

    doc, models, bikes0 = load_cluster(args.bikes)
    print(f"{len(models)} stations from {args.bikes.name} (source {doc['source']['kind']}); "
          f"{args.vans} van(s) x {args.van_capacity} bikes, dispatch every {args.interval} min, {args.days} days\n")
    rows = compare(models, bikes0, args.vans, args.van_capacity, args.days, args.seed, args.interval)
    print_table(rows)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            cols = list(next(iter(rows.values())))
            w.writerow(["policy"] + cols)
            for name, r in rows.items():
                w.writerow([name] + [r[c] for c in cols])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
