"""Markov model of the bike count at one Bluebikes station (FAI project, part 1 of the tracker).

State: b = bikes docked, 0..C (C = dock capacity). b = 0 is EMPTY, b = C is FULL.

Within each clock hour h the station is a continuous-time birth-death chain:
  b -> b+1 at rate lam_in[h]   (a rider returns a bike; blocked when full -> unmet dock demand)
  b -> b-1 at rate lam_out[h]  (a rider takes a bike;  blocked when empty -> unmet bike demand)
Rates are trips per hour, estimated from trip history (build_bikes.py).

The discrete-time chain the tracker uses is P_h = expm(Q_h * dt) for one poll interval dt (10 min,
matching the GBFS polling plan). expm is computed by uniformization, so no scipy is needed:
  P = sum_k Poisson(k; q dt) * U^k,   U = I + Q / q,   q = max exit rate.

Forecasts:
  forecast()       distribution of b over the next k polls (inhomogeneous: P changes each hour)
  hit_probability() P(station is empty / full at ANY moment in the next k polls), from the same chain
                   with the target state made absorbing. This catches run-outs between polls.

Simulation (simulate_day) is the exact chain run event by event (Gillespie), and also counts the
riders it turns away. It is the ground truth the forecasts are scored against in the demo.

Everything is plain numpy. The team plan is PyTorch tensors only; every function here is matrix
products and elementwise ops, so swapping `np` for `torch` is mechanical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOURS = 24
# Poll interval, minutes. Same as the GBFS polling plan in the proposal.
STEP_MIN = 10
# Uniformization stops once the Poisson tail is below this.
UNIFORM_TOL = 1e-12


@dataclass(frozen=True)
class StationModel:
    capacity: int
    rates_in: np.ndarray   # (24,) bikes returned per hour
    rates_out: np.ndarray  # (24,) bikes taken per hour

    def __post_init__(self):
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")
        for name in ("rates_in", "rates_out"):
            r = np.asarray(getattr(self, name), dtype=float)
            if r.shape != (HOURS,) or not np.all(np.isfinite(r)) or np.any(r < 0):
                raise ValueError(f"{name}: expected 24 finite non-negative rates, got {r!r}")


def generator(capacity: int, lam_in: float, lam_out: float, absorbing: tuple[int, ...] = ()) -> np.ndarray:
    """Rate matrix Q (per minute) of the birth-death chain on 0..capacity. Rows sum to 0.
    States in `absorbing` get a zero row (once there, the chain stays)."""
    n = capacity + 1
    q = np.zeros((n, n))
    a, d = lam_in / 60.0, lam_out / 60.0
    for b in range(n):
        if b in absorbing:
            continue
        if b < capacity:
            q[b, b + 1] = a
        if b > 0:
            q[b, b - 1] = d
        q[b, b] = -q[b].sum()
    return q


def transition(q: np.ndarray, minutes: float) -> np.ndarray:
    """expm(Q * minutes) by uniformization. Rows are probability distributions."""
    n = q.shape[0]
    rate = float(np.max(-np.diag(q))) if n else 0.0
    if rate == 0.0:
        return np.eye(n)
    u = np.eye(n) + q / rate
    lam = rate * minutes
    # Poisson weights computed in log space so large lam does not underflow at k = 0.
    p = np.zeros((n, n))
    term = np.eye(n)
    k, total = 0, 0.0
    k_max = int(lam + 10 * math.sqrt(lam) + 20)
    while k <= k_max:
        w = math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))
        p += w * term
        total += w
        if 1.0 - total < UNIFORM_TOL and k > lam:
            break
        term = term @ u
        k += 1
    # Renormalise away the truncated tail.
    return p / p.sum(axis=1, keepdims=True)


def hourly_transitions(m: StationModel, step_min: float = STEP_MIN, absorbing: tuple[int, ...] = ()) -> np.ndarray:
    """(24, C+1, C+1): one poll-interval transition matrix per clock hour."""
    return np.stack(
        [transition(generator(m.capacity, m.rates_in[h], m.rates_out[h], absorbing), step_min) for h in range(HOURS)]
    )


def _hour_at(start_min: float, step: int, step_min: float) -> int:
    return int(((start_min + step * step_min) // 60) % HOURS)


def forecast(m: StationModel, bikes_now: int, start_min: float, steps: int, step_min: float = STEP_MIN,
             mats: np.ndarray | None = None) -> np.ndarray:
    """(steps+1, C+1): P(b at poll t | b at poll 0 = bikes_now). Row 0 is the point mass.
    start_min is minutes after midnight of poll 0. Each step uses the hour it starts in."""
    if not 0 <= bikes_now <= m.capacity:
        raise ValueError(f"bikes_now {bikes_now} outside 0..{m.capacity}")
    mats = hourly_transitions(m, step_min) if mats is None else mats
    out = np.zeros((steps + 1, m.capacity + 1))
    out[0, bikes_now] = 1.0
    for t in range(steps):
        out[t + 1] = out[t] @ mats[_hour_at(start_min, t, step_min)]
    return out


def hit_probability(m: StationModel, bikes_now: int, start_min: float, steps: int, target: str,
                    step_min: float = STEP_MIN) -> np.ndarray:
    """(steps+1,): P(station has been `target` ("empty" or "full") at some moment by poll t)."""
    state = {"empty": 0, "full": m.capacity}[target]
    mats = hourly_transitions(m, step_min, absorbing=(state,))
    return forecast(m, bikes_now, start_min, steps, step_min, mats)[:, state]


def stationary_occupancy(m: StationModel, hour: int) -> np.ndarray:
    """Long-run distribution if hour `hour`'s rates held forever (truncated geometric)."""
    a, d = m.rates_in[hour], m.rates_out[hour]
    b = np.arange(m.capacity + 1)
    if d == 0:
        pi = (b == m.capacity).astype(float)
    else:
        pi = (a / d) ** b if a > 0 else (b == 0).astype(float)
    return pi / pi.sum()


@dataclass
class DayTrace:
    bikes: np.ndarray        # (1441,) count at each minute from start_min
    missed_pickups: int      # riders who found the station empty
    missed_returns: int      # riders who found it full
    empty_minutes: float
    full_minutes: float


def simulate_day(m: StationModel, bikes0: int, rng: np.random.Generator, start_min: int = 0,
                 minutes: int = 24 * 60) -> DayTrace:
    """Exact event-by-event run of the chain (Gillespie). Exponential clocks are memoryless, so
    restarting them at each hour boundary, when rates change, is exact."""
    b = bikes0
    t = float(start_min)
    end = float(start_min + minutes)
    bikes = np.empty(minutes + 1, dtype=int)
    filled = 0  # next minute index to record
    missed_p = missed_r = 0
    empty_m = full_m = 0.0

    def record_until(t_next: float):
        nonlocal filled, empty_m, full_m
        stop = min(int(math.floor(t_next - start_min)), minutes)
        while filled <= stop:
            bikes[filled] = b
            filled += 1

    while t < end:
        h = int((t // 60) % HOURS)
        seg_end = min(end, (math.floor(t / 60) + 1) * 60.0)
        a, d = m.rates_in[h] / 60.0, m.rates_out[h] / 60.0
        total = a + d
        while True:
            dt = rng.exponential(1.0 / total) if total > 0 else math.inf
            t_ev = t + dt
            t_stop = min(t_ev, seg_end)
            if b == 0:
                empty_m += t_stop - t
            elif b == m.capacity:
                full_m += t_stop - t
            record_until(t_stop - 1e-9 if t_ev < seg_end else t_stop - 1e-9)
            if t_ev >= seg_end:
                t = seg_end
                break
            t = t_ev
            if rng.random() < a / total:  # a rider brings a bike
                if b < m.capacity:
                    b += 1
                else:
                    missed_r += 1
            else:  # a rider wants a bike
                if b > 0:
                    b -= 1
                else:
                    missed_p += 1
    record_until(end)
    return DayTrace(bikes, missed_p, missed_r, empty_m, full_m)
