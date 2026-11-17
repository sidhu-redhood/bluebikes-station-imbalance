"""Planted-truth simulator for the station HMM.

Generates stations whose true bike counts, van visits and missed riders are known, records only what
Bluebikes would record (the trips that happened, and optionally GBFS polls), and keeps the truth on the
side. The HMM is then scored on how much of the hidden part it recovers.

Per station and 10-minute slot, the same story as station_hmm.py:
  slot start   van visit with probability p[h] (the count jumps to L ~ g), else a one-bike move with
               probability u, else nothing
  inside       riders arrive as Poisson processes (pickups lam, returns mu, per clock hour and day type),
               event by event. A rider who finds no bike, or no dock, leaves and nothing is recorded.
  polls        every `poll_every` minutes at a random phase, count plus discretized Laplace noise

Options to stress the model where it is simplified:
  day_noise    each day's demand is multiplied by a Gamma factor with mean 1 and this spread, the same
               for every station (weather, holidays). The HMM estimates one multiplier per day.
  station_day_noise   the same, but drawn separately for each station and day (a game at Fenway, an
               event on campus). The HMM has no term for this.
  van_inside   van visits happen at a uniform time inside the slot instead of at its start.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coolboston.bike_markov import STEP_MIN
from coolboston.hmm_data import SlotData, build_slots
from coolboston.station_hmm import poll_log_table


@dataclass
class StationTruth:
    count: np.ndarray        # (T,) bikes at slot start
    after: np.ndarray        # (T,) bikes after the unrecorded move
    visit: np.ndarray        # (T,) van visit in the slot
    added: np.ndarray        # (T,) bikes added by unrecorded moves
    removed: np.ndarray      # (T,) bikes removed
    wanted_out: np.ndarray   # (T,) riders who wanted a bike
    wanted_in: np.ndarray    # (T,) riders who wanted to return one
    empty_min: np.ndarray    # (T,) minutes with no bike
    full_min: np.ndarray     # (T,) minutes with no free dock


@dataclass
class NetworkTruth:
    count: np.ndarray        # (T, S)
    after: np.ndarray
    visit: np.ndarray
    added: np.ndarray
    removed: np.ndarray
    wanted_out: np.ndarray
    wanted_in: np.ndarray
    empty_min: np.ndarray
    full_min: np.ndarray
    lam: np.ndarray          # (S, 2, 24) true demand on a day with multiplier 1
    mu: np.ndarray
    day_mult: np.ndarray     # (D,) shared demand multiplier of each day

    def demand(self, weekend: bool = False, data: SlotData | None = None) -> tuple[np.ndarray, np.ndarray]:
        """True (lam, mu) on an average day of the type, over the days simulated."""
        d = int(weekend)
        if data is None:
            return self.lam[:, d], self.mu[:, d]
        types = np.zeros(len(self.day_mult), dtype=int)
        types[data.day] = data.weekend.astype(int)
        m = self.day_mult[types == d].mean() if (types == d).any() else 1.0
        return self.lam[:, d] * m, self.mu[:, d] * m


def simulate_station(capacity: int, lam: np.ndarray, mu: np.ndarray, p: np.ndarray, g: np.ndarray, u: float,
                     start: pd.Timestamp, n_slots: int, rng: np.random.Generator, step_min: int = STEP_MIN,
                     poll_every: float | None = 10.0, poll_sigma: float = 0.6, poll_outlier: float = 0.02,
                     day_mult: np.ndarray | None = None, van_inside: bool = False, b0: int | None = None):
    """Returns (events, polls, truth). events = (minutes, delta); polls = (minutes, bikes) or empty.
    day_mult[k] multiplies demand on day k (default 1)."""
    C = int(capacity)
    stamps = pd.date_range(start, periods=n_slots, freq=f"{step_min}min")
    hours, days = np.asarray(stamps.hour), np.asarray(stamps.dayofweek >= 5).astype(int)
    day_index = np.asarray((stamps - stamps[0].normalize()).days)
    n_days = int(day_index.max()) + 1
    mult = np.ones(n_days) if day_mult is None else np.asarray(day_mult, dtype=float)

    b = int(rng.integers(0, C + 1)) if b0 is None else int(b0)
    ev_t, ev_d = [], []
    tr = {k: np.zeros(n_slots) for k in StationTruth.__dataclass_fields__}
    for t in range(n_slots):
        h, d, t0 = hours[t], days[t], t * step_min
        tr["count"][t] = b
        r = rng.random()
        target, when = None, t0
        if r < p[h]:
            target = int(rng.choice(C + 1, p=g))
            tr["visit"][t] = 1
            if van_inside:
                when = t0 + rng.random() * step_min
        elif r < p[h] + u:
            step = 1 if rng.random() < 0.5 else -1
            nb = min(max(b + step, 0), C)
            tr["added"][t] += max(nb - b, 0)
            tr["removed"][t] += max(b - nb, 0)
            b = nb
        if target is not None and not van_inside:
            tr["added"][t] += max(target - b, 0)
            tr["removed"][t] += max(b - target, 0)
            b, target = target, None
        tr["after"][t] = b
        a_rate = mu[d, h] * mult[day_index[t]] / 60.0
        d_rate = lam[d, h] * mult[day_index[t]] / 60.0
        total = a_rate + d_rate
        now, end = float(t0), float(t0 + step_min)
        while True:
            dt = rng.exponential(1 / total) if total > 0 else np.inf
            nxt = min(now + dt, end)
            if target is not None and when <= nxt:   # van arrives before the next rider
                span = when - now
                tr["empty_min"][t] += span * (b == 0)
                tr["full_min"][t] += span * (b == C)
                tr["added"][t] += max(target - b, 0)
                tr["removed"][t] += max(b - target, 0)
                b, target, now = target, None, when
                continue
            tr["empty_min"][t] += (nxt - now) * (b == 0)
            tr["full_min"][t] += (nxt - now) * (b == C)
            if now + dt >= end:
                break
            now += dt
            if rng.random() < a_rate / total:
                tr["wanted_in"][t] += 1
                if b < C:
                    b += 1
                    ev_t.append(now)
                    ev_d.append(1)
            else:
                tr["wanted_out"][t] += 1
                if b > 0:
                    b -= 1
                    ev_t.append(now)
                    ev_d.append(-1)
    events = (np.array(ev_t), np.array(ev_d, dtype=int))

    polls = (np.zeros(0), np.zeros(0, dtype=int))
    if poll_every:
        phase = rng.random() * poll_every
        pt = np.arange(phase, n_slots * step_min, poll_every)
        slot = (pt // step_min).astype(int)
        # True count at the poll: count after the move plus trips since the slot start. (With
        # van_inside the move may come later in the slot; the poll then sees the pre-visit count.)
        g0 = np.concatenate([[0], np.cumsum(events[1])])
        off = g0[np.searchsorted(events[0], pt)] - g0[np.searchsorted(events[0], slot * step_min)]
        truth = np.clip(tr["after"][slot] + off, 0, C).astype(int)
        table = np.exp(poll_log_table(C, poll_sigma, poll_outlier))
        seen = np.array([rng.choice(C + 1, p=table[m] / table[m].sum()) for m in truth])
        polls = (pt, seen)
    return events, polls, StationTruth(**{k: v for k, v in tr.items()})


def default_vans(capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """A plausible operator: visits peak in the rush hours and leave the station near half full."""
    p = np.full(24, 0.004)
    p[6:10] = 0.03
    p[15:19] = 0.03
    k = np.arange(capacity + 1)
    g = np.exp(-0.5 * ((k - capacity / 2) / max(capacity / 8, 1)) ** 2)
    return p, g / g.sum()


def _gamma_mult(rng, spread: float, n: int) -> np.ndarray:
    return rng.gamma(1 / spread ** 2, spread ** 2, n) if spread > 0 else np.ones(n)


def simulate_network(capacity, lam, mu, start: pd.Timestamp, n_slots: int, seed: int = 0,
                     poll_every: float | None = 10.0, vans=None, u: float = 0.003, day_noise: float = 0.0,
                     station_day_noise: float = 0.0, **kw):
    """capacity (S,), lam/mu (S, 2, 24). vans: list of (p, g) per station, default default_vans.
    Returns (SlotData, NetworkTruth)."""
    rng = np.random.default_rng(seed)
    capacity = np.asarray(capacity, dtype=int)
    n_days = int(np.ceil(n_slots * STEP_MIN / 1440)) + 1
    shared = _gamma_mult(rng, day_noise, n_days)
    events, polls, truths = [], [], []
    for s, C in enumerate(capacity):
        p, g = vans[s] if vans is not None else default_vans(int(C))
        mult = shared * _gamma_mult(rng, station_day_noise, n_days)
        e, pl, tr = simulate_station(int(C), lam[s], mu[s], p, g, u, start, n_slots, rng,
                                     poll_every=poll_every, day_mult=mult, **kw)
        events.append(e)
        polls.append(pl)
        truths.append(tr)
    data = build_slots(events, capacity, start, n_slots, polls=polls if poll_every else None,
                       station_ids=[f"SIM{s:02d}" for s in range(len(capacity))])
    stack = {k: np.stack([getattr(tr, k) for tr in truths], axis=1) for k in StationTruth.__dataclass_fields__}
    return data, NetworkTruth(lam=np.asarray(lam, float), mu=np.asarray(mu, float),
                              day_mult=shared[: int(data.day.max()) + 1], **stack)


def thin_polls(data: SlotData, keep_every: int = 6, holdout: float = 0.0, seed: int = 0):
    """Keep one poll in `keep_every` (per station, in time order), then hold out a random share of the
    kept ones. Returns (data with the remaining polls, held-out polls as (t, s, offset, y))."""
    rng = np.random.default_rng(seed)
    order = np.lexsort((data.poll_t, data.poll_s))
    rank = np.empty(len(order), dtype=int)
    for s in np.unique(data.poll_s):
        idx = order[data.poll_s[order] == s]
        rank[idx] = np.arange(len(idx))
    kept = (rank % keep_every) == 0
    hold = kept & (rng.random(len(kept)) < holdout)
    use = kept & ~hold
    pick = lambda m: (data.poll_t[m], data.poll_s[m], data.poll_offset[m], data.poll_y[m])  # noqa: E731
    new = SlotData(**{**data.__dict__, "poll_t": data.poll_t[use], "poll_s": data.poll_s[use],
                      "poll_offset": data.poll_offset[use], "poll_y": data.poll_y[use]})
    return new, pick(hold)
