"""Hidden Markov model of the bike count at each Bluebikes station (FAI project, part 2: hidden state).

The trip history says when bikes left and arrived, but never how many were docked. This model infers
that count, the riders the trips missed while a station was empty or full, and the van moves that the
trip data does not record.

Hidden state: b = rentable bikes docked at the start of a 10-minute slot, 0..C (C = docks).

One slot, for one station:
  1. Unrecorded move (vans, staff trips, broken bikes taken away). The count becomes b':
       nothing      probability 1 - p[h] - u
       van visit    probability p[h]: the van sets the station to a level L drawn from g
                    (operators rebalance to a target, so "jump to L" fits better than "add k")
       one bike     probability u/2 each way: b' = b + 1 or b - 1, kept inside 0..C
  2. Trips in the slot are observed exactly and in order, so the count follows the known path
     b' + offset(tau). They tell us about b' in two ways:
       feasibility  every pickup needed a bike and every return a dock. A violation by k bikes costs
                    exp(-kappa * k) instead of being impossible, so one bad record cannot break the fit.
       demand       riders want bikes at rate lam[h] * m[day] (Poisson) but are only served while there
                    is a bike: P(pickups | b') is proportional to exp(-rate * hours_with_a_bike(b')).
                    A busy hour with no pickups therefore points to an empty station. Returns work the
                    same way with rate mu[h] * m[day] and free docks. m[day] is one multiplier per day
                    shared by all stations (weather, holidays): without it a rainy day's missing riders
                    would be read as empty stations.
  3. Polls (GBFS snapshots, when we have them) see b' + offset(poll time), with a little noise.
  Next slot: b = b' + net trips of the slot.

Learning is EM (Baum-Welch), written out here:
  E-step  scaled forward-backward over all slots of the month, all stations at once
  M-step  p, g, u from the expected number of each kind of move (closed form; u is shared across
          hours, so p and u take a few rounds of coordinate ascent)
          lam = pickups / expected hours with a bike. This is the censored-Poisson estimate, i.e. the
          demand correction: trips that happened, divided by the time riders could actually get a bike.
          mu  = returns / expected hours with a free dock.
          m   = the day's trips / the trips the station rates predict for it.
  Weak priors (pseudo-counts) keep every estimate finite, e.g. demand in an hour when the station is
  always empty falls back to the station's average. EM never decreases the objective; the tests check.

Everything is numpy, like bike_markov.py. The team plan is PyTorch tensors; every step here is
elementwise ops, sums and index gathers, so swapping np for torch is mechanical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from coolboston.hmm_data import SlotData

HOURS = 24
WEEKDAY, WEEKEND = 0, 1


@dataclass(frozen=True)
class HMMConfig:
    kappa: float = math.log(1e4)      # cost per bike of an impossible trip record (data glitch)
    use_demand: bool = True           # demand evidence (busy hour with no pickups -> empty)
    poll_sigma: float = 0.6           # poll noise scale, bikes (discretized Laplace)
    poll_outlier: float = 0.02        # share of polls that are just wrong
    prior_p: tuple[float, float] = (0.1, 20.0)   # pseudo-slots (van visit, no visit) per station-hour
    prior_u: tuple[float, float] = (0.1, 50.0)   # pseudo-slots (one-bike move, none) per station
    prior_g: float = 0.5              # pseudo-visits per target level
    prior_rate_hours: float = 1.0     # pseudo-hours at the station's average rate, per demand cell
    prior_day: float = 10.0           # Gamma(a, a) prior on each day's demand multiplier (mean 1)
    max_iter: int = 60
    tol: float = 1e-7                 # stop when the objective improves by less than tol * |objective|


@dataclass
class HMMParams:
    p: np.ndarray      # (S, 24) chance of a van visit in a slot starting in hour h
    g: np.ndarray      # (S, N) where a van visit leaves the station (0 beyond capacity)
    u: np.ndarray      # (S,) chance of a one-bike unrecorded move per slot
    lam: np.ndarray    # (S, 2, 24) pickup demand, riders per hour, [weekday, weekend] x clock hour
    mu: np.ndarray     # (S, 2, 24) return demand
    day_scale: np.ndarray  # (D,) demand multiplier of each calendar day, shared by all stations

    def copy(self) -> "HMMParams":
        return HMMParams(*(x.copy() for x in (self.p, self.g, self.u, self.lam, self.mu, self.day_scale)))

    def demand(self, data: SlotData) -> tuple[np.ndarray, np.ndarray]:
        """(lam, mu) for an average day of each type: the rates times the mean multiplier of that day type.
        This is what to report; lam and mu alone are only defined up to the day multipliers."""
        dtype = _day_types(data)
        scale = np.array([self.day_scale[dtype == d].mean() if (dtype == d).any() else 1.0 for d in (0, 1)])
        return self.lam * scale[None, :, None], self.mu * scale[None, :, None]


@dataclass
class Posterior:
    gamma: np.ndarray        # (T, S, N) P(b at slot start | everything), float32
    gamma_after: np.ndarray  # (T, S, N) P(b' after the unrecorded move | everything), float32
    p_visit: np.ndarray      # (T, S) P(van visit in the slot)
    added: np.ndarray        # (T, S) expected bikes added by unrecorded moves
    removed: np.ndarray      # (T, S) expected bikes removed
    empty_min: np.ndarray    # (T, S) expected minutes with no bike
    full_min: np.ndarray     # (T, S) expected minutes with no free dock
    loglik: float


@dataclass
class FitResult:
    params: HMMParams
    posterior: Posterior
    history: list[float] = field(default_factory=list)  # objective after each EM iteration


# ---------------------------------------------------------------- precomputation

def _day_types(data: SlotData) -> np.ndarray:
    """(D,) 0 weekday, 1 weekend, for each calendar day in the data."""
    D = int(data.day.max()) + 1
    out = np.zeros(D, dtype=int)
    out[data.day] = data.weekend.astype(int)
    return out


class _Prep:
    """Per-slot arrays the passes reuse. Shapes (T, S, N) with N = max capacity + 2 (one spare state
    so a +1 move at capacity can be folded back)."""

    def __init__(self, data: SlotData, config: HMMConfig):
        T, S = data.n_slots, data.n_stations
        C = data.capacity.astype(int)
        N = int(C.max()) + 2
        self.T, self.S, self.N, self.C = T, S, N, C
        self.rows = np.arange(S)
        self.states = np.arange(N)
        self.mask = (self.states[None, :] <= C[:, None])
        self.hour = data.hour.astype(int)
        self.day = data.weekend.astype(int)
        self.date = data.day.astype(int)
        self.D = int(self.date.max()) + 1
        self.date_type = _day_types(data)
        self.net = data.net
        self.step = float(data.step_min)

        b = self.states[None, None, :]
        lo, hi = data.lo[:, :, None], data.hi[:, :, None]
        self.viol = (np.maximum(0, -(b + lo)) + np.maximum(0, b + hi - C[None, :, None])).astype(np.float32)

        # Minutes with a bike (A) and with a free dock (B), for every start count b'. Built from the
        # segments with a difference array along the state axis, then a cumulative sum.
        t, s, lev, dur = data.seg_t, data.seg_s, data.seg_level, data.seg_min.astype(np.float32)
        d = np.zeros((T, S, N + 1), dtype=np.float32)
        np.add.at(d, (t, s, np.clip(1 - lev, 0, N)), dur)           # bike while b' + lev >= 1
        self.A = np.cumsum(d, axis=2)[:, :, :N]
        d[:] = 0
        np.add.at(d, (t, s, np.zeros_like(t)), dur)                 # dock while b' + lev <= C - 1
        np.add.at(d, (t, s, np.clip(C[s] - lev, 0, N)), -dur)
        self.B = np.cumsum(d, axis=2)[:, :, :N]
        del d

        self.logE = self._poll_logs(data, config) if data.n_polls else None

        # Observed trips per demand cell, and slots per clock hour.
        self.dep = np.zeros((S, 2, HOURS))
        self.arr = np.zeros((S, 2, HOURS))
        np.add.at(self.dep, (slice(None), self.day, self.hour), data.dep.T)
        np.add.at(self.arr, (slice(None), self.day, self.hour), data.arr.T)
        self.cell_hours = np.zeros((2, HOURS))
        np.add.at(self.cell_hours, (self.day, self.hour), self.step / 60.0)
        self.slots_per_hour = np.bincount(self.hour, minlength=HOURS).astype(float)
        self.trips_per_date = np.bincount(self.date, weights=(data.dep + data.arr).sum(axis=1), minlength=self.D)

    def _poll_logs(self, data: SlotData, config: HMMConfig) -> np.ndarray:
        logE = np.zeros((self.T, self.S, self.N), dtype=np.float32)
        for s in range(self.S):
            sel = data.poll_s == s
            if not sel.any():
                continue
            C = int(self.C[s])
            table = poll_log_table(C, config.poll_sigma, config.poll_outlier)     # [true m, seen y]
            m = np.clip(self.states[None, :] + data.poll_offset[sel][:, None], 0, C)  # (P, N)
            y = np.clip(data.poll_y[sel], 0, C)[:, None]
            vals = table[m, y]
            vals[:, C + 1:] = 0.0
            np.add.at(logE, (data.poll_t[sel][:, None], s, self.states[None, :]), vals)
        return logE

    def shift_index(self, t: int) -> np.ndarray:
        """Flat index of the next slot's start state for each (station, b')."""
        idx = np.clip(self.states[None, :] + self.net[t][:, None], 0, self.C[:, None])
        return idx + self.rows[:, None] * self.N

    def log_factor(self, t: int, params: HMMParams, config: HMMConfig) -> np.ndarray:
        lf = -config.kappa * self.viol[t].astype(float)
        if config.use_demand:
            d, h, m = self.day[t], self.hour[t], params.day_scale[self.date[t]]
            lf = lf - m * (params.lam[:, d, h][:, None] * self.A[t] + params.mu[:, d, h][:, None] * self.B[t]) / 60.0
        if self.logE is not None:
            lf = lf + self.logE[t]
        return np.where(self.mask, lf, -np.inf)


def poll_log_table(capacity: int, sigma: float, outlier: float) -> np.ndarray:
    """log P(poll shows y | true count m), m and y in 0..C: discretized Laplace plus a uniform outlier."""
    k = np.arange(capacity + 1)
    w = np.exp(-np.abs(k[None, :] - k[:, None]) / sigma)
    lap = w / w.sum(axis=1, keepdims=True)
    return np.log((1 - outlier) * lap + outlier / (capacity + 1))


# ---------------------------------------------------------------- the move kernel V(b -> b')

def _up(x: np.ndarray, C: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Push mass one state up, folding the mass that would pass capacity back onto C."""
    out = np.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    out[rows, C] += out[rows, C + 1]
    out[rows, C + 1] = 0.0
    return out


def _down(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[:, :-1] = x[:, 1:]
    out[:, 0] += x[:, 0]
    return out


def _move_forward(alpha, p_h, u, g, prep):
    """Row vector times V. Returns the total and its three parts (stay, van, one-bike)."""
    stay = (1.0 - p_h - u)[:, None] * alpha
    van = (p_h * alpha.sum(axis=1))[:, None] * g
    one = (0.5 * u)[:, None] * (_up(alpha, prep.C, prep.rows) + _down(alpha))
    return stay + van + one, stay, van, one


def _move_backward(x, p_h, u, g, prep):
    """V times column vector: E[x(b') | b]."""
    x_up = np.zeros_like(x)
    x_up[:, :-1] = x[:, 1:]
    x_up[prep.rows, prep.C] = x[prep.rows, prep.C]
    x_down = np.empty_like(x)
    x_down[:, 1:] = x[:, :-1]
    x_down[:, 0] = x[:, 0]
    out = (1.0 - p_h - u)[:, None] * x + (p_h * (g * x).sum(axis=1))[:, None] + (0.5 * u)[:, None] * (x_up + x_down)
    return out * prep.mask


# ---------------------------------------------------------------- E-step

def _e_step(prep: _Prep, params: HMMParams, config: HMMConfig, store: bool = False):
    T, S, N = prep.T, prep.S, prep.N
    alpha0 = prep.mask / (prep.C + 1)[:, None]
    filt = np.empty((T, S, N))        # P(b'_t | data up to slot t)
    norm = np.empty((T, S))           # forward normalizers (of the max-shifted factor)
    loglik = 0.0

    alpha = alpha0
    for t in range(T):
        p_h = params.p[:, prep.hour[t]]
        m = _move_forward(alpha, p_h, params.u, params.g, prep)[0]
        lf = prep.log_factor(t, params, config)
        mx = lf.max(axis=1)
        a = m * np.exp(lf - mx[:, None])
        c = a.sum(axis=1)
        filt[t] = a / c[:, None]
        norm[t] = c
        loglik += float(np.sum(np.log(c) + mx))
        alpha = np.bincount(prep.shift_index(t).ravel(), weights=filt[t].ravel(), minlength=S * N).reshape(S, N)

    stats = {
        "van": np.zeros((S, HOURS)), "stay": np.zeros((S, HOURS)), "one": np.zeros(S),
        "G": np.zeros((S, N)), "EA": np.zeros((S, prep.D, HOURS)), "EB": np.zeros((S, prep.D, HOURS)),
    }
    if store:
        out = {k: np.zeros((T, S, N), dtype=np.float32) for k in ("gamma", "gamma_after")}
        out.update({k: np.zeros((T, S)) for k in ("p_visit", "added", "removed", "empty_min", "full_min")})
        lvl = prep.states.astype(float)

    beta = prep.mask.astype(float)    # beta at slot T
    for t in range(T - 1, -1, -1):
        h, d = prep.hour[t], prep.date[t]
        p_h = params.p[:, h]
        beta_after = beta.ravel()[prep.shift_index(t)]
        lf = prep.log_factor(t, params, config)
        mx = lf.max(axis=1)
        w = np.exp(lf - mx[:, None]) * beta_after / norm[t][:, None]
        if t == 0:
            alpha = alpha0
        else:
            alpha = np.bincount(prep.shift_index(t - 1).ravel(), weights=filt[t - 1].ravel(),
                                minlength=S * N).reshape(S, N)
        m, stay, van, one = _move_forward(alpha, p_h, params.u, params.g, prep)
        gam_after = m * w
        stats["van"][:, h] += (van * w).sum(axis=1)
        stats["stay"][:, h] += (stay * w).sum(axis=1)
        stats["one"] += (one * w).sum(axis=1)
        stats["G"] += van * w
        stats["EA"][:, d, h] += (gam_after * prep.A[t]).sum(axis=1)
        stats["EB"][:, d, h] += (gam_after * prep.B[t]).sum(axis=1)
        beta = _move_backward(w, p_h, params.u, params.g, prep)
        if store:
            out["gamma"][t] = alpha * beta
            out["gamma_after"][t] = gam_after
            out["p_visit"][t] = (van * w).sum(axis=1)
            out["empty_min"][t] = (gam_after * (prep.step - prep.A[t])).sum(axis=1)
            out["full_min"][t] = (gam_after * (prep.step - prep.B[t])).sum(axis=1)
            # Van visit from b to L: posterior weight alpha(b) * van_w(L), independent in b and L.
            van_w = van * w / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-300)
            ca, cba = np.cumsum(alpha, axis=1), np.cumsum(alpha * lvl, axis=1)
            below = np.zeros_like(alpha)                          # sum_{b<L} alpha(b) (L - b)
            below[:, 1:] = lvl[1:] * ca[:, :-1] - cba[:, :-1]
            tot_a, tot_ba = ca[:, -1:], cba[:, -1:]
            above = (tot_ba - cba) - lvl * (tot_a - ca)           # sum_{b>L} alpha(b) (b - L)
            up_real = np.zeros_like(alpha)                        # +1 moves that were not folded
            up_real[:, 1:] = alpha[:, :-1]
            up_real[prep.rows, prep.C + 1] = 0.0
            down_real = np.zeros_like(alpha)
            down_real[:, :-1] = alpha[:, 1:]
            half_u = (0.5 * params.u)[:, None]
            out["added"][t] = (van_w * below).sum(axis=1) + (half_u * up_real * w).sum(axis=1)
            out["removed"][t] = (van_w * above).sum(axis=1) + (half_u * down_real * w).sum(axis=1)

    post = None
    if store:
        post = Posterior(loglik=loglik, **out)
    return loglik, stats, post


# ---------------------------------------------------------------- M-step and objective

def _solve_u(S_h: np.ndarray, p: np.ndarray, U: float) -> float:
    """argmax_u  sum_h S_h log(1 - p_h - u) + U log u, by bisection on the derivative."""
    hi = 1.0 - p.max() - 1e-12
    lo = 1e-15
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if U / mid - np.sum(S_h / (1.0 - p - mid)) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _m_step(prep: _Prep, params: HMMParams, stats: dict, config: HMMConfig) -> HMMParams:
    new = params.copy()
    a_p, b_p = config.prior_p
    a_u, b_u = config.prior_u
    for s in range(prep.S):
        R = stats["van"][s] + a_p
        Sh = stats["stay"][s] + b_p + b_u / HOURS
        U = stats["one"][s] + a_u
        u = float(params.u[s])
        for _ in range(8):
            p = R * (1.0 - u) / (Sh + R)
            u = _solve_u(Sh, p, U)
        new.p[s], new.u[s] = p, u
    g = (stats["G"] + config.prior_g) * prep.mask
    new.g = g / g.sum(axis=1, keepdims=True)
    if config.use_demand:
        _m_step_demand(prep, new, stats, config)
    return new


def _m_step_demand(prep: _Prep, new: HMMParams, stats: dict, config: HMMConfig) -> None:
    """Coordinate ascent on (lam, mu) and the day multipliers m, then the exact best common rescaling
    (the likelihood depends only on lam * m, so only the priors fix the scale)."""
    b0, a = config.prior_rate_hours, config.prior_day
    EA, EB = stats["EA"] / 60.0, stats["EB"] / 60.0          # (S, D, 24) expected hours with bike / dock
    shape_l = prep.dep + b0 * _mean_rate(prep, prep.dep)
    shape_m = prep.arr + b0 * _mean_rate(prep, prep.arr)
    onehot = np.eye(2)[prep.date_type]                         # (D, 2)
    m = new.day_scale.copy()
    for _ in range(10):
        new.lam = shape_l / (np.einsum("sdh,d,dk->skh", EA, m, onehot) + b0)
        new.mu = shape_m / (np.einsum("sdh,d,dk->skh", EB, m, onehot) + b0)
        lam_d = new.lam[:, prep.date_type, :]                    # (S, D, 24)
        mu_d = new.mu[:, prep.date_type, :]
        expected = np.einsum("sdh,sdh->d", lam_d, EA) + np.einsum("sdh,sdh->d", mu_d, EB)
        m = (prep.trips_per_date + a) / (expected + a)
    # Rescale lam, mu by c and m by 1/c: the likelihood is unchanged (the trip counts' log c terms cancel),
    # and c maximizes what is left of the priors: K log c - L c - a M / c, a quadratic for f'(c) = 0.
    K = (shape_l - prep.dep).sum() + (shape_m - prep.arr).sum() - a * prep.D
    L = b0 * (new.lam.sum() + new.mu.sum())
    M = a * m.sum()
    c = (K + math.sqrt(K * K + 4 * L * M)) / (2 * L)
    new.lam, new.mu, new.day_scale = new.lam * c, new.mu * c, m / c


def _mean_rate(prep: _Prep, n: np.ndarray) -> np.ndarray:
    """(S, 2, 1): a station's average trips per hour on weekdays / weekends (the prior's centre)."""
    hrs = prep.cell_hours.sum(axis=1)[None, :, None]
    return np.where(hrs > 0, n.sum(axis=2, keepdims=True) / np.where(hrs > 0, hrs, 1.0), 0.0)


def _log_prior_and_constants(prep: _Prep, params: HMMParams, config: HMMConfig) -> float:
    """Terms of the objective that do not depend on the hidden path: the prior, and lam^pickups *
    mu^returns from the Poisson likelihood of the observed trips."""
    a_p, b_p = config.prior_p
    a_u, b_u = config.prior_u
    stay = 1.0 - params.p - params.u[:, None]
    val = np.sum(a_p * np.log(params.p) + (b_p + b_u / HOURS) * np.log(stay)) + np.sum(a_u * np.log(params.u))
    val += config.prior_g * np.sum(np.log(np.where(prep.mask, params.g, 1.0)))
    if config.use_demand:
        b0 = config.prior_rate_hours
        for rate, n in ((params.lam, prep.dep), (params.mu, prep.arr)):
            val += np.sum((n + b0 * _mean_rate(prep, n)) * np.log(np.maximum(rate, 1e-300)) - b0 * rate)
        m, a = params.day_scale, config.prior_day
        val += np.sum(prep.trips_per_date * np.log(m)) + np.sum(a * np.log(m) - a * m)
    return float(val)


def initial_params(data: SlotData) -> HMMParams:
    S = data.n_stations
    N = int(data.capacity.max()) + 2
    mask = np.arange(N)[None, :] <= data.capacity[:, None]
    hours = np.zeros((2, HOURS))
    day, hour = data.weekend.astype(int), data.hour.astype(int)
    np.add.at(hours, (day, hour), data.step_min / 60.0)
    dep = np.zeros((S, 2, HOURS))
    arr = np.zeros((S, 2, HOURS))
    np.add.at(dep, (slice(None), day, hour), data.dep.T)
    np.add.at(arr, (slice(None), day, hour), data.arr.T)
    safe = np.where(hours > 0, hours, 1.0)
    return HMMParams(
        p=np.full((S, HOURS), 0.005), g=mask / mask.sum(axis=1, keepdims=True), u=np.full(S, 0.002),
        lam=np.maximum(dep / safe, 1e-3), mu=np.maximum(arr / safe, 1e-3),
        day_scale=np.ones(int(data.day.max()) + 1),
    )


def fit(data: SlotData, config: HMMConfig = HMMConfig(), init: HMMParams | None = None,
        learn: bool = True, verbose: bool = False) -> FitResult:
    """EM from `init` (default: raw trip rates, rare moves), then one more pass that keeps the
    posterior. learn=False only runs that last pass with the given parameters."""
    prep = _Prep(data, config)
    params = (init or initial_params(data)).copy()
    history: list[float] = []
    if learn:
        for it in range(config.max_iter):
            ll, stats, _ = _e_step(prep, params, config)
            obj = ll + _log_prior_and_constants(prep, params, config)
            history.append(obj)
            if verbose:
                print(f"[hmm] EM {it + 1:2d}  objective {obj:,.2f}")
            if it > 0 and abs(history[-1] - history[-2]) < config.tol * abs(history[-1]):
                break
            params = _m_step(prep, params, stats, config)
    ll, _, post = _e_step(prep, params, config, store=True)
    return FitResult(params=params, posterior=post, history=history)


# ---------------------------------------------------------------- summaries

def count_quantiles(gamma: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    """(len(qs), T, S): smallest b with P(count <= b) >= q."""
    cdf = np.cumsum(gamma, axis=-1)
    return np.stack([(cdf < q - 1e-6).sum(axis=-1) for q in qs])


def count_mean(gamma: np.ndarray) -> np.ndarray:
    return (gamma * np.arange(gamma.shape[-1])).sum(axis=-1)
