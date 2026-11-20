"""Learn HMM parameters from polled bike counts (maximum likelihood), instead of fixing them.

Learned per station, from the poller's counts alone:

  eps    probability a polled count is off by one. Closed-form EM: E-step is the forward-backward
         posterior (hmm.posterior); M-step is the expected fraction of polls where the hidden count
         differs from the reading. OPT-IN (fit_station(fit_noise=True)): eps and the rate level are
         confounded when the count moves several bikes per 10 minutes, because fast real movement can be
         explained as reading noise. On simulated data with true eps = 0 the joint fit drifted to eps ~ 0.25
         and under-estimated the rates by about 30%. The feed reports exact dock counts, so the default
         keeps eps fixed.
  scale  (s_in, s_out): multipliers on the trip-based hourly rate shapes. The chain blocks returns at
         FULL and pickups at EMPTY by itself, so its rates are true (uncensored) demand. Fitting the
         scales by likelihood therefore corrects the censoring directly, with no after-the-fact divide.
         There is no closed-form M-step here (the 10-minute matrices are matrix exponentials of the
         rates), so each multiplier is maximised by a log-spaced grid search, then refined around the best
         grid point. The hourly shape stays as estimated from trips; only the level is learned.

Observations are per day, a list of 144 polled counts (None = missed poll), as hmm.load_day returns.
"""

from __future__ import annotations

import numpy as np

from coolboston.bike_markov import StationModel, hourly_transitions, stationary_occupancy, _hour_at, STEP_MIN
from coolboston.hmm import EPS, emission_matrix, forward, posterior

SCALE_GRID = np.exp(np.linspace(np.log(0.5), np.log(4.0), 15))
EPS_BOUNDS = (0.005, 0.4)


def scaled(model: StationModel, s_in: float, s_out: float) -> StationModel:
    return StationModel(model.capacity, model.rates_in * s_in, model.rates_out * s_out)


def log_likelihood(model: StationModel, day_obs: list[list], eps: float = EPS) -> float:
    """Sum over days of log P(readings | model). Missed polls contribute nothing."""
    mats = hourly_transitions(model)
    B = emission_matrix(model.capacity, eps)
    total = 0.0
    for obs in day_obs:
        if all(o is None for o in obs):
            continue
        pi = stationary_occupancy(model, _hour_at(0, 0, STEP_MIN))
        _, scale = forward(pi, mats, obs, B, 0)
        total += float(np.log(scale).sum())
    return total


def fit_eps(model: StationModel, day_obs: list[list], eps: float = EPS, iters: int = 15, tol: float = 1e-4) -> float:
    """EM for the reading-error probability. Monotone in likelihood; stops when eps stops moving."""
    for _ in range(iters):
        wrong = seen = 0.0
        for obs in day_obs:
            if all(o is None for o in obs):
                continue
            gamma, _ = posterior(model, obs, eps=eps)
            for t, o in enumerate(obs):
                if o is None:
                    continue
                o = min(max(o, 0), model.capacity)
                wrong += 1.0 - gamma[t, o]
                seen += 1
        if seen == 0:
            return eps
        new = float(np.clip(wrong / seen, *EPS_BOUNDS))
        done = abs(new - eps) < tol
        eps = new
        if done:
            break
    return eps


def _best_scale(model, day_obs, eps, fixed_other, which):
    """Grid search one multiplier, then refine around the winner. Returns (scale, loglik).
    which = "in" / "out" moves one multiplier with the other fixed; "both" moves them together."""
    def ll(x):
        if which == "both":
            s_in, s_out = x, x
        else:
            s_in, s_out = (x, fixed_other) if which == "in" else (fixed_other, x)
        return log_likelihood(scaled(model, s_in, s_out), day_obs, eps)

    scores = [ll(x) for x in SCALE_GRID]
    i = int(np.argmax(scores))
    lo, hi = SCALE_GRID[max(i - 1, 0)], SCALE_GRID[min(i + 1, len(SCALE_GRID) - 1)]
    fine = np.exp(np.linspace(np.log(lo), np.log(hi), 7))
    fine_scores = [ll(x) for x in fine]
    j = int(np.argmax(fine_scores))
    return float(fine[j]), float(fine_scores[j])


def fit_station(model: StationModel, day_obs: list[list], rounds: int = 2, eps: float = EPS, fit_noise: bool = False):
    """Alternate the two rate multipliers (and eps, if fit_noise).
    Returns (fitted StationModel, eps, s_in, s_out, loglik)."""
    # The likelihood is a narrow ridge along "in and out scale together" (their ratio sets the occupancy,
    # their level sets how fast it moves). Moving one multiplier at a time stalls on that ridge, so the
    # common level is searched first and the two are only separated afterwards.
    level, ll = _best_scale(model, day_obs, eps, 1.0, "both")
    s_in = s_out = level
    for _ in range(rounds):
        if fit_noise:
            eps = fit_eps(scaled(model, s_in, s_out), day_obs, eps)
        s_out, _ = _best_scale(model, day_obs, eps, s_in, "out")
        s_in, ll = _best_scale(model, day_obs, eps, s_out, "in")
    return scaled(model, s_in, s_out), eps, s_in, s_out, ll
