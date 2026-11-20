import numpy as np
import pytest

from coolboston.bike_markov import StationModel, simulate_day
from coolboston.hmm_fit import fit_eps, fit_station, log_likelihood, scaled


def truth_model():
    out = np.full(24, 6.0)
    out[7:10] = 18.0
    return StationModel(12, np.full(24, 8.0), out)


def simulate_polls(model, days, seed, eps=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(days):
        counts = simulate_day(model, model.capacity // 2, rng).bikes[::10][:144].astype(object)
        for t in range(len(counts)):
            if rng.random() < eps:
                counts[t] = int(np.clip(counts[t] + rng.choice([-1, 1]), 0, model.capacity))
        out.append(list(counts))
    return out


def test_fit_recovers_rate_level_from_wrong_start():
    truth = truth_model()
    obs = simulate_polls(truth, 10, seed=1)
    wrong = scaled(truth, 0.5, 0.5)  # the trip log under-counted everything by half
    fitted, eps, s_in, s_out, ll = fit_station(wrong, obs)
    assert 1.6 < s_in < 2.5 and 1.6 < s_out < 2.5  # truth is 2.0 for both
    assert ll > log_likelihood(wrong, obs)


def test_fit_improves_likelihood_over_default():
    truth = truth_model()
    obs = simulate_polls(truth, 6, seed=2)
    start = scaled(truth, 1.8, 0.6)
    fitted, *_ = fit_station(start, obs)
    assert log_likelihood(fitted, obs) > log_likelihood(start, obs)


def test_fit_eps_learns_noise_level_when_count_moves_slowly():
    # Slow station: the count rarely changes between polls, so a one-off wrong reading is identifiable.
    slow = StationModel(12, np.full(24, 0.8), np.full(24, 0.8))
    clean = fit_eps(slow, simulate_polls(slow, 8, seed=3, eps=0.0))
    noisy = fit_eps(slow, simulate_polls(slow, 8, seed=3, eps=0.2))
    assert clean < 0.05
    assert noisy > clean + 0.05


def test_blank_days_are_ignored():
    truth = truth_model()
    assert log_likelihood(truth, [[None] * 144]) == 0.0
    assert fit_eps(truth, [[None] * 144], eps=0.07) == pytest.approx(0.07)
