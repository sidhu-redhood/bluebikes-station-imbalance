import numpy as np
import pytest

from coolboston.bike_markov import (
    StationModel, forecast, generator, hit_probability, simulate_day, stationary_occupancy, transition,
)


def _model(cap=10, a=6.0, d=6.0):
    return StationModel(cap, np.full(24, a), np.full(24, d))


def test_generator_rows_sum_to_zero_and_respect_bounds():
    q = generator(5, 6.0, 12.0)
    assert np.allclose(q.sum(axis=1), 0)
    assert q[5, 5 + 0] == pytest.approx(-0.2)  # full: only departures (12/h = 0.2/min)
    assert q[0, 1] == pytest.approx(0.1) and q[0, 0] == pytest.approx(-0.1)  # empty: only arrivals


def test_transition_is_stochastic_and_matches_series_expm():
    q = generator(8, 9.0, 4.0)
    p = transition(q, 10)
    assert np.allclose(p.sum(axis=1), 1) and np.all(p >= -1e-15)
    # Reference: Taylor series with many terms (small matrix, short time: converges).
    ref, term = np.eye(9), np.eye(9)
    for k in range(1, 80):
        term = term @ (q * 10) / k
        ref += term
    assert np.allclose(p, ref, atol=1e-10)


def test_semigroup_two_halves_equal_whole():
    q = generator(12, 20.0, 7.0)
    assert np.allclose(transition(q, 5) @ transition(q, 5), transition(q, 10), atol=1e-12)


def test_zero_rates_are_identity():
    assert np.allclose(transition(generator(4, 0, 0), 10), np.eye(5))


def test_forecast_converges_to_stationary_distribution():
    m = _model(cap=10, a=8.0, d=4.0)
    f = forecast(m, 3, start_min=0, steps=400)
    assert np.allclose(f[-1], stationary_occupancy(m, 0), atol=1e-6)


def test_hit_probability_is_monotone_and_at_least_the_marginal():
    m = _model(cap=6, a=2.0, d=10.0)
    hit = hit_probability(m, 2, 0, 12, "empty")
    marg = forecast(m, 2, 0, 12)[:, 0]
    assert hit[0] == 0 and np.all(np.diff(hit) >= -1e-12)
    assert np.all(hit >= marg - 1e-12)


def test_inhomogeneous_forecast_uses_each_hours_rates():
    r_in, r_out = np.zeros(24), np.zeros(24)
    r_out[8] = 30.0  # only the 8 o'clock hour drains the station
    m = StationModel(20, r_in, r_out)
    before = forecast(m, 20, start_min=7 * 60, steps=6)  # 07:00-08:00: nothing happens
    assert before[-1, 20] == pytest.approx(1.0)
    during = forecast(m, 20, start_min=8 * 60, steps=6)  # mean drop 30 bikes, capped at 20
    assert during[-1, 20] < 1e-6


def test_simulation_matches_forecast_mean():
    m = _model(cap=15, a=12.0, d=6.0)
    rng = np.random.default_rng(0)
    ends = [simulate_day(m, 5, rng, start_min=0, minutes=60).bikes[-1] for _ in range(3000)]
    expected = forecast(m, 5, 0, 6) @ np.arange(16)
    assert np.mean(ends) == pytest.approx(expected[-1], abs=0.15)


def test_simulation_counts_turned_away_riders():
    m = StationModel(3, np.zeros(24), np.full(24, 60.0))  # one rider a minute, no returns
    tr = simulate_day(m, 3, np.random.default_rng(1), start_min=0, minutes=60)
    assert tr.bikes[-1] == 0 and tr.missed_pickups > 30 and tr.empty_minutes > 50


def test_model_rejects_bad_rates():
    with pytest.raises(ValueError):
        StationModel(5, np.full(23, 1.0), np.full(24, 1.0))
    with pytest.raises(ValueError):
        StationModel(5, np.full(24, -1.0), np.full(24, 1.0))
