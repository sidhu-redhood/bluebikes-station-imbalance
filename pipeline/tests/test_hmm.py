import itertools

import numpy as np
import pytest

from coolboston.bike_markov import StationModel, hourly_transitions, simulate_day, stationary_occupancy
from coolboston.hmm import corrected_rates, emission_matrix, hourly_empty_full, obs_likelihood, posterior


def make_model(cap=3, a=9.0, d=5.0):
    return StationModel(cap, np.full(24, a), np.full(24, d))


def test_emission_rows_sum_to_one():
    B = emission_matrix(5, 0.1)
    assert np.allclose(B.sum(axis=1), 1)
    assert B[0, 1] == pytest.approx(0.1)
    assert B[2, 1] == pytest.approx(0.05)


def test_posterior_matches_brute_force():
    m = make_model()
    obs = [1, None, 2, 3, None]
    A = hourly_transitions(m)[0]
    pi = stationary_occupancy(m, 0)
    B = emission_matrix(3, 0.2)
    n, T = 4, len(obs)
    total = 0.0
    marg = np.zeros((T, n))
    for path in itertools.product(range(n), repeat=T):
        p = pi[path[0]] * obs_likelihood(obs[0], B)[path[0]]
        for t in range(1, T):
            p *= A[path[t - 1], path[t]] * obs_likelihood(obs[t], B)[path[t]]
        total += p
        marg[np.arange(T), path] += p
    gamma, loglik = posterior(m, obs, 0, eps=0.2)
    assert loglik == pytest.approx(np.log(total))
    assert np.allclose(gamma, marg / total)


def test_hmm_beats_raw_counts():
    rng = np.random.default_rng(0)
    m = make_model(cap=12, a=6.0, d=6.0)
    truth = simulate_day(m, 6, rng).bikes[::10][:144]
    obs = []
    for b in truth:
        r = rng.random()
        if r < 0.15:
            obs.append(None)
        elif r < 0.25:
            obs.append(int(np.clip(b + rng.choice([-1, 1]), 0, 12)))
        else:
            obs.append(int(b))
    gamma, _ = posterior(m, obs, 0, eps=0.1)
    raw = np.array([-1 if o is None else o for o in obs])
    assert (gamma.argmax(axis=1) != truth).sum() < (raw != truth).sum()


def test_censoring_correction():
    gamma = np.zeros((12, 4))
    gamma[:6, 0] = 1.0
    gamma[6:, 1] = 0.5
    gamma[6:, 3] = 0.5
    p_empty, p_full = hourly_empty_full(gamma)
    assert p_empty[0] == 1.0 and p_full[1] == 0.5 and np.isnan(p_empty[2])
    rates_in, rates_out = corrected_rates(make_model(a=4.0, d=4.0), p_empty, p_full)
    assert rates_out[0] == pytest.approx(20.0)
    assert rates_in[1] == pytest.approx(8.0)
    assert rates_out[2] == 4.0


def test_load_day_puts_polls_in_slots(tmp_path):
    import sqlite3
    from datetime import date, datetime
    from pathlib import Path
    from coolboston.hmm import boston_tz, load_day

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript((Path(__file__).resolve().parents[2] / "polling" / "schema.sql").read_text())
    conn.execute("INSERT INTO stations (station_id, short_name, first_seen, last_seen) VALUES ('u1', 'B1', 0, 0)")
    t0 = int(datetime(2026, 11, 3, tzinfo=boston_tz()).timestamp())
    for i, (offset, bikes) in enumerate([(30, 4), (630, 5), (1300, 7), (-60, 9)]):
        conn.execute("INSERT INTO fetch_log (fetch_id, feed_name, fetched_at, ok) VALUES (?, 'station_status', ?, 1)",
                     (i + 1, t0 + offset))
        conn.execute("INSERT INTO station_status (station_id, fetched_at, fetch_id, num_bikes_available) "
                     "VALUES ('u1', ?, ?, ?)", (t0 + offset, i + 1, bikes))
    conn.commit()
    conn.close()
    obs = load_day(db, "B1", date(2026, 11, 3))
    assert len(obs) == 144
    assert obs[:4] == [4, 5, 7, None]
