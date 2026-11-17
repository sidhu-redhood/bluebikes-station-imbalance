import itertools
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.hmm_data import build_slots, load_polls
from coolboston.hmm_sim import simulate_network
from coolboston.station_hmm import (
    HMMConfig, HMMParams, _Prep, count_mean, count_quantiles, fit, initial_params,
)

START = pd.Timestamp("2026-08-03")  # a Monday
SCHEMA = Path(__file__).resolve().parents[2] / "polling" / "schema.sql"


def _hand_example():
    # One station, 5 docks, 3 slots. Slot 0: pickup at 1, pickup at 2, return at 3. Slot 1: return at 15.
    events = [(np.array([1.0, 2.0, 3.0, 15.0]), np.array([-1, -1, 1, 1]))]
    polls = [(np.array([2.5, 15.0]), np.array([3, 4]))]
    return build_slots(events, [5], START, 3, polls=polls)


def test_slot_summaries_on_a_hand_example():
    d = _hand_example()
    assert d.net[:, 0].tolist() == [-1, 1, 0]
    assert d.lo[:, 0].tolist() == [-2, 0, 0] and d.hi[:, 0].tolist() == [0, 1, 0]
    assert d.dep[:, 0].tolist() == [2, 0, 0] and d.arr[:, 0].tolist() == [1, 1, 0]
    # Segments cover every slot exactly once.
    per_slot = np.bincount(d.seg_t, weights=d.seg_min, minlength=3)
    assert np.allclose(per_slot, 10.0)
    seg0 = sorted(zip(d.seg_level[d.seg_t == 0], d.seg_min[d.seg_t == 0]))
    assert seg0 == [(-2, 1.0), (-1, 1.0), (-1, 7.0), (0, 1.0)]
    # Poll at 2.5 min comes after two pickups; the poll at 15.0 does not include the return at 15.0.
    assert d.poll_t.tolist() == [0, 1] and d.poll_offset.tolist() == [-2, 0]
    assert d.hour.tolist() == [0, 0, 0] and not d.weekend.any()


def test_minutes_with_a_bike_and_with_a_dock():
    prep = _Prep(_hand_example(), HMMConfig())
    # Slot 0 path is b', b'-1, b'-2, b'-1 for 1, 1, 1 and 7 minutes.
    assert prep.A[0, 0, :5].tolist() == [0.0, 1.0, 9.0, 10.0, 10.0]
    assert prep.B[0, 0, 4] == 10.0 and prep.B[0, 0, 5] == 9.0
    # Feasibility: two pickups in a row need at least 2 bikes at the start.
    assert prep.viol[0, 0, :4].tolist() == [2.0, 1.0, 0.0, 0.0]


def _brute_force(prep, params, cfg, s):
    """Enumerate every start count and every move in every slot."""
    C, T = int(prep.C[s]), prep.T
    F = [np.exp(prep.log_factor(t, params, cfg)[s, :C + 1]) for t in range(T)]
    moves = [("stay", None), ("up", None), ("down", None)] + [("van", L) for L in range(C + 1)]
    Z = 0.0
    m_b, m_after = np.zeros((T, C + 1)), np.zeros((T, C + 1))
    pvan, added, removed = np.zeros(T), np.zeros(T), np.zeros(T)
    for b0 in range(C + 1):
        for path in itertools.product(moves, repeat=T):
            w, b, rec = 1.0 / (C + 1), b0, []
            for t, (kind, L) in enumerate(path):
                p, u = params.p[s, prep.hour[t]], params.u[s]
                if kind == "stay":
                    w, nb = w * (1 - p - u), b
                elif kind == "up":
                    w, nb = w * u / 2, min(b + 1, C)
                elif kind == "down":
                    w, nb = w * u / 2, max(b - 1, 0)
                else:
                    w, nb = w * p * params.g[s, L], L
                w *= F[t][nb]
                rec.append((b, nb, kind == "van"))
                b = int(np.clip(nb + prep.net[t, s], 0, C))
            Z += w
            for t, (bb, nb, v) in enumerate(rec):
                m_b[t, bb] += w
                m_after[t, nb] += w
                pvan[t] += w * v
                added[t] += w * max(nb - bb, 0)
                removed[t] += w * max(bb - nb, 0)
    return Z, m_b / Z, m_after / Z, pvan / Z, added / Z, removed / Z


def test_forward_backward_matches_brute_force_enumeration():
    rng = np.random.default_rng(3)
    events = [(np.array([2.0, 4.0, 11.0, 23.0, 27.0]), np.array([-1, 1, 1, -1, -1])),
              (np.array([5.0, 6.0, 18.0]), np.array([1, 1, -1]))]
    polls = [(np.array([12.0]), np.array([2])), (np.array([3.0, 25.0]), np.array([0, 3]))]
    data = build_slots(events, [2, 3], START + pd.Timedelta(hours=7), 3, polls=polls)
    cfg = HMMConfig()
    params = initial_params(data)
    N = params.g.shape[1]
    params = HMMParams(p=rng.uniform(0.05, 0.3, (2, 24)), g=params.g, u=np.array([0.1, 0.2]),
                       lam=rng.uniform(1, 9, (2, 2, 24)), mu=rng.uniform(1, 9, (2, 2, 24)),
                       day_scale=np.array([0.7]))
    g = rng.uniform(0.2, 1.0, (2, N)) * (np.arange(N)[None, :] <= np.array([[2], [3]]))
    params.g = g / g.sum(axis=1, keepdims=True)
    post = fit(data, cfg, init=params, learn=False).posterior
    prep = _Prep(data, cfg)
    total = 0.0
    for s in range(2):
        Z, m_b, m_after, pvan, added, removed = _brute_force(prep, params, cfg, s)
        C = int(data.capacity[s])
        total += np.log(Z)
        assert np.allclose(post.gamma[:, s, :C + 1], m_b, atol=1e-6)
        assert np.allclose(post.gamma_after[:, s, :C + 1], m_after, atol=1e-6)
        assert np.allclose(post.p_visit[:, s], pvan, atol=1e-9)
        assert np.allclose(post.added[:, s], added, atol=1e-9)
        assert np.allclose(post.removed[:, s], removed, atol=1e-9)
        empty = (m_after * (10.0 - prep.A[:, s, :C + 1])).sum(axis=1)
        assert np.allclose(post.empty_min[:, s], empty, atol=1e-6)
    assert np.isclose(post.loglik, total, rtol=1e-10)


def _sim(days=5, polls=True, seed=0, **kw):
    S = 3
    cap = np.array([15, 19, 23])
    h = np.arange(24)
    base = np.where((h >= 7) & (h <= 19), 6.0, 1.0)
    lam = np.stack([np.stack([base * f, base * f]) for f in (1.3, 0.8, 1.0)])
    mu = np.stack([np.stack([base * f, base * f]) for f in (0.8, 1.3, 1.0)])
    return simulate_network(cap, lam, mu, START, 144 * days, seed=seed,
                            poll_every=10.0 if polls else None, **kw)


def test_em_objective_never_decreases():
    data, _ = _sim(days=3, polls=False, seed=4)
    res = fit(data, HMMConfig(max_iter=15))
    h = np.array(res.history)
    assert len(h) >= 5 and np.all(np.diff(h) >= -1e-9 * abs(h[-1]))


def test_polls_pin_the_count_down():
    data, truth = _sim(days=4, polls=True, seed=1)
    post = fit(data, HMMConfig(max_iter=20)).posterior
    assert np.abs(count_mean(post.gamma) - truth.count).mean() < 0.2


def test_trips_alone_give_calibrated_counts():
    # Station 0 drains (pickups 1.3x returns) and runs empty often; station 2 is balanced and rarely
    # touches 0 or capacity, so trips alone cannot pin its level down. The model should say so.
    data, truth = _sim(days=7, polls=False, seed=2)
    post = fit(data, HMMConfig(max_iter=30)).posterior
    mean = count_mean(post.gamma)
    lo, hi = count_quantiles(post.gamma, (0.1, 0.9))
    cover = ((truth.count >= lo) & (truth.count <= hi)).mean(axis=0)
    width = (hi - lo).mean(axis=0)
    assert np.all((cover >= 0.75) & (cover <= 0.99))
    assert width[0] < 2 < width[2]
    assert np.abs(mean[:, 0] - truth.count[:, 0]).mean() < 1.5
    assert abs(post.empty_min[:, 0].sum() - truth.empty_min[:, 0].sum()) < 0.15 * truth.empty_min[:, 0].sum()


def test_demand_correction_reduces_censoring_bias():
    data, truth = _sim(days=10, polls=False, seed=5)
    res = fit(data, HMMConfig(max_iter=30))
    wk = ~data.weekend
    days = wk.sum() / 144
    raw = data.dep[wk].sum(axis=0) / days
    true = truth.demand(False, data)[0].sum(axis=1)
    est = res.params.demand(data)[0][:, 0, :].sum(axis=1)
    s = int(np.argmax(true - raw))          # the station that runs empty most
    assert true[s] - raw[s] > 5             # the setup really is censored
    assert abs(est[s] - true[s]) < 0.5 * abs(raw[s] - true[s])


def test_poll_loader_joins_short_name_and_converts_to_local_time():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "b.db"
        conn = sqlite3.connect(db)
        conn.executescript(SCHEMA.read_text())
        conn.executemany("INSERT INTO fetch_log (fetch_id, feed_name, fetched_at, ok) VALUES (?, 'station_status', 0, 1)",
                         [(i,) for i in range(1, 5)])
        conn.execute("INSERT INTO stations (station_id, short_name, name, capacity, first_seen, last_seen) "
                     "VALUES ('gbfs-1', 'B32062', 'Forsyth', 15, 0, 0), ('gbfs-2', 'Z99999', 'Far', 10, 0, 0)")
        t = int(pd.Timestamp("2026-08-03 16:00", tz="UTC").timestamp())  # 12:00 in Boston (EDT)
        cols = "(station_id, fetched_at, fetch_id, last_reported, num_bikes_available, is_installed, is_renting)"
        conn.executemany(f"INSERT INTO station_status {cols} VALUES (?, ?, ?, ?, ?, ?, ?)", [
            ("gbfs-1", t + 30, 1, t, 7, 1, 1),
            ("gbfs-1", t + 630, 2, t, 7, 1, 1),            # same report seen twice: kept once
            ("gbfs-1", t + 1230, 3, None, 6, 1, 1),        # no last_reported: use fetch time
            ("gbfs-1", t + 1830, 4, t + 1800, 9, 1, 0),    # not renting: dropped
            ("gbfs-2", t + 30, 1, t, 3, 1, 1),             # not one of our stations
        ])
        conn.commit()
        conn.close()
        p = load_polls(db, ["B32062"], pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01"))
    assert p["station_id"].tolist() == ["B32062", "B32062"]
    assert p["ts"].tolist() == [pd.Timestamp("2026-08-03 12:00"), pd.Timestamp("2026-08-03 12:20:30")]
    assert p["bikes"].tolist() == [7, 6]


def test_day_multipliers_track_busy_and_quiet_days():
    # Demand swings 40% from day to day for every station at once (weather). The model should find the
    # swings, and should not read a quiet day's missing riders as empty stations.
    data, truth = _sim(days=10, polls=False, seed=6, day_noise=0.4)
    on = fit(data, HMMConfig(max_iter=30))
    off = fit(data, HMMConfig(max_iter=30, prior_day=1e9))          # multipliers pinned at 1
    est = on.params.day_scale / on.params.day_scale.mean()
    true = truth.day_mult / truth.day_mult.mean()
    assert np.corrcoef(est, true)[0, 1] > 0.9
    err_on = abs(on.posterior.empty_min.sum() - truth.empty_min.sum())
    err_off = abs(off.posterior.empty_min.sum() - truth.empty_min.sum())
    assert err_on < err_off
