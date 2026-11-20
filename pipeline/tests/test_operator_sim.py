import numpy as np
import pytest

from coolboston.bike_markov import StationModel
from coolboston.operator_sim import (Forecast, Move, NoOp, Threshold, apply_moves, compare, evaluate, run_day,
                                     summarize)


def drain_and_fill():
    """A drains (pickups >> returns), B fills (returns >> pickups), C is quiet. One van can fix A and B."""
    a = StationModel(15, np.full(24, 2.0), np.full(24, 12.0))
    b = StationModel(15, np.full(24, 12.0), np.full(24, 2.0))
    c = StationModel(15, np.full(24, 4.0), np.full(24, 4.0))
    return [a, b, c], [8, 8, 8]


def test_apply_moves_conserves_bikes_and_clips():
    models, _ = drain_and_fill()
    bikes = [10, 14, 3]
    carried = apply_moves(models, bikes, [Move(0, 1, 9), Move(0, 2, 4)])
    assert sum(bikes) == 27
    assert bikes[1] == 15  # only one dock was free at station 1
    assert carried == 1 + 4
    assert all(0 <= b <= m.capacity for b, m in zip(bikes, models))


def test_noop_moves_nothing_and_counts_are_sane():
    models, b0 = drain_and_fill()
    r = run_day(models, b0, NoOp(), np.random.default_rng(0))
    assert r.moved == 0
    assert r.missed_pickups >= 0 and r.missed_returns >= 0
    assert 0 <= r.empty_hours <= 3 * 24 and 0 <= r.full_hours <= 3 * 24


def test_same_seed_is_reproducible():
    models, b0 = drain_and_fill()
    a = run_day(models, b0, NoOp(), np.random.default_rng(5))
    b = run_day(models, b0, NoOp(), np.random.default_rng(5))
    assert a == b


@pytest.mark.parametrize("make", [lambda ms: Threshold(1, 10), lambda ms: Forecast(ms, 1, 10)])
def test_a_van_reduces_unmet_demand_on_an_imbalanced_pair(make):
    models, b0 = drain_and_fill()
    base = summarize(evaluate(models, b0, NoOp(), 30, seed=1))
    helped = summarize(evaluate(models, b0, make(models), 30, seed=1))
    assert helped["moved"] > 0
    assert helped["unmet"] < base["unmet"]


def test_policies_never_break_capacity():
    models, b0 = drain_and_fill()
    for policy in (Threshold(2, 12), Forecast(models, 2, 12)):
        bikes = list(b0)
        for minute in range(300, 300 + 24 * 60, 60):
            apply_moves(models, bikes, policy.moves(models, bikes, minute))
            assert all(0 <= b <= m.capacity for b, m in zip(bikes, models))


def test_compare_returns_all_policies():
    models, b0 = drain_and_fill()
    rows = compare(models, b0, vans=1, van_capacity=10, days=3)
    assert list(rows) == ["no-op", "threshold", "forecast"]
    assert rows["no-op"]["moved"] == 0
