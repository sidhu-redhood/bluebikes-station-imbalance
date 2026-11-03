import json

import numpy as np
import pandas as pd
import pytest

from coolboston.build_bikes import (
    STATION_FIELDS, am_net_flow, build_records, demo_doc, hourly_rates, initial_bikes, make_doc,
    select_cluster, validate, write_bikes,
)


def _trips(columns="new"):
    # Mon 2025-09-01 .. Sun 2025-09-07 : 5 weekdays. Station A: 10 pickups at 08:xx each weekday,
    # returned to B. One Saturday trip that must be ignored.
    rows = []
    for day in range(1, 6):
        for k in range(10):
            rows.append((f"2025-09-0{day} 08:{k:02d}:00", f"2025-09-0{day} 08:{k + 20:02d}:00", "A", "B"))
    rows.append(("2025-09-06 12:00:00", "2025-09-06 12:30:00", "A", "B"))
    names = {"new": ["started_at", "ended_at", "start_station_id", "end_station_id"],
             "old": ["starttime", "stoptime", "start station id", "end station id"]}[columns]
    return pd.DataFrame(rows, columns=names)


@pytest.mark.parametrize("columns", ["new", "old"])
def test_hourly_rates_are_weekday_means(columns):
    out, inn, n = hourly_rates(_trips(columns))
    assert n == 5
    assert out["A"][8] == pytest.approx(10.0) and out["A"].sum() == pytest.approx(10.0)
    assert inn["B"][8] == pytest.approx(10.0) and "B" not in out


def test_initial_bikes_starts_draining_stations_fuller():
    r = np.zeros(24)
    drain = r.copy(); drain[7:9] = 6
    assert initial_bikes(20, r, drain) > initial_bikes(20, r, r) == 10 > initial_bikes(20, drain, r)
    assert 1 <= initial_bikes(20, r, np.full(24, 99)) <= 19


def test_select_cluster_filters_by_distance_and_orders_by_volume():
    st = [
        {"_key": "near_busy", "lon": -71.06, "lat": 42.356, "capacity": 15},
        {"_key": "near_quiet", "lon": -71.061, "lat": 42.355, "capacity": 15},
        {"_key": "far", "lon": -71.20, "lat": 42.40, "capacity": 15},
        {"_key": "no_cap", "lon": -71.06, "lat": 42.355, "capacity": 0},
    ]
    vol = {"near_busy": 100, "near_quiet": 5, "far": 999, "no_cap": 50}
    assert [s["_key"] for s in select_cluster(st, vol, size=5)] == ["near_busy", "near_quiet"]


def test_records_round_trip_and_validate(tmp_path):
    st = [{"_key": "A", "station_id": "uuid-a", "name": "A St", "lat": 42.35, "lon": -71.06, "capacity": 12}]
    out, inn, _ = hourly_rates(_trips())
    doc = make_doc(build_records(st, out, inn, status={"uuid-a": 40}), {"kind": "test", "month": "202509", "weekdays": 5})
    rec = doc["stations"][0]
    assert all(k in rec for k in STATION_FIELDS) and rec["initialBikes"] == 12  # clipped to capacity
    path = write_bikes(doc, tmp_path / "bikes.json")
    assert json.loads(path.read_text())["stations"][0]["ratesOut"][8] == 10.0


def test_validate_rejects_bad_station():
    doc = demo_doc()
    doc["stations"][0]["ratesIn"] = [1.0] * 23
    with pytest.raises(ValueError, match="ratesIn"):
        validate(doc)


def test_am_net_flow_sign_and_tagging():
    """Morning destinations (more returns than pickups 6-10am) get a positive net flow;
    residential origins get a negative one. build_records tags isAmDestination accordingly."""
    dest_in = np.zeros(24); dest_in[6:10] = 8     # bikes arrive in the morning
    dest_out = np.zeros(24); dest_out[6:10] = 2
    orig_in = np.zeros(24); orig_in[6:10] = 2
    orig_out = np.zeros(24); orig_out[6:10] = 8    # bikes leave in the morning
    assert am_net_flow(dest_in, dest_out) > 0
    assert am_net_flow(orig_in, orig_out) < 0

    st = [{"_key": "D", "station_id": "d", "name": "Dest", "lat": 42.34, "lon": -71.09, "capacity": 20}]
    out = {"D": dest_out}
    inn = {"D": dest_in}
    rec = build_records(st, out, inn)[0]
    assert rec["isAmDestination"] is True
    assert rec["amNetFlow"] > 0


def test_demo_doc_is_valid_and_has_commute_imbalance():
    doc = demo_doc()
    by = {s["name"]: s for s in doc["stations"]}
    south = by["South Station - 700 Atlantic Ave"]
    home = by["Charles St at Beacon St"]
    am = slice(7, 10)
    assert sum(south["ratesIn"][am]) > sum(south["ratesOut"][am])  # downtown fills in the morning
    assert sum(home["ratesOut"][am]) > sum(home["ratesIn"][am])    # residential drains
