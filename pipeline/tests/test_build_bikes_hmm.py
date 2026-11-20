import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from coolboston.bike_markov import StationModel, simulate_day, stationary_occupancy
from coolboston.build_bikes_hmm import correct_doc, main
from coolboston.hmm import boston_tz

SCHEMA = Path(__file__).resolve().parents[2] / "polling" / "schema.sql"
DAY0 = date(2026, 11, 3)


def true_model():
    # Heavy morning pull-out: the station sits empty a lot, so trips under-count real demand.
    out = np.full(24, 3.0)
    out[7:10] = 16.0
    return StationModel(10, np.full(24, 5.0), out)


def censored_rates(m):
    """What a trip log would show: pickups only happen while a bike is there (stationary approximation)."""
    p_empty = np.array([stationary_occupancy(m, h)[0] for h in range(24)])
    return m.rates_in, m.rates_out * (1 - p_empty)


def make_db(path, station_id, truths_by_day):
    """truths_by_day: list of 144-long bike counts (one per 10-minute poll) per day."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA.read_text())
    conn.execute("INSERT INTO stations (station_id, name, short_name, capacity, first_seen, last_seen) VALUES (?,?,?,?,0,0)",
                 ("uuid-1", "Test St", station_id, 10))
    fetch_id = 0
    for d, counts in enumerate(truths_by_day):
        day = DAY0 + timedelta(days=d)
        start = int(datetime(day.year, day.month, day.day, tzinfo=boston_tz()).timestamp())
        for slot, bikes in enumerate(counts):
            fetch_id += 1
            t = start + slot * 600 + 30
            conn.execute("INSERT INTO fetch_log (fetch_id, feed_name, fetched_at, ok) VALUES (?,?,?,1)",
                         (fetch_id, "station_status", t))
            conn.execute("INSERT INTO station_status (station_id, fetched_at, fetch_id, num_bikes_available) VALUES (?,?,?,?)",
                         ("uuid-1", t, fetch_id, int(bikes)))
    conn.commit()
    conn.close()


def doc_for(station_id, rin, rout, capacity=10):
    return {"version": 1, "stepMinutes": 10, "startMinute": 300,
            "source": {"kind": "test", "month": None, "weekdays": None, "generated": "2026-11-19"},
            "stations": [{"id": station_id, "name": "Test St", "lat": 42.3, "lon": -71.1, "capacity": capacity,
                          "initialBikes": 5, "ratesIn": [float(x) for x in rin], "ratesOut": [float(x) for x in rout]},
                         {"id": "NOPOLL", "name": "No data", "lat": 42.3, "lon": -71.1, "capacity": 10,
                          "initialBikes": 5, "ratesIn": [1.0] * 24, "ratesOut": [1.0] * 24}]}


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(7)
    truth = true_model()
    days = [simulate_day(truth, 5, rng).bikes[::10][:144] for _ in range(6)]
    db = tmp_path_factory.mktemp("poll") / "poll.db"
    make_db(db, "B1", days)
    return truth, db


def test_correction_moves_censored_rates_toward_truth(synthetic):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    doc = doc_for("B1", rin, rout)
    new, report = correct_doc(doc, db, DAY0, 6)
    got = np.array(new["stations"][0]["ratesOut"])
    morning = slice(7, 10)
    err_before = np.abs(rout[morning] - truth.rates_out[morning]).mean()
    err_after = np.abs(got[morning] - truth.rates_out[morning]).mean()
    assert err_after < err_before
    assert got[morning].mean() > rout[morning].mean()
    assert report[0]["corrected"] and report[0]["days"] == 6


def test_station_without_poller_data_is_untouched_and_input_not_mutated(synthetic):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    doc = doc_for("B1", rin, rout)
    snapshot = json.dumps(doc)
    new, report = correct_doc(doc, db, DAY0, 6)
    assert json.dumps(doc) == snapshot
    assert new["stations"][1]["ratesOut"] == [1.0] * 24
    assert [r["corrected"] for r in report] == [True, False]
    assert new["source"]["kind"] == "test+hmm"
    assert new["source"]["hmm"]["stationsCorrected"] == 1


def test_min_days_keeps_original_rates(synthetic):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    new, report = correct_doc(doc_for("B1", rin, rout), db, DAY0, 6, min_days=7)
    assert not report[0]["corrected"]
    assert new["stations"][0]["ratesOut"] == [float(x) for x in rout]


def test_cli_writes_valid_json_and_csv(synthetic, tmp_path):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    src, out, csv_path = tmp_path / "in.json", tmp_path / "out.json", tmp_path / "r.csv"
    src.write_text(json.dumps(doc_for("B1", rin, rout)))
    code = main(["--db", str(db), "--start", DAY0.isoformat(), "--days", "6", "--in", str(src),
                 "--out", str(out), "--report", str(csv_path)])
    assert code == 0
    doc = json.loads(out.read_text())
    assert len(doc["stations"]) == 2 and csv_path.read_text().startswith("id,name,days")


def test_cli_refuses_to_write_when_no_data(synthetic, tmp_path):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    src, out = tmp_path / "in.json", tmp_path / "out.json"
    src.write_text(json.dumps(doc_for("B1", rin, rout)))
    code = main(["--db", str(db), "--start", "2026-01-01", "--days", "3", "--in", str(src), "--out", str(out)])
    assert code == 1 and not out.exists()


def test_fit_mode_learns_scale_and_raises_morning_rates(synthetic):
    truth, db = synthetic
    rin, rout = censored_rates(truth)
    new, report = correct_doc(doc_for("B1", rin, rout), db, DAY0, 6, fit=True)
    assert new["source"]["hmm"]["method"] == "mle"
    assert report[0]["corrected"] and report[0]["scale_out"] > 0
    got = np.array(new["stations"][0]["ratesOut"])
    assert np.all(np.isfinite(got)) and got.shape == (24,)
    # trip rates under-count the heavy morning pull-out; the learned level moves them toward the truth
    assert abs(got[7:10].mean() - truth.rates_out[7:10].mean()) < abs(rout[7:10].mean() - truth.rates_out[7:10].mean())
