"""Offline tests for the Bluebikes poller.

No network: we call the feed writers directly against a temporary SQLite
database created from schema.sql, and we monkeypatch poll.fetch for the
end-to-end poll_once test.

Run:
    python -m pytest polling/test_poll.py -q
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import poll


@pytest.fixture()
def conn(tmp_path):
    db = tmp_path / "test.db"
    c = poll.connect(db)
    yield c
    c.close()


def _station_status_doc(stations, last_updated=1_700_000_000):
    return {"last_updated": last_updated, "ttl": 60, "version": "1.1",
            "data": {"stations": stations}}


def test_schema_applied(conn):
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"fetch_log", "station_status", "stations", "raw_payloads"} <= tables


def test_station_status_typed_write(conn):
    fetched_at = 1_700_000_000
    fetch_id = poll.record_fetch_log(conn, "station_status", "u", fetched_at,
                                     _station_status_doc([]), 1, 200, None, 2, "sha")
    stations = [
        {"station_id": 42, "num_bikes_available": 3, "num_docks_available": 7,
         "is_installed": True, "is_renting": True, "is_returning": True,
         "last_reported": 1_699_999_000, "legacy_id": 42},
        {"station_id": "uuid-9", "num_bikes_available": 0, "num_docks_available": 0,
         "is_installed": 1, "is_renting": 0, "is_returning": 1,
         "last_reported": 86400},  # sentinel
    ]
    n = poll.write_station_status(conn, fetch_id, fetched_at, stations)
    conn.commit()
    assert n == 2

    rows = conn.execute(
        "SELECT * FROM station_status ORDER BY station_id").fetchall()
    assert rows[0]["station_id"] == "42"  # coerced to TEXT
    assert isinstance(rows[0]["station_id"], str)
    assert rows[0]["num_bikes_available"] == 3

    # 86400 sentinel: raw kept, cleaned is NULL.
    sentinel = conn.execute(
        "SELECT last_reported, last_reported_raw FROM station_status "
        "WHERE station_id='uuid-9'").fetchone()
    assert sentinel["last_reported"] is None
    assert sentinel["last_reported_raw"] == 86400


def test_empty_full_view(conn):
    fetched_at = 1_700_000_000
    fid = poll.record_fetch_log(conn, "station_status", "u", fetched_at,
                                _station_status_doc([]), 1, 200, None, 0, "s")
    poll.write_station_status(conn, fid, fetched_at, [
        {"station_id": "empty", "num_bikes_available": 0, "num_docks_available": 5,
         "is_installed": 1, "is_renting": 1, "is_returning": 1},
        {"station_id": "full", "num_bikes_available": 5, "num_docks_available": 0,
         "is_installed": 1, "is_renting": 1, "is_returning": 1},
        {"station_id": "offline-empty", "num_bikes_available": 0,
         "num_docks_available": 5, "is_installed": 0, "is_renting": 0,
         "is_returning": 0},  # offline: not counted empty
    ])
    conn.commit()
    view = {r["station_id"]: (r["is_empty"], r["is_full"])
            for r in conn.execute("SELECT * FROM v_station_status")}
    assert view["empty"] == (1, 0)
    assert view["full"] == (0, 1)
    assert view["offline-empty"] == (0, 0)


def test_station_information_upsert_keeps_first_seen(conn):
    s = {"station_id": "s1", "name": "Ames St", "lat": 42.36, "lon": -71.09,
         "capacity": 19, "legacy_id": 100, "has_kiosk": True}
    poll.write_stations(conn, 1000, [s])
    conn.commit()
    s2 = dict(s, name="Ames St at Main St", capacity=20)
    poll.write_stations(conn, 2000, [s2])
    conn.commit()
    row = conn.execute("SELECT * FROM stations WHERE station_id='s1'").fetchone()
    assert row["name"] == "Ames St at Main St"
    assert row["capacity"] == 20
    assert row["first_seen"] == 1000  # unchanged
    assert row["last_seen"] == 2000   # advanced
    assert row["has_kiosk"] == 1      # bool -> int


def test_as_bool():
    assert poll.as_bool(True) == 1
    assert poll.as_bool(False) == 0
    assert poll.as_bool("true") == 1
    assert poll.as_bool("0") == 0
    assert poll.as_bool(None) is None
    assert poll.as_bool(2) == 1


def test_data_hash_ignores_key_order():
    a = {"stations": [{"x": 1, "y": 2}]}
    b = {"stations": [{"y": 2, "x": 1}]}
    assert poll.data_hash(a) == poll.data_hash(b)


def test_raw_payload_dedup(conn):
    fid = poll.record_fetch_log(conn, "station_information", "u", 1, None, 1,
                                200, None, 0, "abc")
    poll.archive_payload(conn, "station_information", fid, 1, "abc", {"k": 1})
    poll.archive_payload(conn, "station_information", fid, 1, "abc", {"k": 1})
    count = conn.execute("SELECT COUNT(*) c FROM raw_payloads").fetchone()["c"]
    assert count == 1


def test_poll_once_end_to_end(conn, monkeypatch):
    """Drive a whole cycle with two feeds, faking the network."""
    payloads = {
        "https://x/station_status.json": _station_status_doc([
            {"station_id": "s1", "num_bikes_available": 4, "num_docks_available": 6,
             "is_installed": 1, "is_renting": 1, "is_returning": 1},
        ]),
        "https://x/station_information.json": {
            "last_updated": 1, "ttl": 60, "version": "1.1",
            "data": {"stations": [
                {"station_id": "s1", "name": "Test", "lat": 1.0, "lon": 2.0,
                 "capacity": 10}]}},
    }

    def fake_fetch(url):
        return payloads[url], 200

    monkeypatch.setattr(poll, "fetch", fake_fetch)
    feeds = {
        "station_status": "https://x/station_status.json",
        "station_information": "https://x/station_information.json",
    }
    ok = poll.poll_once(conn, feeds)
    assert ok == 2

    # Two fetch_log rows, both ok, sharing one fetched_at.
    logs = conn.execute("SELECT feed_name, ok, fetched_at FROM fetch_log").fetchall()
    assert len(logs) == 2
    assert all(r["ok"] == 1 for r in logs)
    assert len({r["fetched_at"] for r in logs}) == 1

    # Occupancy row landed, linked to its fetch_id.
    ss = conn.execute("SELECT * FROM station_status").fetchone()
    assert ss["num_bikes_available"] == 4
    linked = conn.execute(
        "SELECT ok FROM fetch_log WHERE fetch_id=?", (ss["fetch_id"],)).fetchone()
    assert linked["ok"] == 1


def test_system_regions_keyed_by_region_id(conn):
    """system_regions PK is (fetch_id, region_id), not item_index."""
    doc = {"last_updated": 1, "ttl": 60, "version": "1.1",
           "data": {"regions": [
               {"region_id": "r1", "name": "Boston"},
               {"region_id": "r2", "name": "Cambridge"},
           ]}}
    fid = poll.record_fetch_log(conn, "system_regions", "u", 1, doc, 1, 200,
                                None, 2, "sha")
    poll.finalize_feed_data(conn, "system_regions", fid, 1, doc["data"], "sha")
    conn.commit()
    rows = conn.execute(
        "SELECT region_id, name FROM system_regions ORDER BY region_id").fetchall()
    assert [(r["region_id"], r["name"]) for r in rows] == [
        ("r1", "Boston"), ("r2", "Cambridge")]


def test_poll_feed_records_error_on_failure(conn, monkeypatch):
    def boom(url):
        raise poll.PollError("network down")

    monkeypatch.setattr(poll, "fetch", boom)
    poll.poll_feed(conn, "station_status", "https://x", 1_700_000_000)
    row = conn.execute("SELECT ok, error FROM fetch_log").fetchone()
    assert row["ok"] == 0
    assert "network down" in row["error"]
