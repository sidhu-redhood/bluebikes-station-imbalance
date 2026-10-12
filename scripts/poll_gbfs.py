"""Poll the Bluebikes GBFS feed and save station snapshots to SQLite.

Usage:
    python scripts/poll_gbfs.py            # loop forever, poll every 5 minutes
    python scripts/poll_gbfs.py --once     # one poll then exit (for cron or GitHub Actions)

Only uses the standard library.
"""
import argparse
import json
import os
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

GBFS_URL = "https://gbfs.bluebikes.com/gbfs/gbfs.json"
DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "bluebikes_gbfs.db")
STATION_REFRESH_SECONDS = 24 * 60 * 60


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {msg}", flush=True)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "cs5100-bluebikes-project"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def find_feed_urls(gbfs_url):
    """Read gbfs.json and return {feed name: url}. Handles v2 and v3 layouts."""
    data = fetch(gbfs_url)["data"]
    feeds = data.get("feeds") or next(iter(data.values()))["feeds"]
    return {f["name"]: f["url"] for f in feeds}


def init_db(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS stations (
            station_id TEXT PRIMARY KEY,
            name TEXT, lat REAL, lon REAL, capacity INTEGER
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            polled_at TEXT,
            station_id TEXT,
            bikes INTEGER,
            ebikes INTEGER,
            docks INTEGER,
            is_installed INTEGER,
            is_renting INTEGER,
            is_returning INTEGER,
            last_reported TEXT,
            PRIMARY KEY (polled_at, station_id)
        );
        """
    )


def save_stations(conn, url):
    stations = fetch(url)["data"]["stations"]
    rows = []
    for s in stations:
        name = s.get("name")
        if isinstance(name, list):  # GBFS v3 uses localized text
            name = name[0].get("text") if name else None
        rows.append((str(s["station_id"]), name, s.get("lat"), s.get("lon"), s.get("capacity")))
    conn.executemany("INSERT OR REPLACE INTO stations VALUES (?,?,?,?,?)", rows)
    conn.commit()
    log(f"saved {len(rows)} stations")


def save_snapshot(conn, url):
    polled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stations = fetch(url)["data"]["stations"]
    rows = []
    for s in stations:
        bikes = s.get("num_bikes_available", s.get("num_vehicles_available"))
        rows.append((
            polled_at,
            str(s["station_id"]),
            bikes,
            s.get("num_ebikes_available"),
            s.get("num_docks_available"),
            s.get("is_installed"),
            s.get("is_renting"),
            s.get("is_returning"),
            str(s.get("last_reported")),
        ))
    conn.executemany("INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    log(f"saved snapshot of {len(rows)} stations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--interval", type=int, default=300, help="seconds between polls")
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--gbfs-url", default=GBFS_URL)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    conn = sqlite3.connect(args.db)
    init_db(conn)

    feeds = None
    last_station_refresh = 0.0

    while True:
        started = time.time()
        try:
            if feeds is None:
                feeds = find_feed_urls(args.gbfs_url)
            if started - last_station_refresh > STATION_REFRESH_SECONDS:
                save_stations(conn, feeds["station_information"])
                last_station_refresh = started
            save_snapshot(conn, feeds["station_status"])
        except Exception as exc:  # keep going on network hiccups
            log(f"poll failed: {exc!r}")
            feeds = None  # rediscover feed URLs next time
            if args.once:
                raise SystemExit(1)
        if args.once:
            break
        time.sleep(max(0, args.interval - (time.time() - started)))

    conn.close()


if __name__ == "__main__":
    main()
