"""Poll the Bluebikes GBFS feeds and store them in SQLite.

How it works
- fetch_feed() downloads one feed. It never touches the database.
- store_feed() parses one feed and writes it in ONE transaction. If anything
  fails, that feed rolls back and a failure row (ok = 0) goes to fetch_log.
  One broken feed never costs us the others, especially station_status.
- Each feed has one parser function. See FEEDS at the bottom of the parsers.
- station_status runs on every poll. The other feeds run about hourly.

Run it (the systemd timer does this every 10 minutes):
    python3 polling/poller.py --db /srv/bluebikes/bluebikes.db

Standard library only. No pip install needed.
"""

import argparse
import hashlib
import http.client
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("poller")

HERE = Path(__file__).resolve().parent
DEFAULT_FEEDS = HERE / "feeds" / "feeds.json"
DEFAULT_DB = Path(os.environ.get("BLUEBIKES_DB", "/srv/bluebikes/bluebikes.db"))

TIMEOUT_SECONDS = 15
USER_AGENT = "bluebikes-imbalance-poller/1.0 (Northeastern CS5100 student project)"
SCHEMA_VERSION = 2
# "Hourly" feeds run when the last good fetch is older than this. It is a bit
# under 60 minutes so timer jitter does not push a feed to every second hour.
HOURLY_SECONDS = 55 * 60


# ----------------------------------------------------------------------
# Fetching. No database code in here.
# ----------------------------------------------------------------------

@dataclass
class FetchResult:
    raw: Optional[bytes]          # response body, None if the fetch failed
    fetched_at: int               # our clock, epoch seconds, taken before the request
    http_status: Optional[int]
    error: Optional[str]


def fetch_feed(name: str, url: str) -> FetchResult:
    """Download one feed. Returns raw bytes, fetch time, HTTP status, error."""
    fetched_at = int(time.time())
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return FetchResult(response.read(), fetched_at, response.status, None)
    except urllib.error.HTTPError as err:   # must come before OSError
        return FetchResult(None, fetched_at, err.code, f"HTTP {err.code}: {err.reason}")
    except (http.client.HTTPException, OSError) as err:
        # OSError covers URLError and timeouts. HTTPException covers cut off bodies.
        return FetchResult(None, fetched_at, None, f"{type(err).__name__}: {err}")


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def to_int(value):
    """Booleans and numbers to int. Anything else becomes NULL."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            return 1 if value.lower() == "true" else 0
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def to_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_text(value):
    return None if value is None else str(value)


def json_or_none(value):
    return None if value is None else json.dumps(value, sort_keys=True)


def canonical(data: dict) -> str:
    """Canonical JSON of the data object. This is what we hash and store."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def items(data: dict, key: str) -> list:
    """Return data[key] as a list of dicts. Missing means empty."""
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"data.{key} is not a list of objects")
    return value


def leftover(obj: dict, known: set) -> Optional[str]:
    """JSON of the fields we did not map, so no value is lost."""
    extra = {k: v for k, v in obj.items() if k not in known}
    return json.dumps(extra, sort_keys=True) if extra else None


# ----------------------------------------------------------------------
# Database helpers
# ----------------------------------------------------------------------

def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database. Foreign keys are per connection, so set them here."""
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"{db_path} does not exist. Build it first: sqlite3 {db_path} < polling/schema.sql"
        )
    # isolation_level=None means we control BEGIN and COMMIT ourselves.
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")   # safe with WAL, fewer disk syncs
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise RuntimeError(f"Database schema version is {version}, expected {SCHEMA_VERSION}")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def log_failure(conn, feed_name, url, fetched_at, http_status, error) -> None:
    """Write an ok = 0 row to fetch_log, in its own transaction."""
    with transaction(conn):
        conn.execute(
            "INSERT INTO fetch_log (feed_name, url, fetched_at, ok, http_status, error) "
            "VALUES (?, ?, ?, 0, ?, ?)",
            (feed_name, url, fetched_at, http_status, error[:500]),
        )


# ----------------------------------------------------------------------
# Parsers. Each one takes (conn, fetch_id, fetched_at, data), writes rows,
# and returns the record count. They run inside store_feed's transaction.
# ----------------------------------------------------------------------

STATUS_FIELDS = {
    "station_id", "last_reported", "num_bikes_available", "num_ebikes_available",
    "num_bikes_disabled", "num_docks_available", "num_docks_disabled",
    "is_installed", "is_renting", "is_returning", "legacy_id",
    "eightd_has_available_keys", "num_scooters_available", "num_scooters_unavailable",
}


def parse_station_status(conn, fetch_id, fetched_at, data):
    rows = []
    for st in items(data, "stations"):
        station_id = st.get("station_id")
        if station_id is None:
            log.warning("station_status row without station_id skipped: %r", st)
            continue
        reported_raw = st.get("last_reported")
        rows.append((
            str(station_id), fetched_at, fetch_id,
            None if reported_raw in (None, 86400) else to_int(reported_raw),
            to_int(reported_raw),
            to_int(st.get("num_bikes_available")),
            to_int(st.get("num_ebikes_available")),
            to_int(st.get("num_bikes_disabled")),
            to_int(st.get("num_docks_available")),
            to_int(st.get("num_docks_disabled")),
            to_int(st.get("is_installed")),
            to_int(st.get("is_renting")),
            to_int(st.get("is_returning")),
            to_text(st.get("legacy_id")),
            to_int(st.get("eightd_has_available_keys")),
            to_int(st.get("num_scooters_available")),
            to_int(st.get("num_scooters_unavailable")),
            leftover(st, STATUS_FIELDS),
        ))
    # OR REPLACE: a repeated station_id in one response must not cost us the poll.
    conn.executemany(
        "INSERT OR REPLACE INTO station_status (station_id, fetched_at, fetch_id, "
        "last_reported, last_reported_raw, num_bikes_available, num_ebikes_available, "
        "num_bikes_disabled, num_docks_available, num_docks_disabled, is_installed, "
        "is_renting, is_returning, legacy_id, eightd_has_available_keys, "
        "num_scooters_available, num_scooters_unavailable, extra_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


STATION_SOURCE_KEYS = {
    "station_id", "name", "short_name", "lat", "lon", "capacity", "region_id",
    "legacy_id", "external_id", "station_type", "rental_uris", "rental_methods",
    "electric_bike_surcharge_waiver", "eightd_has_key_dispenser", "has_kiosk",
    "eightd_station_services",
}
STATION_COLUMNS = [
    "name", "short_name", "lat", "lon", "capacity", "region_id", "legacy_id",
    "external_id", "station_type", "rental_uris_json", "rental_methods_json",
    "electric_bike_surcharge_waiver", "eightd_has_key_dispenser", "has_kiosk",
    "eightd_station_services_json", "extra_json",
]
_station_cols = ["station_id"] + STATION_COLUMNS + ["first_seen", "last_seen"]
UPSERT_STATION = (
    f"INSERT INTO stations ({', '.join(_station_cols)}) "
    f"VALUES ({', '.join('?' * len(_station_cols))}) "
    "ON CONFLICT(station_id) DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in STATION_COLUMNS)
    + ", last_seen = excluded.last_seen"          # first_seen is left alone
)


def parse_station_information(conn, fetch_id, fetched_at, data):
    rows = []
    for st in items(data, "stations"):
        station_id = st.get("station_id")
        if station_id is None:
            log.warning("station_information row without station_id skipped: %r", st)
            continue
        rows.append((
            str(station_id),
            st.get("name"), st.get("short_name"),
            to_float(st.get("lat")), to_float(st.get("lon")),
            to_int(st.get("capacity")),
            to_text(st.get("region_id")),
            to_text(st.get("legacy_id")), to_text(st.get("external_id")),
            st.get("station_type"),
            json_or_none(st.get("rental_uris")),
            json_or_none(st.get("rental_methods")),
            to_int(st.get("electric_bike_surcharge_waiver")),
            to_int(st.get("eightd_has_key_dispenser")),
            to_int(st.get("has_kiosk")),
            json_or_none(st.get("eightd_station_services")),
            leftover(st, STATION_SOURCE_KEYS),
            fetched_at, fetched_at,
        ))
    conn.executemany(UPSERT_STATION, rows)
    return len(rows)


def parse_gbfs(conn, fetch_id, fetched_at, data):
    rows = []
    for language, body in data.items():
        if not isinstance(body, dict):
            continue
        for entry in items(body, "feeds"):
            if "name" in entry and "url" in entry:
                rows.append((language, entry["name"], entry["url"], fetched_at, fetched_at))
    conn.executemany(
        "INSERT INTO gbfs_feeds (language, name, url, first_seen, last_seen) "
        "VALUES (?,?,?,?,?) ON CONFLICT(language, name) DO UPDATE SET "
        "url = excluded.url, last_seen = excluded.last_seen",
        rows,
    )
    return len(rows)


def parse_gbfs_versions(conn, fetch_id, fetched_at, data):
    rows = [
        (str(v["version"]), v.get("url"), fetched_at, fetched_at)
        for v in items(data, "versions") if "version" in v
    ]
    conn.executemany(
        "INSERT INTO gbfs_versions (version, url, first_seen, last_seen) "
        "VALUES (?,?,?,?) ON CONFLICT(version) DO UPDATE SET "
        "url = excluded.url, last_seen = excluded.last_seen",
        rows,
    )
    return len(rows)


def parse_ebikes_at_stations(conn, fetch_id, fetched_at, data):
    rows = []
    for st in items(data, "stations"):
        station_id = st.get("station_id")
        if station_id is None:
            log.warning("ebikes_at_stations row without station_id skipped: %r", st)
            continue
        for index, bike in enumerate(st.get("ebikes") or []):
            rows.append((fetch_id, str(station_id), index, json.dumps(bike, sort_keys=True)))
    conn.executemany(
        "INSERT INTO ebikes_at_stations (fetch_id, station_id, item_index, item_json) "
        "VALUES (?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_free_bike_status(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, i, to_text(b.get("bike_id")), to_float(b.get("lat")), to_float(b.get("lon")),
         to_int(b.get("is_reserved")), to_int(b.get("is_disabled")), json.dumps(b, sort_keys=True))
        for i, b in enumerate(items(data, "bikes"))
    ]
    conn.executemany(
        "INSERT INTO free_bike_status (fetch_id, item_index, bike_id, lat, lon, "
        "is_reserved, is_disabled, item_json) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_system_information(conn, fetch_id, fetched_at, data):
    conn.execute(
        "INSERT INTO system_information (fetch_id, system_id, language, name, short_name, "
        "operator, url, purchase_url, start_date, phone_number, email, timezone, "
        "license_url, item_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (fetch_id, data.get("system_id"), data.get("language"), data.get("name"),
         data.get("short_name"), data.get("operator"), data.get("url"),
         data.get("purchase_url"), data.get("start_date"), data.get("phone_number"),
         data.get("email"), data.get("timezone"), data.get("license_url"),
         json.dumps(data, sort_keys=True)),
    )
    return 1


def parse_system_regions(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, str(r["region_id"]), r.get("name"), json.dumps(r, sort_keys=True))
        for r in items(data, "regions") if "region_id" in r
    ]
    conn.executemany(
        "INSERT INTO system_regions (fetch_id, region_id, name, item_json) VALUES (?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_system_pricing_plans(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, str(p["plan_id"]), p.get("url"), p.get("name"), p.get("currency"),
         to_float(p.get("price")),            # the feed sends a string like "3.00"
         to_int(p.get("is_taxable")),         # the feed sends a JSON boolean
         p.get("description"), json.dumps(p, sort_keys=True))
        for p in items(data, "plans") if "plan_id" in p
    ]
    conn.executemany(
        "INSERT INTO system_pricing_plans (fetch_id, plan_id, url, name, currency, price, "
        "is_taxable, description, item_json) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_system_alerts(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, i, to_text(a.get("alert_id")), a.get("type"),
         json_or_none(a.get("times")), json_or_none(a.get("station_ids")),
         json_or_none(a.get("region_ids")), a.get("url"), a.get("summary"),
         a.get("description"), to_int(a.get("last_updated")), json.dumps(a, sort_keys=True))
        for i, a in enumerate(items(data, "alerts"))
    ]
    conn.executemany(
        "INSERT INTO system_alerts (fetch_id, item_index, alert_id, type, times_json, "
        "station_ids_json, region_ids_json, url, summary, description, last_updated, "
        "item_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_system_hours(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, i, json_or_none(h.get("user_types")), json_or_none(h.get("days")),
         h.get("start_time"), h.get("end_time"), json.dumps(h, sort_keys=True))
        for i, h in enumerate(items(data, "rental_hours"))
    ]
    conn.executemany(
        "INSERT INTO system_hours (fetch_id, item_index, user_types_json, days_json, "
        "start_time, end_time, item_json) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


def parse_system_calendar(conn, fetch_id, fetched_at, data):
    rows = [
        (fetch_id, i, to_int(c.get("start_month")), to_int(c.get("start_day")),
         to_int(c.get("start_year")), to_int(c.get("end_month")), to_int(c.get("end_day")),
         to_int(c.get("end_year")), json.dumps(c, sort_keys=True))
        for i, c in enumerate(items(data, "calendars"))
    ]
    conn.executemany(
        "INSERT INTO system_calendar (fetch_id, item_index, start_month, start_day, "
        "start_year, end_month, end_day, end_year, item_json) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows)


# ----------------------------------------------------------------------
# Feed table
#   snapshot  : new rows on every fetch (they change all the time)
#   upsert    : update one current row per item (stations, gbfs lists)
#   on_change : insert rows only when the payload hash is new
#   interval  : 0 means every poll, otherwise HOURLY_SECONDS
# raw_payloads policy: every changed payload except station_status.
# Every raw_payloads and payload hash lookup filters on feed_name. The hash
# covers the data object only, so two feeds can share one.
# ----------------------------------------------------------------------

@dataclass
class FeedSpec:
    mode: str
    interval: int
    parser: Callable


FEEDS = {
    "station_status":       FeedSpec("snapshot",  0,              parse_station_status),
    "ebikes_at_stations":   FeedSpec("snapshot",  0,              parse_ebikes_at_stations),
    "free_bike_status":     FeedSpec("snapshot",  0,              parse_free_bike_status),
    "station_information":  FeedSpec("upsert",    HOURLY_SECONDS, parse_station_information),
    "gbfs":                 FeedSpec("upsert",    HOURLY_SECONDS, parse_gbfs),
    "gbfs_versions":        FeedSpec("upsert",    HOURLY_SECONDS, parse_gbfs_versions),
    "system_information":   FeedSpec("on_change", HOURLY_SECONDS, parse_system_information),
    "system_regions":       FeedSpec("on_change", HOURLY_SECONDS, parse_system_regions),
    "system_pricing_plans": FeedSpec("on_change", HOURLY_SECONDS, parse_system_pricing_plans),
    "system_alerts":        FeedSpec("on_change", HOURLY_SECONDS, parse_system_alerts),
    "system_hours":         FeedSpec("on_change", HOURLY_SECONDS, parse_system_hours),
    "system_calendar":      FeedSpec("on_change", HOURLY_SECONDS, parse_system_calendar),
}


# ----------------------------------------------------------------------
# Storing one feed
# ----------------------------------------------------------------------

def store_feed(conn, feed_name, url, raw: bytes, fetched_at: int, http_status: int = 200) -> bool:
    """Parse one feed and write it in one transaction. Returns True on success.

    On any failure the feed's rows roll back and an ok = 0 row goes to fetch_log.
    """
    spec = FEEDS[feed_name]
    try:
        doc = json.loads(raw)
        data = doc.get("data") if isinstance(doc, dict) else None
        if not isinstance(data, dict):
            raise ValueError("response has no data object")
        digest = sha256_hex(canonical(data))

        with transaction(conn):
            known = conn.execute(
                "SELECT 1 FROM raw_payloads WHERE feed_name = ? AND payload_sha256 = ?",
                (feed_name, digest),
            ).fetchone() is not None

            cursor = conn.execute(
                "INSERT INTO fetch_log (feed_name, url, fetched_at, feed_last_updated, ttl, "
                "version, ok, http_status, payload_sha256) VALUES (?,?,?,?,?,?,1,?,?)",
                (feed_name, url, fetched_at, to_int(doc.get("last_updated")),
                 to_int(doc.get("ttl")), to_text(doc.get("version")), http_status, digest),
            )
            fetch_id = cursor.lastrowid

            if feed_name != "station_status" and not known:
                conn.execute(
                    "INSERT INTO raw_payloads (payload_sha256, feed_name, first_fetch_id, "
                    "first_seen, payload) VALUES (?,?,?,?,?)",
                    (digest, feed_name, fetch_id, fetched_at, canonical(data)),
                )

            if spec.mode == "on_change" and known:
                previous = conn.execute(
                    "SELECT record_count FROM fetch_log WHERE feed_name = ? "
                    "AND payload_sha256 = ? AND record_count IS NOT NULL "
                    "ORDER BY fetch_id DESC LIMIT 1", (feed_name, digest),
                ).fetchone()
                count = previous[0] if previous else None
            else:
                count = spec.parser(conn, fetch_id, fetched_at, data)

            conn.execute("UPDATE fetch_log SET record_count = ? WHERE fetch_id = ?",
                         (count, fetch_id))
        log.info("%s ok, %s records", feed_name, count)
        return True
    except Exception as err:
        log.error("%s failed: %s: %s", feed_name, type(err).__name__, err)
        log_failure(conn, feed_name, url, fetched_at, http_status, f"{type(err).__name__}: {err}")
        return False


# ----------------------------------------------------------------------
# One poll
# ----------------------------------------------------------------------

def due_feeds(conn, feeds: dict, now: int, force: bool = False) -> list:
    """Which feeds should run now. station_status always goes first."""
    due = []
    for name in sorted(feeds, key=lambda n: (n != "station_status", n)):
        spec = FEEDS.get(name)
        if spec is None:
            log.warning("no parser for feed %s, skipping", name)
            continue
        if not force and spec.interval:
            last = conn.execute(
                "SELECT MAX(fetched_at) FROM fetch_log WHERE feed_name = ? AND ok = 1", (name,)
            ).fetchone()[0]
            if last is not None and now - last < spec.interval:
                continue
        due.append((name, feeds[name]))
    return due


def run_poll(conn, feeds: dict, fetch=fetch_feed, force: bool = False, now: Optional[int] = None):
    """Fetch and store every due feed. Returns (ok_count, failed_count)."""
    now = int(time.time()) if now is None else now
    ok = failed = 0
    for name, url in due_feeds(conn, feeds, now, force):
        result = fetch(name, url)
        if result.raw is None:
            log.error("%s fetch failed: %s", name, result.error)
            log_failure(conn, name, url, result.fetched_at, result.http_status, result.error or "unknown")
            failed += 1
        elif store_feed(conn, name, url, result.raw, result.fetched_at, result.http_status or 200):
            ok += 1
        else:
            failed += 1
    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    parser.add_argument("--force", action="store_true", help="run every feed, ignore the hourly rule")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    feeds = json.loads(args.feeds.read_text())
    conn = connect(args.db)
    try:
        ok, failed = run_poll(conn, feeds, force=args.force)
    finally:
        conn.close()
    log.info("poll done: %d ok, %d failed", ok, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
