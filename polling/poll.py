"""Poll the Bluebikes GBFS feeds once and record them in the SQLite database.

Reads the feed URL map written by fetch_feeds.py (polling/feeds/feeds.json),
fetches every feed, and writes rows into the schema defined in polling/schema.sql.

Design follows the conventions documented in schema.sql:
  - All times are UTC epoch seconds (INTEGER). We never store local time.
  - station_id is always TEXT.
  - Booleans are stored as INTEGER 0 or 1.
  - Every feed attempt gets one fetch_log row, so "nothing happened" is
    distinguishable from "the poller was down".
  - Content is deduplicated by a sha256 over the feed's `data` object only
    (canonical JSON, sorted keys). The top-level last_updated/ttl change on
    every fetch, so hashing the whole response would make every poll look new.
  - station_status is fully typed (the main occupancy table). Other feeds keep
    the full original object in an item_json / extra_json column so no value is
    lost if the feed changes.

One poll:
    python polling/poll.py

Loop forever every 10 minutes (Ctrl+C to stop):
    python polling/poll.py --loop

Use a different database or feeds file:
    python polling/poll.py --db path/to/bluebikes.db --feeds path/to/feeds.json
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FEEDS = HERE / "feeds" / "feeds.json"
DEFAULT_DB = HERE / "bluebikes.db"
SCHEMA_PATH = HERE / "schema.sql"

TIMEOUT_SECONDS = 15
POLL_INTERVAL_SECONDS = 600  # 10 minutes, matching the proposal.

# Feeds whose changed payloads we always archive in raw_payloads. station_status
# is excluded on purpose: its typed table already holds every value.
ARCHIVE_FEEDS = frozenset(
    {
        "station_information",
        "gbfs",
        "gbfs_versions",
        "system_information",
        "system_regions",
        "system_pricing_plans",
        "system_alerts",
        "system_hours",
        "system_calendar",
        "ebikes_at_stations",
        "free_bike_status",
    }
)


class PollError(Exception):
    """Raised for problems that should abort a single poll cycle, not the loop."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def now() -> int:
    """Current time as UTC epoch seconds."""
    return int(time.time())


def as_bool(value) -> int | None:
    """Normalise a GBFS boolean to INTEGER 0/1, or None when absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if value else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes"):
            return 1
        if v in ("0", "false", "no", ""):
            return 0
    return None


def data_hash(data_obj) -> str:
    """sha256 over the feed's `data` object as canonical JSON (sorted keys)."""
    canonical = json.dumps(data_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_last_reported(raw):
    """The feed sends 86400 as a 'never' sentinel. Keep the raw value, and a
    cleaned value that is NULL for the sentinel."""
    if raw is None:
        return None, None
    cleaned = None if raw == 86400 else raw
    return cleaned, raw


# --------------------------------------------------------------------------
# Database setup
# --------------------------------------------------------------------------
def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database, applying the schema if the tables are not there yet.
    foreign_keys is per-connection, so we turn it on every time."""
    first_time = not db_path.exists()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    if first_time or conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fetch_log'"
    ).fetchone() is None:
        apply_schema(conn)
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    if not SCHEMA_PATH.exists():
        raise PollError(f"schema file not found: {SCHEMA_PATH}")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()


def load_feeds(feeds_path: Path) -> dict:
    if not feeds_path.exists():
        raise PollError(
            f"{feeds_path} not found. Run 'python polling/fetch_feeds.py' first."
        )
    try:
        feeds = json.loads(feeds_path.read_text())
    except json.JSONDecodeError as err:
        raise PollError(f"{feeds_path} is not valid JSON: {err}") from err
    if not isinstance(feeds, dict) or not feeds:
        raise PollError(f"{feeds_path} is empty or not an object")
    return feeds


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def fetch(url: str):
    """Download a feed. Returns (parsed_json, http_status). Raises PollError on
    transport or JSON failure, carrying the http status when we have one."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
            body = response.read()
    except urllib.error.HTTPError as err:
        raise PollError(f"HTTP {err.code} for {url}") from err
    except (http.client.HTTPException, OSError) as err:
        raise PollError(f"could not fetch {url}: {err}") from err

    try:
        return json.loads(body), status
    except ValueError as err:
        raise PollError(f"response from {url} is not valid JSON: {err}") from err


# --------------------------------------------------------------------------
# Writers: one per feed shape. Each returns a record count for the fetch_log.
# --------------------------------------------------------------------------
def record_fetch_log(conn, feed_name, url, fetched_at, doc, ok, http_status,
                     error, record_count, payload_sha256) -> int:
    cur = conn.execute(
        """INSERT INTO fetch_log
             (feed_name, url, fetched_at, feed_last_updated, ttl, version,
              ok, http_status, error, record_count, payload_sha256)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            feed_name,
            url,
            fetched_at,
            (doc or {}).get("last_updated"),
            (doc or {}).get("ttl"),
            str((doc or {}).get("version")) if doc and "version" in doc else None,
            ok,
            http_status,
            error,
            record_count,
            payload_sha256,
        ),
    )
    return cur.lastrowid


def archive_payload(conn, feed_name, fetch_id, fetched_at, sha, data_obj) -> None:
    """Store a changed payload once. Deduped by hash across all time."""
    exists = conn.execute(
        "SELECT 1 FROM raw_payloads WHERE payload_sha256 = ?", (sha,)
    ).fetchone()
    if exists:
        return
    conn.execute(
        """INSERT INTO raw_payloads
             (payload_sha256, feed_name, first_fetch_id, first_seen, payload)
           VALUES (?, ?, ?, ?, ?)""",
        (sha, feed_name, fetch_id, fetched_at,
         json.dumps(data_obj, sort_keys=True, separators=(",", ":"))),
    )


def write_station_status(conn, fetch_id, fetched_at, stations) -> int:
    known = {
        "station_id", "last_reported", "num_bikes_available", "num_ebikes_available",
        "num_bikes_disabled", "num_docks_available", "num_docks_disabled",
        "is_installed", "is_renting", "is_returning", "legacy_id",
        "eightd_has_available_keys", "num_scooters_available",
        "num_scooters_unavailable",
    }
    rows = []
    for s in stations:
        if "station_id" not in s:
            continue
        cleaned, raw = normalize_last_reported(s.get("last_reported"))
        extra = {k: v for k, v in s.items() if k not in known}
        rows.append((
            str(s["station_id"]),
            fetched_at,
            fetch_id,
            cleaned,
            raw,
            s.get("num_bikes_available"),
            s.get("num_ebikes_available"),
            s.get("num_bikes_disabled"),
            s.get("num_docks_available"),
            s.get("num_docks_disabled"),
            as_bool(s.get("is_installed")),
            as_bool(s.get("is_renting")),
            as_bool(s.get("is_returning")),
            str(s["legacy_id"]) if s.get("legacy_id") is not None else None,
            as_bool(s.get("eightd_has_available_keys")),
            s.get("num_scooters_available"),
            s.get("num_scooters_unavailable"),
            json.dumps(extra, sort_keys=True) if extra else None,
        ))
    conn.executemany(
        """INSERT OR IGNORE INTO station_status
             (station_id, fetched_at, fetch_id, last_reported, last_reported_raw,
              num_bikes_available, num_ebikes_available, num_bikes_disabled,
              num_docks_available, num_docks_disabled, is_installed, is_renting,
              is_returning, legacy_id, eightd_has_available_keys,
              num_scooters_available, num_scooters_unavailable, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    return len(rows)


def write_stations(conn, fetched_at, stations) -> int:
    """Upsert current station_information. first_seen stays, last_seen advances."""
    known = {
        "station_id", "name", "short_name", "lat", "lon", "capacity", "region_id",
        "legacy_id", "external_id", "station_type", "rental_uris", "rental_methods",
        "electric_bike_surcharge_waiver", "eightd_has_key_dispenser", "has_kiosk",
        "eightd_station_services",
    }
    n = 0
    for s in stations:
        if "station_id" not in s:
            continue
        sid = str(s["station_id"])
        extra = {k: v for k, v in s.items() if k not in known}
        conn.execute(
            """INSERT INTO stations
                 (station_id, name, short_name, lat, lon, capacity, region_id,
                  legacy_id, external_id, station_type, rental_uris_json,
                  rental_methods_json, electric_bike_surcharge_waiver,
                  eightd_has_key_dispenser, has_kiosk, eightd_station_services_json,
                  extra_json, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(station_id) DO UPDATE SET
                  name=excluded.name, short_name=excluded.short_name,
                  lat=excluded.lat, lon=excluded.lon, capacity=excluded.capacity,
                  region_id=excluded.region_id, legacy_id=excluded.legacy_id,
                  external_id=excluded.external_id, station_type=excluded.station_type,
                  rental_uris_json=excluded.rental_uris_json,
                  rental_methods_json=excluded.rental_methods_json,
                  electric_bike_surcharge_waiver=excluded.electric_bike_surcharge_waiver,
                  eightd_has_key_dispenser=excluded.eightd_has_key_dispenser,
                  has_kiosk=excluded.has_kiosk,
                  eightd_station_services_json=excluded.eightd_station_services_json,
                  extra_json=excluded.extra_json, last_seen=excluded.last_seen""",
            (
                sid,
                s.get("name"),
                s.get("short_name"),
                s.get("lat"),
                s.get("lon"),
                s.get("capacity"),
                s.get("region_id"),
                str(s["legacy_id"]) if s.get("legacy_id") is not None else None,
                s.get("external_id"),
                s.get("station_type"),
                json.dumps(s["rental_uris"]) if "rental_uris" in s else None,
                json.dumps(s["rental_methods"]) if "rental_methods" in s else None,
                as_bool(s.get("electric_bike_surcharge_waiver")),
                as_bool(s.get("eightd_has_key_dispenser")),
                as_bool(s.get("has_kiosk")),
                json.dumps(s["eightd_station_services"]) if "eightd_station_services" in s else None,
                json.dumps(extra, sort_keys=True) if extra else None,
                fetched_at,
                fetched_at,
            ),
        )
        n += 1
    return n


def write_generic_items(conn, table, fetch_id, items, columns) -> int:
    """Write a list feed where we keep full objects in item_json plus a few
    typed columns. `columns` maps a db column to a key in each item."""
    rows = []
    for idx, item in enumerate(items):
        values = [fetch_id, idx]
        for _col, key in columns:
            values.append(item.get(key) if isinstance(item, dict) else None)
        values.append(json.dumps(item, sort_keys=True))
        rows.append(tuple(values))
    col_names = ["fetch_id", "item_index"] + [c for c, _ in columns] + ["item_json"]
    placeholders = ", ".join("?" for _ in col_names)
    conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


# --------------------------------------------------------------------------
# One feed, one poll
# --------------------------------------------------------------------------
def poll_feed(conn, feed_name, url, fetched_at) -> None:
    """Fetch and store one feed. Always writes a fetch_log row, even on error."""
    try:
        doc, http_status = fetch(url)
    except PollError as err:
        record_fetch_log(conn, feed_name, url, fetched_at, None, 0, None,
                         str(err), None, None)
        conn.commit()
        print(f"  {feed_name}: ERROR {err}", file=sys.stderr)
        return

    data_obj = doc.get("data") if isinstance(doc, dict) else None
    sha = data_hash(data_obj) if data_obj is not None else None

    record_count = store_feed_data(conn, feed_name, fetched_at, doc, data_obj, sha)

    # fetch_log is written after we know the record count; archiving (which needs
    # the fetch_id) is handled inside store_feed_data for station_status-excluded
    # feeds via a second pass below.
    fetch_id = record_fetch_log(conn, feed_name, url, fetched_at, doc, 1,
                               http_status, None, record_count, sha)

    # Re-run writes that need fetch_id now that we have it.
    finalize_feed_data(conn, feed_name, fetch_id, fetched_at, data_obj, sha)
    conn.commit()
    print(f"  {feed_name}: ok ({record_count if record_count is not None else 0} records)")


def store_feed_data(conn, feed_name, fetched_at, doc, data_obj, sha):
    """Writes that don't need fetch_id. Returns a record count or None."""
    if data_obj is None:
        return None
    if feed_name == "station_information":
        return write_stations(conn, fetched_at, data_obj.get("stations", []))
    if feed_name == "station_status":
        return len(data_obj.get("stations", []))  # real write needs fetch_id
    if feed_name == "gbfs":
        feeds = (data_obj.get("en") or {}).get("feeds", []) if "en" in data_obj else []
        for f in feeds:
            conn.execute(
                """INSERT INTO gbfs_feeds (language, name, url, first_seen, last_seen)
                   VALUES ('en', ?, ?, ?, ?)
                   ON CONFLICT(language, name) DO UPDATE SET
                     url=excluded.url, last_seen=excluded.last_seen""",
                (f.get("name"), f.get("url"), fetched_at, fetched_at),
            )
        return len(feeds)
    if feed_name == "gbfs_versions":
        versions = data_obj.get("versions", [])
        for v in versions:
            conn.execute(
                """INSERT INTO gbfs_versions (version, url, first_seen, last_seen)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(version) DO UPDATE SET
                     url=excluded.url, last_seen=excluded.last_seen""",
                (str(v.get("version")), v.get("url"), fetched_at, fetched_at),
            )
        return len(versions)
    # Everything else is written in finalize (needs fetch_id). Report a count now.
    return _count_items(feed_name, data_obj)


def _count_items(feed_name, data_obj):
    if feed_name == "station_status":
        return len(data_obj.get("stations", []))
    if feed_name == "free_bike_status":
        return len(data_obj.get("bikes", []))
    if feed_name == "ebikes_at_stations":
        return sum(len(s.get("ebikes", [])) for s in data_obj.get("stations", []))
    if feed_name == "system_regions":
        return len(data_obj.get("regions", []))
    if feed_name == "system_pricing_plans":
        return len(data_obj.get("plans", []))
    if feed_name == "system_alerts":
        return len(data_obj.get("alerts", []))
    if feed_name == "system_hours":
        return len(data_obj.get("rental_hours", []))
    if feed_name == "system_calendar":
        return len(data_obj.get("calendars", []))
    if feed_name == "system_information":
        return 1
    return None


def finalize_feed_data(conn, feed_name, fetch_id, fetched_at, data_obj, sha):
    """Writes that need the fetch_id, plus archiving of changed payloads."""
    if data_obj is None:
        return

    if feed_name == "station_status":
        write_station_status(conn, fetch_id, fetched_at, data_obj.get("stations", []))

    elif feed_name == "free_bike_status":
        write_generic_items(
            conn, "free_bike_status", fetch_id, data_obj.get("bikes", []),
            [("bike_id", "bike_id"), ("lat", "lat"), ("lon", "lon"),
             ("is_reserved", "is_reserved"), ("is_disabled", "is_disabled")],
        )

    elif feed_name == "ebikes_at_stations":
        rows = []
        for st in data_obj.get("stations", []):
            sid = str(st.get("station_id"))
            for idx, bike in enumerate(st.get("ebikes", [])):
                rows.append((fetch_id, sid, idx, json.dumps(bike, sort_keys=True)))
        conn.executemany(
            """INSERT OR IGNORE INTO ebikes_at_stations
                 (fetch_id, station_id, item_index, item_json) VALUES (?, ?, ?, ?)""",
            rows,
        )

    elif feed_name == "system_information":
        info = data_obj
        conn.execute(
            """INSERT OR IGNORE INTO system_information
                 (fetch_id, system_id, language, name, short_name, operator, url,
                  purchase_url, start_date, phone_number, email, timezone,
                  license_url, item_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fetch_id, info.get("system_id"), info.get("language"), info.get("name"),
             info.get("short_name"), info.get("operator"), info.get("url"),
             info.get("purchase_url"), info.get("start_date"), info.get("phone_number"),
             info.get("email"), info.get("timezone"), info.get("license_url"),
             json.dumps(info, sort_keys=True)),
        )

    elif feed_name == "system_regions":
        for r in data_obj.get("regions", []):
            conn.execute(
                """INSERT OR IGNORE INTO system_regions
                     (fetch_id, region_id, name, item_json)
                   VALUES (?, ?, ?, ?)""",
                (fetch_id, str(r.get("region_id")), r.get("name"),
                 json.dumps(r, sort_keys=True)),
            )

    elif feed_name == "system_pricing_plans":
        for idx, p in enumerate(data_obj.get("plans", [])):
            conn.execute(
                """INSERT OR IGNORE INTO system_pricing_plans
                     (fetch_id, plan_id, url, name, currency, price, is_taxable,
                      description, item_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fetch_id, str(p.get("plan_id")), p.get("url"), p.get("name"),
                 p.get("currency"), p.get("price"), as_bool(p.get("is_taxable")),
                 p.get("description"), json.dumps(p, sort_keys=True)),
            )

    elif feed_name == "system_alerts":
        for idx, a in enumerate(data_obj.get("alerts", [])):
            conn.execute(
                """INSERT OR IGNORE INTO system_alerts
                     (fetch_id, item_index, alert_id, type, times_json,
                      station_ids_json, region_ids_json, url, summary, description,
                      last_updated, item_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fetch_id, idx, a.get("alert_id"), a.get("type"),
                 json.dumps(a.get("times")) if "times" in a else None,
                 json.dumps(a.get("station_ids")) if "station_ids" in a else None,
                 json.dumps(a.get("region_ids")) if "region_ids" in a else None,
                 a.get("url"), a.get("summary"), a.get("description"),
                 a.get("last_updated"), json.dumps(a, sort_keys=True)),
            )

    elif feed_name == "system_hours":
        write_generic_items(conn, "system_hours", fetch_id,
                            data_obj.get("rental_hours", []),
                            [("start_time", "start_time"), ("end_time", "end_time")])
        _add_hours_json(conn, fetch_id, data_obj.get("rental_hours", []))

    elif feed_name == "system_calendar":
        for idx, c in enumerate(data_obj.get("calendars", [])):
            conn.execute(
                """INSERT OR IGNORE INTO system_calendar
                     (fetch_id, item_index, start_month, start_day, start_year,
                      end_month, end_day, end_year, item_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fetch_id, idx, c.get("start_month"), c.get("start_day"),
                 c.get("start_year"), c.get("end_month"), c.get("end_day"),
                 c.get("end_year"), json.dumps(c, sort_keys=True)),
            )

    # Archive changed payloads for feeds we keep history on.
    if sha is not None and feed_name in ARCHIVE_FEEDS:
        archive_payload(conn, feed_name, fetch_id, fetched_at, sha, data_obj)


def _add_hours_json(conn, fetch_id, hours):
    for idx, h in enumerate(hours):
        conn.execute(
            "UPDATE system_hours SET user_types_json=?, days_json=? "
            "WHERE fetch_id=? AND item_index=?",
            (json.dumps(h.get("user_types")) if "user_types" in h else None,
             json.dumps(h.get("days")) if "days" in h else None,
             fetch_id, idx),
        )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def poll_once(conn, feeds: dict) -> int:
    """Run one poll cycle across all feeds. Returns the count of feeds that
    succeeded. Uses a single fetched_at for the whole cycle."""
    fetched_at = now()
    print(f"poll at {fetched_at} ({len(feeds)} feeds)")
    ok = 0
    for feed_name, url in sorted(feeds.items()):
        try:
            poll_feed(conn, feed_name, url, fetched_at)
            ok += 1
        except Exception as err:  # keep the cycle going even on an unexpected bug
            conn.rollback()
            record_fetch_log(conn, feed_name, url, fetched_at, None, 0, None,
                             f"unexpected: {err}", None, None)
            conn.commit()
            print(f"  {feed_name}: UNEXPECTED {err}", file=sys.stderr)
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Poll Bluebikes GBFS into SQLite.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB,
                        help=f"SQLite database path (default: {DEFAULT_DB})")
    parser.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS,
                        help=f"feeds.json path (default: {DEFAULT_FEEDS})")
    parser.add_argument("--loop", action="store_true",
                        help="poll every 10 minutes until interrupted")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL_SECONDS,
                        help="seconds between polls in --loop mode (default 600)")
    args = parser.parse_args(argv)

    try:
        feeds = load_feeds(args.feeds)
        conn = connect(args.db)
    except PollError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    try:
        if not args.loop:
            poll_once(conn, feeds)
            return 0
        print(f"looping every {args.interval}s. Ctrl+C to stop.")
        while True:
            start = time.monotonic()
            poll_once(conn, feeds)
            elapsed = time.monotonic() - start
            time.sleep(max(0, args.interval - elapsed))
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
