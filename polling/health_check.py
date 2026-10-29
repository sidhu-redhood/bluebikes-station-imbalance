
"""Check that the Bluebikes poller is healthy.

Reads fetch_log and reports problems. It never writes to the database.

Run it:
    python3 polling/health_check.py --db /srv/bluebikes/bluebikes.db
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("BLUEBIKES_DB", "/srv/bluebikes/bluebikes.db"))

MAX_AGE_SECONDS = 30 * 60
DUPLICATE_SECONDS = 5 * 60
GAP_SECONDS = 15 * 60
WINDOW_SECONDS = 24 * 60 * 60


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open the database read only, so the health check can never change it."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} does not exist")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def last_good_poll(conn, feed_name: str = "station_status"):
    """Time of the last ok fetch of this feed, or None if it never worked."""
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM fetch_log WHERE feed_name = ? AND ok = 1",
        (feed_name,),
    ).fetchone()
    return row[0]



def poll_times(conn, since: int, feed_name: str = "station_status") -> list:
    """Times of ok fetches of this feed since `since`, oldest first."""
    rows = conn.execute(
        "SELECT fetched_at FROM fetch_log WHERE feed_name = ? AND ok = 1 "
        "AND fetched_at >= ? ORDER BY fetched_at",
        (feed_name, since),
    ).fetchall()
    return [row[0] for row in rows]


def find_gaps(times: list, max_gap: int = GAP_SECONDS) -> list:
    """(start, end) pairs where two polls in a row are more than max_gap apart."""
    gaps = []
    for earlier, later in zip(times, times[1:]):
        if later - earlier > max_gap:
            gaps.append((earlier, later))
    return gaps

def find_duplicates(times: list, min_gap: int = DUPLICATE_SECONDS) -> list:
    """(first, second) pairs where two polls in a row are less than min_gap apart."""
    duplicates = []
    for earlier, later in zip(times, times[1:]):
        if later - earlier < min_gap:
            duplicates.append((earlier, later))
    return duplicates

def fmt_time(epoch: int) -> str:
    """Epoch seconds to a readable UTC time, like 2026-10-28 15:00 UTC."""
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(epoch))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    conn = connect_readonly(args.db)
    now = int(time.time())
    last = last_good_poll(conn)
    times = poll_times(conn, since=now - WINDOW_SECONDS)
    conn.close()

    problems = []

    if last is None:
        problems.append("station_status has never been fetched successfully")
    elif now - last > MAX_AGE_SECONDS:
        problems.append(f"last good poll was {(now - last) / 60:.0f} minutes ago")

    gaps = find_gaps(times)
    for start, end in gaps:
        problems.append(f"gap of {(end - start) / 60:.0f} minutes "
                        f"from {fmt_time(start)} to {fmt_time(end)}")
    duplicates = find_duplicates(times)
    for first, second in duplicates:
        problems.append(f"polls only {second - first} seconds apart "
                        f"at {fmt_time(first)} and {fmt_time(second)}")

    print(f"station_status: {len(times)} good polls in the last 24 hours, "
          f"{len(gaps)} gaps, {len(duplicates)} duplicates")
    if last is not None:
        print(f"last good poll: {fmt_time(last)} ({(now - last) / 60:.0f} minutes ago)")

    for problem in problems:
        print(f"PROBLEM: {problem}")
    if not problems:
        print("OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())