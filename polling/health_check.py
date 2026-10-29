
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

# The poller runs every 10 minutes. If the last good poll is older than this,
# something is wrong.
MAX_AGE_SECONDS = 30 * 60


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    conn = connect_readonly(args.db)
    now = int(time.time())
    last = last_good_poll(conn)
    conn.close()

    if last is None:
        print("PROBLEM: station_status has never been fetched successfully")
        return 1

    age_min = (now - last) / 60
    if now - last > MAX_AGE_SECONDS:
        print(f"PROBLEM: last good poll was {age_min:.0f} minutes ago")
        return 1

    print(f"OK: last good poll was {age_min:.0f} minutes ago")
    return 0


if __name__ == "__main__":
    sys.exit(main())