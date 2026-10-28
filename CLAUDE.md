# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CS5100 (Foundations of AI, Northeastern) course project: predict where Boston Bluebikes stations run empty or full and compare operator responses (vans, rider incentives, nudges) in a simulator over a cluster of 30–60 stations.

Two hard constraints that shape every design decision:

- **All modeling methods are implemented from scratch.** No scikit-learn / torch / statsmodels for the forecasting, hidden-count estimation, or policy work. Check with the user before reaching for a modeling library.
- **The polling layer is standard library only** (`urllib.request`, `sqlite3`, `json`, `unittest`). There is no `requirements.txt` yet despite the README's setup block, and the poller is meant to run on a small server with no pip install. Keep it that way unless asked.

## Commands

```bash
# Build a NEW polling database. Every statement is CREATE ... IF NOT EXISTS, so running
# this against an EXISTING database changes nothing but the user_version stamp — move an
# existing database forward with the files in polling/migrations instead (poller branch).
sqlite3 bluebikes.db < polling/schema.sql

# Refresh polling/feeds/feeds.json from the GBFS discovery document
python3 polling/fetch_feeds.py

# Run one poll cycle (poller.py currently lives on the `poller` branch)
python3 polling/poller.py --db ./bluebikes.db          # or set BLUEBIKES_DB
python3 polling/poller.py --db ./bluebikes.db --force  # ignore the hourly rule, poll every feed

# Tests (unittest, no pytest; run from the repo root)
python3 -m unittest polling.test_poller -v
python3 -m unittest polling.test_poller.PollerTest.test_station_status -v
```

## Data pipeline

Two stages, deliberately separated:

1. `polling/fetch_feeds.py` reads the discovery document at `gbfs.bluebikes.com/gbfs/gbfs.json`, flattens `data.en.feeds` into a `{name: url}` map, and writes `polling/feeds/feeds.json` atomically (temp file + replace). It refuses to write when `station_information` or `station_status` are missing, and leaves the old file untouched on any error. The URLs it produces point at `gbfs.lyft.com` — that indirection is normal. `feeds.json` is committed, so a run that changes it is a reviewable diff.
2. `polling/poller.py` fetches those 12 feeds into SQLite. `fetch_feed()` never touches the database; `store_feed()` writes one feed in one transaction, so a single malformed feed rolls back on its own and never costs us `station_status`. `station_status` runs every poll; the rest run roughly hourly (`HOURLY_SECONDS = 55 * 60`, trimmed under 60 so timer jitter doesn't skip an hour). Deployment target is a systemd timer.

The database itself is never committed (`data/`, `*.db`, `*.sqlite` are gitignored). Default path is `/srv/bluebikes/bluebikes.db`, overridable by `BLUEBIKES_DB` or `--db`.

## Schema conventions (polling/schema.sql)

These are load-bearing; read the comments in `schema.sql` before changing any table.

- **All times are UTC epoch seconds (INTEGER).** Local time is never stored. Booleans are INTEGER 0/1. `station_id` is always TEXT — the feed mixes UUIDs and long numbers.
- **`fetch_id` is the join key across the whole database**, not `fetched_at`. `station_status` is keyed `(station_id, fetch_id)` because two polls can land in the same second. Every feed-mirroring table carries a `fetch_id` referencing `fetch_log`.
- **`fetch_log` has one row per feed per poll attempt, success or failure.** It is how we distinguish "nothing changed" from "the poller was down" — any analysis of gaps must consult it.
- **Payload hashes cover the `data` object only**, as canonical JSON with sorted keys. The top-level `last_updated`/`ttl` change on every fetch of all 12 feeds, so hashing the whole response would make every fetch look new. `raw_payloads` dedupes on that hash and therefore doubles as the change history for `station_information`.
- **Every mirrored table keeps an `extra_json` / `item_json` column** holding the full original object, so a new or renamed feed field is never lost.
- **`station_status` has no foreign key to `stations` on purpose** — a station can appear in `station_status` before it appears in `station_information`.
- **Empty/full is defined once, in the `v_station_status` view**: empty is `num_bikes_available = 0 AND is_installed AND is_renting`; full is `num_docks_available = 0 AND is_installed AND is_returning`. An offline or non-renting station is neither. Query the view rather than re-deriving this rule.
- `last_reported` is NULL when the feed sends the sentinel `86400`; `last_reported_raw` keeps exactly what the feed sent.
- `foreign_keys` is per-connection in SQLite, so the poller runs `PRAGMA foreign_keys = ON` on every connect (there is a test asserting this).

## Neighborhood tagging

`polling/reference/boston_neighborhoods.geojson` (BPDA boundaries from Analyze Boston, PDDL) is point-in-polygon matched against each station's lon/lat, and the result goes into the separate `station_neighborhoods` table — raw station data is never mutated, so the boundary set can be swapped later. Covers Boston only: stations in Cambridge, Brookline, and Somerville get NULL, which is expected, not a bug. Kenmore is inside `Fenway`. Don't hand-edit the geojson; add a new file beside it and document the source in `polling/reference/README.md`.

## Branch state

`main` holds the schema, the feed-discovery script, and the reference data. Two PRs are open against it:

- **#16 (`poller`)** — the poller implementation and its tests (`polling/poller.py`, `polling/test_poller.py`, `polling/test_fetch_feeds.py`), plus `polling/migrations/` and schema `user_version = 2`.
- **#15 (`Samarth-dev`)** — a teammate's prototype: Next.js map UI under `web/`, Markov model under `pipeline/`, and a *third* poller at `polling/poll.py`. Its layout diverges heavily from `main`; don't assume it applies here.

The team decided to consolidate on `poller.py`: once #15 merges, `polling/poll.py` and `polling/test_poll.py` are removed on the `poller` branch. Do not add a fourth poller, and do not fix bugs in `poll.py` — it is scheduled for deletion.
