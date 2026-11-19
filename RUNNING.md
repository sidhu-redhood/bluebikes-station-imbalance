# Code guide and how to run it

CS5100 Foundations of AI, Northeastern. Forecast where Bluebikes stations run empty or full around Northeastern, then compare operator responses in a simulator. This file explains what each part of the repo does and how to run it. Commands are for macOS/Linux (bash/zsh); Windows equivalents are in the last section.

## Status

| Step | State |
|---|---|
| 1. Load data and build the NEU station cluster | Done |
| 2. Markov model of bikes at each station | Done |
| 3. HMM to correct demand censoring | Not started |
| 4. Simulator comparing vans, incentives, nudges | Not started |

## How the pieces fit together

```
Bluebikes trip CSV + station list
        |  code/load_data.py                 (step 1: NEU cluster, dock capacity)
        v
pipeline/src/coolboston/build_bikes_saad.py  (hourly weekday rates per station)
        |
        v
web/data/bikes.json  <------ the contract between everything below
        |
        +--> pipeline/src/coolboston/bike_markov.py   (the model, Python)
        +--> web/lib/markov.ts                        (same model, TypeScript)
        +--> demo/bikes/markov.js                     (same model, JavaScript)
                 |                    |
                 v                    v
        web app /bikes (map)    demo/bikes/dist/*.html (one offline file)

polling/   live GBFS poller -> SQLite; separate from the model, not used by it yet
```

## What the model does

Each station is a birth-death Markov chain. The state is the number of bikes docked, 0 to capacity (0 is empty, capacity is full). Within each clock hour, a bike arrives at rate `ratesIn[h]` and leaves at rate `ratesOut[h]` (trips per hour). Arrivals at a full station and departures from an empty one are blocked, and those blocked riders are the unmet demand.

- `transition` turns the rates into a 10-minute transition matrix by uniformization (a matrix exponential using only numpy).
- `forecast` chains those matrices to give the distribution of bike counts over the next few polls. The matrix changes each hour because the rates do.
- `hit_probability` makes empty (or full) an absorbing state, which gives the probability the station hits it at any moment, not just at poll times.
- `simulate_day` runs the same chain event by event (Gillespie) and counts empty minutes, full minutes and turned-away riders. It is the ground truth the forecasts are checked against.

Known limits: rates are censored (an empty station records no pickups, so demand is under-counted), stations are modelled independently, and starting bike counts are an assumption unless live data is used.

## Repo map

| Path | What it is |
|---|---|
| `code/load_data.py` | Step 1. Loads trips, station list and weather; builds the NEU cluster. Config (month, radius, paths) is at the top. |
| `code/load_data_helper.py` | Quick check that prints the cluster. Needs no weather file. |
| `pipeline/src/coolboston/bike_markov.py` | The model. |
| `pipeline/src/coolboston/build_bikes.py` | Builds `bikes.json` from a month of trips, or from demo profiles. |
| `pipeline/src/coolboston/build_bikes_saad.py` | Builds `bikes.json` for the NEU cluster using `code/load_data.py`. This produced the committed file. |
| `pipeline/src/coolboston/neighborhoods.py` | Tags stations with a Boston neighborhood. |
| `pipeline/tests/` | Python tests for the model and builders. |
| `web/` | Next.js app. `/bikes` draws the map and runs the model in the browser. |
| `web/lib/markov.ts`, `web/lib/bikes.ts` | TypeScript model, and the `bikes.json` validator. |
| `demo/bikes/` | Bundles `bikes.json` and `markov.js` into one HTML file. |
| `polling/` | Polls the live GBFS feeds into SQLite, plus a health check and tests. |
| `Notebooks/` | Exploratory preprocessing notebook (same logic as `load_data.py`). |
| `BIKES.md` | Model description and caveats. |
| `tasks/todo-bike-markov.md` | Task plan for the Markov model. |

`bikes.json` (schema v1) holds `stepMinutes` (10), `startMinute` (300), and per station `ratesIn[24]`, `ratesOut[24]`, `capacity`, `initialBikes`. The committed file is the NEU cluster: 42 stations, August 2026, 22 weekdays.

**Parity rule:** `bike_markov.py`, `web/lib/markov.ts` and `demo/bikes/markov.js` must give the same numbers. The shared fixture is `web/lib/__fixtures__/markov-parity.json`. If you change one, update the others and run both test suites.

## Requirements

- Python 3.11 or newer
- `pip install -r requirements.txt` (pandas, numpy), plus `pip install pytest` to run the tests
- Node.js 18 or newer and npm, for the web app only
- A Mapbox public token (starts with `pk.`), for the web app map only. Get one at https://account.mapbox.com/access-tokens/
- Raw data files in `data/raw/`, only if you rebuild `bikes.json` from real trips (see below)

Set up Python once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
export PYTHONPATH=$PWD/pipeline/src     # coolboston has no installer; run in each new terminal
```

## 1. Run the tests (no data, no network)

```bash
python -m pytest pipeline -q            # model and builders
python -m pytest polling -q             # poller
cd web && npm install && npm test       # Markov parity with Python (vitest)
```

Run a single test:

```bash
python -m pytest pipeline/tests/test_bike_markov.py::<test_name> -q
cd web && npx vitest run lib/markov.test.ts -t "<name>"
```

## 2. Use the model from Python (no data, no network)

The committed `web/data/bikes.json` is enough:

```python
import json, numpy as np
from coolboston.bike_markov import StationModel, forecast, hit_probability, simulate_day

with open("web/data/bikes.json") as f:
    doc = json.load(f)
s = doc["stations"][0]
m = StationModel(s["capacity"], np.array(s["ratesIn"]), np.array(s["ratesOut"]))

b0 = s["initialBikes"]
dist = forecast(m, b0, start_min=8 * 60, steps=12)        # (13, capacity+1): next 2 hours, every 10 min
p_empty = hit_probability(m, b0, 8 * 60, 12, "empty")[-1] # P(empty at some point in the next 2 hours)
day = simulate_day(m, b0, np.random.default_rng(0), start_min=300)
print(p_empty, day.empty_minutes, day.missed_pickups)
```

## 3. Rebuild `bikes.json` from data

Rebuilding the NEU cluster needs the raw files in `data/raw/` (the folder is gitignored; create it with `mkdir -p data/raw`):

| File | Where to get it | Save as |
|---|---|---|
| Trip history, one month | https://bluebikes.com/system-data, `YYYYMM-bluebikes-tripdata.zip`, unzip | `data/raw/202608-bluebikes-tripdata.csv` |
| Station list (dock capacity) | Same page, current station list CSV | `data/raw/current_bluebikes_stations.csv` |
| Daily weather (only for `load_data.py`) | NOAA Climate Data Online, Boston Logan (USW00014739) | `data/raw/noaa_boston_daily_aug.csv` |

Never open and save the trip CSV in Excel. It rewrites the timestamps and deletes the date and hour. Inspect files in Python (`df.head()`).

```bash
python code/load_data.py                       # step 1 outputs to data/processed/ (needs all three files)
python code/load_data_helper.py                # trips and stations only; prints the cluster
python -m coolboston.build_bikes_saad          # NEU cluster -> web/data/bikes.json
python -m coolboston.build_bikes --demo        # no data needed: hand-made demo profiles
python -m coolboston.build_bikes --month 202509   # downloads and caches a month of trips (needs network)
```

The month, cluster radius and file names for the loader are in the CONFIG block at the top of `code/load_data.py`.

## 4. Run the web app

```bash
cd web
npm install                          # first time only
cp .env.local.example .env.local     # then set NEXT_PUBLIC_MAPBOX_TOKEN=pk....
npm run dev                          # open http://localhost:3000/bikes
```

On the page, press play and scrub the timeline from 07:00 to 10:00. Stations in Fenway and Back Bay drain; downtown ones fill. `.env.local` is gitignored, so the token is not committed.

Other scripts: `npm run build` (production build and type check), `npm start` (serve the build), `npm run lint`.

The map simulates one day by default. To simulate more days, change `DAYS` at the top of `web/components/BikesMapView.tsx`. Extra days repeat the same weekday pattern, since the rates are month averages.

## 5. Build the standalone demo

```bash
python demo/bikes/build_demo.py
open demo/bikes/dist/bluebikes-markov-demo.html     # macOS; no server or Mapbox token needed
```

`web/data/cells.json` (land-tile background) is not in the repo, so the demo draws without land tiles; everything else works. The page also has a "Load bikes.json" button to load another `bikes.json`.

## 6. Run the live poller (optional)

Uses only the Python standard library and needs network access.

```bash
python polling/fetch_feeds.py        # writes polling/feeds/feeds.json
python polling/poll.py               # one poll into polling/bluebikes.db (created on first run)
python polling/poll.py --loop        # every 10 minutes; Ctrl+C to stop
python polling/health_check.py       # reports whether polling is healthy (read-only)
```

`polling/poller.py` is a second, separate implementation of the poller (it expects a database you create first with `sqlite3 <db> < polling/schema.sql`). The two have not been merged.

## Troubleshooting

- **`No module named pytest`**: `pip install pytest`.
- **`No module named coolboston`**: run `export PYTHONPATH=$PWD/pipeline/src` from the repo root.
- **`No module named numpy` or `pandas`**: `pip install -r requirements.txt`.
- **`Cannot find module './###.js'` from Next.js**: stale cache. Stop the server, delete `web/.next`, and restart.
- **Map is blank or shows a 401**: the Mapbox token in `web/.env.local` is missing, wrong or incomplete. A valid token has a 22-character signature after the last dot.
- **`load_data.py` fails on missing weather**: it needs the NOAA CSV. `python code/load_data_helper.py` runs the trip and station half without it.
- **`Missing input files` from the loader**: the three raw files are not in `data/raw/` or are named differently from the CONFIG block.

## Windows (PowerShell) differences

- Activate the venv with `.venv\Scripts\Activate.ps1`, and use `python` instead of `python3`.
- Set the path with `$env:PYTHONPATH = "$PWD\pipeline\src"` instead of `export`.
- Copy the env file with `Copy-Item web\.env.local.example web\.env.local`.
- Delete the stale cache with `Remove-Item -Recurse -Force web\.next`.
- Open the demo by double-clicking `demo\bikes\dist\bluebikes-markov-demo.html`.
