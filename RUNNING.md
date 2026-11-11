# Running the project (Samarth-dev)

How to run every program on this branch. Commands are shown for Windows PowerShell;
on macOS/Linux replace `;` with `&&` and `python` with `python3` where needed.

## Prerequisites

- **Python 3.11+** (tested on 3.13)
- **Node.js 18+** and npm (for the web app)
- Python packages for the data pipeline:

  ```powershell
  python -m pip install numpy pandas
  ```

  The pollers (`polling/`) use only the standard library, so they need no install.

## What's in here

| Area | Path | Needs |
| --- | --- | --- |
| GBFS pollers | `polling/` | network |
| Data pipeline + Markov model | `pipeline/` | numpy, pandas |
| Saad's NEU-cluster loader | `code/` | numpy, pandas, trip CSVs in `data/raw/` |
| Standalone demo page | `demo/bikes/` | `web/data/bikes.json` |
| Web app (3D map UI) | `web/` | Node, a Mapbox token |

---

## 1. Poll the live Bluebikes feeds

Fetches the GBFS feed list, then records one snapshot into `polling/bluebikes.db` (SQLite).

```powershell
python polling/fetch_feeds.py   # writes polling/feeds/feeds.json
python polling/poll.py          # one poll cycle
python polling/poll.py --loop   # keep polling every 10 min (Ctrl+C to stop)
```

Tests:

```powershell
python -m pytest polling -q
```

## 2. Build the station data (`web/data/bikes.json`)

`bikes.json` is the input for both the web app and the standalone demo. Three ways to build it:

```powershell
# From the repo root. The coolboston package has no installer, so put src on the path:
$env:PYTHONPATH = "$PWD\pipeline\src"

# a) Demo profiles, no network (quick):
python -m coolboston.build_bikes --demo

# b) Real trip history for a month (downloads + caches the trip zip):
python -m coolboston.build_bikes --month 202509

# c) Saad's NEU cluster around Northeastern (reads code/load_data.py + data/raw CSVs):
python -m coolboston.build_bikes_saad
```

> The version committed on this branch is the **Saad NEU cluster** (`--build_bikes_saad`),
> 42 stations, source `saad-neu-cluster`.

Pipeline tests:

```powershell
$env:PYTHONPATH = "$PWD\pipeline\src"; python -m pytest pipeline -q
```

## 3. Build the standalone demo page

Produces one self-contained HTML file (no server or Mapbox token needed) at
`demo/bikes/dist/bluebikes-markov-demo.html`.

```powershell
python demo/bikes/build_demo.py
```

Then either open the file directly in a browser, or serve it over a URL:

```powershell
python -m http.server 8000 --directory demo/bikes/dist
# open http://localhost:8000/bluebikes-markov-demo.html
```

> `web/data/cells.json` (the land-tile map background) is not included in this repo.
> The builder detects this and draws the demo with no land tiles; everything else works.

## 4. Run the web app (3D map interface)

```powershell
cd web
npm install                       # first time only
Copy-Item .env.local.example .env.local
# edit .env.local and set NEXT_PUBLIC_MAPBOX_TOKEN to your Mapbox public token (pk....)
npm run dev                       # http://localhost:3000  (the map is at /bikes)
```

Get a free Mapbox public token at https://account.mapbox.com/access-tokens/.

Other web scripts:

```powershell
npm test        # vitest unit tests (Markov model parity with the Python version)
npm run build   # production build + type check
npm start       # serve the production build
```

### Simulation length

The map simulates a representative day (24 hours) driven by per-hour weekday averages.
To run it over multiple days, change `DAYS` at the top of
`web/components/BikesMapView.tsx` (e.g. `const DAYS = 30;`). Each extra day repeats the
same daily pattern, since the rates are month-averaged weekday profiles.

---

## Troubleshooting

- **`Cannot find module './###.js'` from Next.js** — a stale `.next` cache, usually from
  running `npm run build` while `npm run dev` was live. Stop the server, delete `.next`
  (`Remove-Item -Recurse -Force web/.next`), and restart `npm run dev`.
- **`ModuleNotFoundError: No module named 'coolboston'`** — set
  `$env:PYTHONPATH = "$PWD\pipeline\src"` before running the `coolboston` commands.
- **`ModuleNotFoundError: numpy` / `pandas`** — run `python -m pip install numpy pandas`.
- **`load_data.py` fails on missing weather** — the top-level `code/load_data.py` run
  needs a NOAA weather CSV (`data/raw/noaa_boston_daily_aug.csv`) that is not in the repo.
  The trip/station half of its logic runs fine via `python code/load_data_helper.py`.
