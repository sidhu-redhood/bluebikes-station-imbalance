# Tasks: Bluebikes station Markov model

Spec: `.kiro/specs/SPEC-bike-markov.md`. Conventions as in `todo.md`:
- Implement, test, verify, then commit, one task per commit.
- Stage specific files only.
- Never push or deploy unasked.
- When a task is done, append "(done)" to its header, tick its criteria, and add a "Notes:" paragraph before **Dependencies**.

Standard verification: `make check` and `make build`.

A reference implementation of every file below is already in the repo. For each task: read the spec section, review the reference against it, fix anything that disagrees, run the verification.

---

## Phase B: Station Markov model

### Task B1: Markov core in Python [M]

**Description:** `bike_markov.py`: `StationModel`, `generator`, `transition` (uniformization), `hourly_transitions`, `forecast`, `hit_probability`, `stationary_occupancy`, `simulate_day`. numpy only.

**Acceptance criteria:**
- [ ] Every Python model test in the spec passes.
- [ ] `transition` uses log-space Poisson weights, no lgamma approximation.
- [ ] Module docstring states the state, rates, step and the numpy → torch note.

**Verification:**
- [ ] `cd pipeline && uv run pytest -q tests/test_bike_markov.py`

**Dependencies:** None

**Files likely touched:** `pipeline/src/coolboston/bike_markov.py`, `pipeline/tests/test_bike_markov.py`, `pipeline/pyproject.toml` (declare `numpy` and `pandas`; today they arrive through geopandas), `pipeline/uv.lock`

### Task B2: bikes.json from trips, and the demo model [M]

**Description:** `build_bikes.py`: fetch (cached), `hourly_rates`, `initial_bikes`, `select_cluster`, `build_records`, `validate`, `make_doc`, `write_bikes`, `demo_doc`, CLI. Commit the `--demo` output as `web/data/bikes.json`.

**Acceptance criteria:**
- [ ] Both trip column formats work; unknown formats fail naming the columns found.
- [ ] Weekend trips are excluded; rates are per weekday.
- [ ] Validation matches the contract; the file is written with `allow_nan=False`.
- [ ] The `initialBikes` rule is labelled ASSUMPTION; the censoring caveat is in the docstring.
- [ ] `--month` on a real month prints the station count (expect 45) and `source.kind = "bluebikes-trips"`. Run once by a human with network; record the month in Notes.

**Verification:**
- [ ] `cd pipeline && uv run pytest -q tests/test_build_bikes.py`
- [ ] `uv run python -m coolboston.build_bikes --demo`

**Dependencies:** B1

**Files likely touched:** `pipeline/src/coolboston/build_bikes.py`, `pipeline/tests/test_build_bikes.py`, `web/data/bikes.json`, `.gitignore` (`pipeline/data/raw/bluebikes/`)

### Task B3: Web contract and TypeScript model [M]

**Description:** `lib/bikes.ts` (`parseBikes`, `BikeDataError`, types, `BIKES_URL`), `app/api/bikes/route.ts`, `lib/markov.ts`. Write the parity fixture from Python and test against it.

**Acceptance criteria:**
- [ ] `lib/markov.test.ts` passes, including parity to 1e-9.
- [ ] `lib/bikes.test.ts` covers the accept/reject cases in the spec.
- [ ] A malformed `bikes.json` fails `next build`.
- [ ] No React in `lib/`.

**Verification:**
- [ ] Standard. Regenerate the fixture: the snippet in Notes (Python writes `web/lib/__fixtures__/markov-parity.json` from four demo stations).

**Dependencies:** B1, B2

**Files likely touched:** `web/lib/bikes.ts`, `web/lib/bikes.test.ts` (new), `web/lib/markov.ts`, `web/lib/markov.test.ts`, `web/lib/__fixtures__/markov-parity.json`, `web/app/api/bikes/route.ts`

### Task B4: /bikes map page [M]

**Description:** `app/bikes/page.tsx`, `components/BikesClient.tsx`, `components/BikesMapView.tsx` per the spec's Web section. Add a "Bikes" link in `TopBar`.

**Acceptance criteria:**
- [ ] Stations render on Mapbox Standard through the existing `DeckOverlay` (one overlay per map; no second overlay).
- [ ] Camera starts at `BOSTON_VIEW`; zoom stays ≥ 15.
- [ ] Play, scrub and New day work; the panel shows 30-minute risks from the last poll, labelled Estimate.
- [ ] Empty and full are marked by shape as well as colour; the error state uses `role="alert"`.

**Verification:**
- [ ] Standard, then `npm run dev`, open `/bikes`, play from 07:00 to 10:00 and confirm Downtown stations fill and Back Bay/South End stations drain.

**Dependencies:** B3

**Files likely touched:** `web/app/bikes/page.tsx`, `web/components/BikesClient.tsx`, `web/components/BikesMapView.tsx`, `web/components/TopBar.tsx`

### Task B5: Station panel and timeline strip in the app [M]

**Description:** Bring the demo's 2 h forecast fan chart, hourly-rate bars, top-5 risk list and empty/full timeline strip into `/bikes`, using Recharts or canvas. Pure data shaping in `lib/bikesView.ts` with tests.

**Acceptance criteria:**
- [ ] Fan chart: observed past hour as a line, forecast distribution as shading, expected count dashed.
- [ ] Network totals match the demo's definitions (empty and full station-minutes, riders who found no bike or dock).

**Verification:**
- [ ] Standard.

**Dependencies:** B4

**Files likely touched:** `web/lib/bikesView.ts`, `web/lib/bikesView.test.ts`, `web/components/BikesPanel.tsx`, `web/components/BikesTimeline.tsx`

### Task B6: Standalone demo and docs [S]

**Description:** `demo/bikes/` (template, `markov.js`, `build_demo.py`), `BIKES.md`, a README section linking to it.

**Acceptance criteria:**
- [ ] `python3 demo/bikes/build_demo.py` writes one HTML file under 1 MB that opens offline (fonts fall back).
- [ ] "Load bikes.json" accepts a `--month` file and the source pill changes to "Rates from trips, YYYY-MM".
- [ ] `markov.js` matches the parity fixture (`node` check in Notes).
- [ ] Light and dark mode, and a 390 px wide layout, are checked by screenshot.

**Verification:**
- [ ] Build, open the page, play the morning rush.

**Dependencies:** B2, B3

**Files likely touched:** `demo/bikes/*`, `BIKES.md`, `README.md`, `.gitignore` (`demo/bikes/dist/`)
