# Spec: bike-markov

Module of `capability-map.md`. Depends on: nothing in CoolBoston's heat modules. Reuses the map setup (`web/lib/mapConfig.ts`, `components/DeckOverlay.tsx`) and the 100 m land grid (`web/data/cells.json`).

Status: DRAFT, awaiting human review. A reference implementation is in the repo (files listed under Tasks); build to this spec, and treat the reference as a guide, not as finished work.

## Objective

FAI project, part 1: track Bluebikes station imbalance. For each station in a Downtown Boston cluster, model the bike count as a Markov chain, forecast it from 10-minute polls, and give the chance the station runs out of bikes or docks before a later poll. Show it on CoolBoston's map.

The app keeps three things distinct, in code and in labels:
1. **Input data:** trip-history rates (or demo rates) and starting counts.
2. **The model's forecasts:** estimates.
3. **Simulated days:** draws from the same model, used as ground truth only in the demo.

Every forecast is labelled an estimate. Demo rates are labelled "Demo rates" wherever they appear.

## Model

State: `b` = bikes docked, `0..C`, `C` = dock capacity. `b = 0` is **empty**, `b = C` is **full**.

Within clock hour `h` the station is a continuous-time birth–death chain (rates in bikes per hour):

```
b → b+1  at rate λin[h]    (a return; blocked when b = C → a rider finds no dock)
b → b−1  at rate λout[h]   (a pickup; blocked when b = 0 → a rider finds no bike)
```

- **Generator** `Q_h` (per minute): tridiagonal, rows sum to 0.
- **Poll matrix** `P_h = expm(Q_h · Δt)`, `Δt = 10` min, by **uniformization**: `q = max exit rate`, `U = I + Q/q`, `P = Σ_k Poisson(k; qΔt) Uᵏ`. Stop when the Poisson tail < 1e-12 and `k > qΔt`; renormalise rows. Poisson weights are built in log space term by term (`log w_k = log w_{k−1} + log λ − log k`), never with an approximate lgamma. No scipy.
- **Forecast** from poll time `t₀` with count `b₀`: `p₀ = e_{b₀}`, `p_{k+1} = p_k · P_{h(t₀ + kΔt)}`. Each step uses the hour it starts in.
- **Hit probability** ("runs out within T"): the same propagation with the target state absorbing (zero row in `Q`). The mass on the target after `T/Δt` steps. This counts run-outs between polls, not only at polls. Must be ≥ the marginal `P(b = target at T)` and non-decreasing in `T`.
- **Simulation**: exact Gillespie per station; exponential clocks restart at each hour boundary (memoryless, so exact). Records the count every minute and the riders turned away, by minute.
- **Stationary distribution** for one hour's rates: truncated geometric, `π_b ∝ (λin/λout)^b`. Used as a test oracle.

Python (`pipeline/src/coolboston/bike_markov.py`, numpy only) and TypeScript (`web/lib/markov.ts`) implement the same functions. **They must agree to 1e-9** on a parity fixture written by the pipeline (`web/lib/__fixtures__/markov-parity.json`). The team plan is PyTorch for tensors; the Python module uses only matrix products and elementwise ops so `np` → `torch` is mechanical. Do not add torch in this module.

## Contract: `web/data/bikes.json`, schema v1

```
version      1
stepMinutes  10
startMinute  300            (simulated days start 05:00)
source       { kind: "demo" | "bluebikes-trips", month: "YYYYMM" | null, weekdays: int | null, generated: "YYYY-MM-DD" }
stations[]   id, name, lat, lon, capacity, initialBikes, ratesIn[24], ratesOut[24]   (exactly these keys, in this order)
```

- Rates: mean **weekday** bikes per hour, local clock hours 0..23. `ratesIn` = returns, `ratesOut` = pickups. Finite, ≥ 0, length 24.
- `capacity` integer ≥ 1; `initialBikes` in `0..capacity`.
- Version is set in two places: `SCHEMA_VERSION` in `build_bikes.py` and `BIKES_SCHEMA_VERSION` in `web/lib/bikes.ts`. Change both in the same commit.
- Pipeline writes with `allow_nan=False` and validates before writing. Web validates with `parseBikes` (throws `BikeDataError` naming the field). `app/api/bikes/route.ts` imports the file statically so a bad file fails `next build`.

## Pipeline: `coolboston.build_bikes`

```
uv run python -m coolboston.build_bikes --month 202509            # real trips
uv run python -m coolboston.build_bikes --month 202509 --status   # start counts from live GBFS
uv run python -m coolboston.build_bikes --demo                    # offline demo model
```

Sources (cite in the module docstring):
- Trips: `https://s3.amazonaws.com/hubway-data/{YYYYMM}-bluebikes-tripdata.zip`. Bluebikes already removes staff trips and trips under 60 s.
- Stations: `https://gbfs.bluebikes.com/gbfs/en/station_information.json` (name, lat, lon, capacity).
- Counts: `https://gbfs.bluebikes.com/gbfs/en/station_status.json` (`num_bikes_available`), only with `--status`.

Rules:
- `fetch_*` download once to `pipeline/data/raw/bluebikes/` and reuse the cache (station_status is timestamped, not reused). Paths are default arguments so tests pass temp paths. No network in tests.
- Trip columns: support the current format (`started_at`, `ended_at`, `start_station_id`, `end_station_id`) and the legacy one (`starttime`, `stoptime`, `start station id`, `end station id`). Fail with a message listing the columns found.
- Station join: trip station id = GBFS `short_name`; fall back to the station name.
- Rates: count pickups by start hour and returns by end hour, weekdays only, divided by the number of weekdays in the file.
- Cluster: stations within 2,600 m of `BOSTON_VIEW`'s centre (−71.06, 42.3555), with capacity and trips, the 45 busiest by total volume.
- `initialBikes` without `--status` is an **ASSUMPTION**, labelled in code: `clip(round(C/2 + (Σ out − Σ in over 05:00–10:00)/2), 1, C−1)`.
- `--demo`: 36 hand-placed stations near real ones in Downtown, Back Bay, South End, North End, Seaport and South Boston, with commute-shaped rates (morning destinations, morning origins, balanced) from a fixed seed. The docstring says positions and capacities are approximate.
- **Censoring caveat**, in the docstring and the UI method text: trips record demand that was served. Empty stations record no pickups, so rates under-count demand where stations often run out. The HMM module will correct this.

## Web: `/bikes`

- `app/bikes/page.tsx`: `TopBar` + map, same grid layout as `/map`.
- `components/BikesClient.tsx`: `checkMapboxToken`, then `next/dynamic` with `ssr: false` (same as `MapClient`).
- `components/BikesMapView.tsx`: `Map` with `MAP_STYLE`, `BASEMAP_CONFIG`, `BOSTON_VIEW` (zoom stays ≥ 15: map rule). Stations are a deck.gl `ScatterplotLayer` through the existing `DeckOverlay`, `slot: "top"`. Opacity in the fill colour's alpha, not the `opacity` prop.
- Colour: bikes ÷ capacity on a diverging ramp, empty `#C8243B` → mid `#8E979F` → full `#1F63B4`. Empty and full stations are also marked by shape (hollow centre / inner ring), not colour alone.
- Timeline: play/pause, time slider 05:00–24:00, "New day" (new seed). Forecasts use the **last 10-minute poll**, shown as "last poll HH:MM".
- Station panel on click: name, bikes and open docks, chance of running out and of filling up within 30 min and 1 h, labelled "Estimate".
- Pure logic stays in `lib/` (`bikes.ts`, `markov.ts`); React only in `components/`.

## Standalone demo: `demo/bikes/`

`python3 demo/bikes/build_demo.py` writes one self-contained HTML page (`demo/bikes/dist/`) for the class presentation, no Mapbox token. Land is CoolBoston's 100 m tiles near the stations in State Plane metres (same projection as `web/lib/grid.ts`); colours are `BASEMAP_CONFIG`'s. It has the map, a "Bikes now / Risk in 30 min" toggle, the empty/full timeline strip, network totals (empty and full station-minutes, riders who found no bike or no dock), the top-5 risk list, the station card (risks, 2 h forecast fan chart with observed past, hourly rates, the hour's transition matrix), a Brier-score table against a base-rate forecast, and "Load bikes.json". `markov.js` there must stay in step with `web/lib/markov.ts` (same parity fixture).

## Out of scope (later modules)

HMM for historical counts; censored-demand estimation; trips linking stations (origin–destination); rebalancing vans, incentives, nudges and the MDP; weekends; weather.

## Testing

- Python: `test_bike_markov.py` (generator rows sum to 0 and respect bounds; transition rows sum to 1 and match a Taylor-series expm; semigroup `P(5)·P(5) = P(10)`; zero rates give I; forecast converges to stationary; hit probability monotone and ≥ marginal; an hour with no rates changes nothing; simulation mean matches forecast mean; simulation counts turned-away riders; bad rates rejected). `test_build_bikes.py` (weekday means for both column formats; `initialBikes` ordering and bounds; cluster filter and order; records round-trip and validate; validator rejects bad rates; demo has the commute imbalance).
- Web: `lib/markov.test.ts` (stochastic matrix, semigroup, stationary, hit monotone, simulation mean, Python parity to 1e-9). `lib/bikes.test.ts` (accepts the demo file; rejects wrong version, missing key, 23 rates, initialBikes > capacity).
- Standard: `make check`, `make build`.
