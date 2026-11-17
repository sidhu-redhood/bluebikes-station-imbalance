# Station HMM: hidden bike counts, missed riders and van moves

Part 2 of the proposal ("estimating the current state"), and the fix for the caveat in `BIKES.md`:
trip rates under-count demand at stations that run empty.

## The problem

The trip history records when bikes left and arrived. It never records how many bikes were docked, and it
leaves out three things we need:

1. **Bike counts.** `bikes.json` starts every station at an assumed half-full count.
2. **Riders who found an empty or full station.** They leave no record, so trip rates are a lower bound on demand.
3. **Van moves.** Rebalancing is not in the trip data. In August 2026, trips alone are impossible on 119 of
   1,302 station-days: the day's net flow swings by more than the station has docks, so bikes must have been
   moved.

## The model

Hidden state: bikes docked at each station at the start of each 10-minute slot (0 to capacity).

In each slot:

- **Unrecorded move.** Usually nothing. With probability `p[station, hour]` a van visits and sets the station to
  a level drawn from `g[station]`. With a small probability `u` a single bike is added or removed.
- **Trips.** These are observed exactly and in order, so given the count after the move, the path through the
  slot is known. Every pickup needs a bike and every return needs a dock; a violation costs `exp(-kappa)` per
  bike, so one bad record cannot break the fit. Riders want bikes at a Poisson rate `lam[station, day type, hour]`
  times a shared multiplier for the day (weather), but they are served only while a bike is there. A busy hour
  with no pickups therefore points to an empty station. Returns work the same way with `mu` and free docks.
- **Polls (when we have them).** A GBFS snapshot sees the count, with a little noise.

All of it is learned by EM (Baum-Welch), written from scratch in numpy (`pipeline/src/coolboston/station_hmm.py`):

- **E-step:** scaled forward-backward over the whole month, all stations at once.
- **M-step:** `p`, `g` and `u` from the expected number of each kind of move. `lam` is pickups divided by the
  expected hours with a bike, which is the censored-Poisson estimate and therefore the demand correction.
  `mu` is returns divided by the expected hours with a free dock. Day multipliers are the day's trips divided by
  what the rates predict.
- **Priors:** weak priors keep everything finite.
- **Monotone objective:** EM never lowers the objective, and a test checks this.

## Run it

From the repo root, with Saad's processed files in `data/processed/` (from `code/load_data.py`):

```bash
PYTHONPATH=pipeline/src python3 -m coolboston.run_hmm --month 2026-08 --plots 4
PYTHONPATH=pipeline/src python3 -m coolboston.run_hmm --month 2026-11 --polls polling/bluebikes.db
PYTHONPATH=pipeline/src python3 -m coolboston.hmm_eval        # planted-truth evaluation, about 8 min
```

A month of 42 stations takes about 80 seconds. Plots need `matplotlib`; everything else needs only `numpy` and
`pandas`.

Outputs go to `outputs/hmm/<month>/` (gitignored):

| File | What it holds |
|---|---|
| `summary.md` | the headline numbers |
| `stations.csv` | per station: empty and full hours a day, pickups seen vs wanted, van visits, bikes added and removed, `count_80pct_width` (how sure the count is) |
| `rates.csv` | per station, day type and hour: raw and corrected pickup and return rates, van visits |
| `counts.csv.gz` | per station and slot: estimated bikes (mean, 10/50/90%), P(empty), P(full), expected empty and full minutes, P(van visit) |
| `days.csv` | per day: trips and the demand multiplier (1 = an average day of that type) |
| `bikes_hmm.json` | `bikes.json` schema v1 with corrected weekday rates and the estimated 05:00 count instead of the half-full assumption. Load it in the demo with "Load bikes.json", or copy it over `web/data/bikes.json`. |
| `plots/` | timelines for the most-censored stations, and demand by hour |

## Does it work? Planted-truth evaluation

`hmm_sim.py` simulates the 42 cluster stations (real capacities, August rates from `bikes.json`, a van
schedule peaking at rush hours). It keeps the true counts, van moves and missed riders, and records only what
Bluebikes would record. Results (`outputs/hmm/eval/eval.md`), 21 days:

| Setting | Count error (bikes): HMM | Baseline | 80% range covers | Empty h/day: true | HMM | Baseline | Demand error: raw | HMM | Bikes moved/day: true | HMM |
|---|---|---|---|---|---|---|---|---|---|---|
| Trips only | 2.10 | 3.23 | 87% | 1.26 | 1.48 | 0.70 | 6.1% | 4.7% | 8.5 | 8.2 |
| Trips only, demand evidence off | 2.94 | 3.23 | 71% | 1.26 | 0.27 | 0.70 | 6.1% | 6.1% | 8.5 | 6.2 |
| Trips + one poll an hour | 0.21 | 0.66 | 98% | 1.30 | 1.32 | 0.77 | 6.3% | 3.2% | 8.0 | 8.4 |
| Trips + a poll every 10 min | 0.03 | 0.50 | 100% | 1.30 | 1.31 | 0.77 | 6.3% | 3.2% | 8.0 | 8.1 |
| Cluster-wide 30% day swings (weather) | 2.21 | 3.22 | 86% | 0.97 | 1.22 | 0.61 | 6.4% | 4.4% | 8.0 | 7.9 |
| Station-specific 30% day swings (events) | 2.19 | 3.11 | 88% | 1.10 | 1.50 | 0.73 | 9.3% | 9.5% | 7.9 | 9.3 |
| Vans arrive mid-slot | 2.11 | 3.21 | 87% | 1.12 | 1.38 | 0.70 | 5.9% | 4.0% | 8.2 | 8.2 |

How to read the table:

- **Baselines.** Without polls, the baseline is today's approach: the half-full assumption at 05:00, then add
  and remove the recorded trips. With polls, it is the last poll plus the trips since.
- **Demand error.** The mean per-station error in weekday pickups a day.
- **Held-out polls.** These are real observations the model never saw. Error: HMM 0.74 vs last poll + trips
  1.10 (hourly polls), and 0.54 vs 0.91 (10-minute polls).

What it shows:

- **Trips alone beat the current assumption** on counts by about a third. They also halve the per-station error
  on empty hours (0.29 vs 0.60 h/day) and recover van moves almost exactly.
- **Even hourly polls make the counts nearly exact.** Collecting poll data is the single biggest improvement
  available.
- **The demand evidence is essential.** Without it the model is overconfident (71% coverage) and misses most of
  the empty time.
- **Station-specific busy and quiet days are the weak spot.** These are swings the cluster-wide multiplier
  cannot see, like a game at Fenway. With trips only, the model then overestimates empty time by about a third,
  and the demand correction stops helping. Polls fix this.

## August 2026, trips only

These results are provisional until poll data covers a trip month.

- **Empty and full time.** The average station is empty 2.9 h and full 1.5 h a day. The simulation suggests the
  empty figure is somewhat high, by up to a third.
- **Missed riders.** The cluster sees 2,272 weekday pickups a day; about 2,526 riders wanted one, so roughly 10%
  were missed. The most censored stations are Whittier St Health Center, Fenway at Museum Rd, Flat 9 at
  Whittier, Kenmore Square and HMS/HSPH, each missing 23–35% of pickups.
- **Van moves.** About 170 bikes are added and 170 removed a day, in roughly 50 van visits. The most activity
  is at Fenway at Museum Rd, Fenway at Forsyth Way, Christian Science Plaza and Mass Ave at Boylston.
- **Certainty varies.** Stations that rarely hit 0 or capacity have wide count ranges (see `count_80pct_width`).
  The model says so rather than guessing.

## Limitations

- **Stations are fitted independently.** A bike leaving one station does not appear at another. This is the
  same as `bike_markov.py`.
- **Day effects are cluster-wide only.** Station-specific events are not modelled (see the evaluation).
- **Van moves are timed to the slot start.** Mid-slot arrivals cost little (see the evaluation).
- **Riders who find a station empty are treated as lost.** Some walk to a neighbour, where their pickup is
  counted twice in a sense: once as missed here, once as real demand there.
- **Rates are averages for each hour and day type.** No weather covariates are used yet; the NOAA file from
  `load_data.py` could explain the day multipliers.
- **Capacity comes from the station list.** Disabled bikes and docks reduce the real capacity; polls (the
  `num_docks_disabled` field) could correct it.
- **EM has a slow tail** on quiet stations. 60 iterations is the default; the objective still moves by about
  0.1 per iteration at that point.

## Files

| File | Role |
|---|---|
| `pipeline/src/coolboston/hmm_data.py` | trips and polls to 10-minute slots; poll loader (joins `stations.short_name`, UTC to Boston time) |
| `pipeline/src/coolboston/station_hmm.py` | the model: forward-backward, EM, summaries |
| `pipeline/src/coolboston/hmm_sim.py` | planted-truth simulator |
| `pipeline/src/coolboston/hmm_eval.py` | evaluation against baselines |
| `pipeline/src/coolboston/run_hmm.py` | command-line run and outputs |
| `pipeline/tests/test_station_hmm.py` | tests, including forward-backward against brute-force enumeration of every path |

## Next

- **Run with polls** once the poller overlaps a published trip month.
- **Feed the RL simulator.** `bikes_hmm.json` gives it corrected demand and realistic starting counts, and the
  learned van schedule (`p`, `g`) is a ready-made "what operators do today" baseline policy.
