# Tasks: Station HMM (hidden counts, missed riders, van moves)

Write-up: `HMM.md`. Conventions as in `todo-bike-markov.md`: one task per commit, stage specific files only,
never push or deploy unasked.

Verification: `cd pipeline && PYTHONPATH=src python3 -m pytest -q tests/test_station_hmm.py`

---

## Phase H: Station HMM

### Task H1: Slots from trips and polls (done) [M]

**Description:** `hmm_data.py`: `build_slots` (net, lowest and highest point, pickups, returns, constant-count segments, poll offsets), trip and station loaders for Saad's processed files, `load_polls` for the poller database.

**Acceptance criteria:**
- [x] Hand-built example gives the expected net, low, high, segments and poll offsets.
- [x] Polls join on `stations.short_name`, use `last_reported` when present, drop stations that are not renting, convert UTC to Boston time.

Notes: trip-file ids equal GBFS `short_name`; GBFS `station_id` is a different id.

**Dependencies:** None

### Task H2: Model, forward-backward, EM (done) [L]

**Description:** `station_hmm.py`: move kernel (van visit to a level, one-bike move), feasibility, demand evidence with a shared day multiplier, poll emissions; scaled forward-backward over all stations at once; closed-form M-step with weak priors.

**Acceptance criteria:**
- [x] Posteriors, log-likelihood and expected moves match brute-force enumeration of every path on a tiny case.
- [x] EM objective never decreases.

Notes: numpy, like `bike_markov.py`; the torch swap is mechanical.

**Dependencies:** H1

### Task H3: Planted-truth simulator and evaluation (done) [M]

**Description:** `hmm_sim.py`, `hmm_eval.py`; settings for trips only, polls, ablations and stress cases; baselines (half-full assumption replay, last poll plus trips).

**Acceptance criteria:**
- [x] Tests: polls pin counts down; trips-only intervals are calibrated; demand correction cuts censoring bias; day multipliers are found.
- [x] `python -m coolboston.hmm_eval` writes `outputs/hmm/eval/eval.md`.

Notes: results table in `HMM.md`. Weak spot: station-specific busy days.

**Dependencies:** H2

### Task H4: Run on a month, outputs, corrected bikes.json (done) [M]

**Description:** `run_hmm.py`: stations, rates, counts, days, params, summary, plots; `bikes_hmm.json` validated by `build_bikes.validate`.

**Acceptance criteria:**
- [x] August 2026 runs in about 80 s and writes every output.
- [x] `bikes_hmm.json` passes the schema check.

Notes: 2026-08, trips only.

**Dependencies:** H2

### Task H5: Run with poll data [M]

**Description:** Once `polling/bluebikes.db` overlaps a published trip month, run with `--polls`, and report the held-out poll error on real data.

**Acceptance criteria:**
- [ ] Held-out poll error reported against last poll plus trips.
- [ ] Capacity check against `num_docks_disabled`.

**Dependencies:** H4, poller running

### Task H6: Hand-off to the RL simulator [S]

**Description:** Use `bikes_hmm.json` rates and start counts in the simulator, and the learned van schedule as the "operators today" baseline policy.

**Dependencies:** H4, RL environment
