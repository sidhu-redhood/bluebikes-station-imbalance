# Bluebikes station tracker (FAI project, Markov model)

Adds a Markov model of the bike count at each Bluebikes station, drawn on CoolBoston's map.

## Files

| Where | What |
|---|---|
| `pipeline/src/coolboston/bike_markov.py` | The model: birth–death chain per station, 10-minute transition matrices by uniformization, forecasts, run-out/fill-up probabilities, exact day simulation |
| `pipeline/src/coolboston/build_bikes.py` | Estimates weekday hourly rates per station from Bluebikes trip history and writes `web/data/bikes.json` |
| `pipeline/tests/test_bike_markov.py`, `test_build_bikes.py` | Tests (no network) |
| `web/lib/markov.ts`, `web/lib/bikes.ts` | Same model in TypeScript, and the bikes.json validator |
| `web/app/bikes/page.tsx`, `web/components/BikesMapView.tsx` | `/bikes` route on the Mapbox Standard map with the deck.gl overlay |

## Run

```bash
# from the repo root (coolboston has no installer, so put src on the path)
export PYTHONPATH=$PWD/pipeline/src
python -m pytest pipeline/tests/test_bike_markov.py pipeline/tests/test_build_bikes.py
python -m coolboston.build_bikes --month 202509            # real trips (downloads once, cached)
python -m coolboston.build_bikes --month 202509 --status   # also take start counts from live GBFS
python -m coolboston.build_bikes --demo                    # offline demo model
python -m coolboston.build_bikes_saad                      # NEU cluster (needs data/raw/, see README)
cd web && npm run dev    # open http://localhost:3000/bikes
```

The standalone demo page accepts any `bikes.json` from this script through its "Load bikes.json" button.

## Caveats

- Rates come from trips that happened. Empty stations record no pickups, so demand is under-counted where stations often run out. The HMM step (`HMM.md`) corrects this: `outputs/hmm/<month>/bikes_hmm.json` has corrected rates and estimated start counts.
- Stations are modelled independently; a bike leaving one station does not appear at another.
- Without `--status`, starting counts are an assumption (half full, shifted by the expected morning drain).
- Demo-mode stations are hand-placed near real stations; positions and capacities are approximate.
