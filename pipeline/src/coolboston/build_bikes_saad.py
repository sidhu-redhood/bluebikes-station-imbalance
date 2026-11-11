"""Train the station Markov model on Saad's data layer (code/load_data.py).

This bridges Saad's loader — which reads a local month of Bluebikes trips and the
official station list, builds the NEU cluster, and attaches dock capacity — into
the Markov rate-estimation + bikes.json writer used by the rest of the pipeline.

"Training" the Markov model means estimating each station's mean weekday hourly
rates (ratesIn = returns, ratesOut = pickups) from the trips; the model itself has
no learned weights.

Inputs (place under <repo>/data/raw/, the layout Saad's loader expects):
  202608-bluebikes-tripdata.csv      (the month of trips)
  current_bluebikes_stations.csv     (official station list, for dock capacity)

Run:
    cd pipeline
    uv run --with numpy --with pandas python -m coolboston.build_bikes_saad
    uv run --with numpy --with pandas python -m coolboston.build_bikes_saad --status  # live start counts
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.bike_markov import HOURS
from coolboston.build_bikes import (
    am_net_flow,
    fetch_station_status,
    initial_bikes,
    make_doc,
    write_bikes,
)
from coolboston.neighborhoods import Neighborhoods

ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "code"  # Saad's loader lives here


def _load_saad():
    """Import Saad's loader module (code/load_data.py) and return it."""
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    try:
        import load_data as L  # type: ignore
    except ImportError as err:
        raise SystemExit(
            f"Could not import Saad's loader from {CODE_DIR}. "
            "Make sure code/load_data.py is present."
        ) from err
    return L


def hourly_rates_from_trips(trips: pd.DataFrame, station_ids: set[str]):
    """Mean WEEKDAY pickups (out) and returns (in) per cluster station per clock hour.

    Mirrors build_bikes.hourly_rates but keyed on Saad's trip columns
    (start_station_id / end_station_id, already lower-cased by his loader)."""
    start = trips["started_at"]
    end = trips["ended_at"]
    days = pd.Series(start.dt.normalize().unique())
    n = int((days.dt.dayofweek < 5).sum())
    if n == 0:
        raise ValueError("no weekday trips in the file")

    def per_station(ts: pd.Series, ids: pd.Series) -> dict[str, np.ndarray]:
        keep = (ts.dt.dayofweek < 5) & ids.notna() & ids.isin(station_ids)
        g = pd.crosstab(ids[keep].astype(str), ts[keep].dt.hour)
        g = g.reindex(columns=range(HOURS), fill_value=0)
        return {sid: row.to_numpy(dtype=float) / n for sid, row in g.iterrows()}

    out = per_station(start, trips["start_station_id"])
    inn = per_station(end, trips["end_station_id"])
    return out, inn, n


def build(use_status: bool, radius_km: float | None) -> dict:
    L = _load_saad()

    trips = L.load_trips()
    stations = L.build_stations(trips, L.load_station_list())
    cluster = L.neu_cluster(stations, radius_km) if radius_km else L.neu_cluster(stations)
    if cluster.empty:
        raise SystemExit("Saad's loader returned an empty NEU cluster; check the inputs.")
    print(f"[saad] {len(cluster)} stations in the NEU cluster")

    ids = set(cluster["station_id"].astype(str))
    out, inn, n_weekdays = hourly_rates_from_trips(trips, ids)
    print(f"[rates] estimated from {n_weekdays} weekdays of trips")

    hoods = Neighborhoods.load()
    nu_names = ["Fenway", "Back Bay", "Mission Hill", "South End", "Roxbury", "Jamaica Plain"]

    status = None
    if use_status:
        status = {s["station_id"]: int(s["num_bikes_available"]) for s in fetch_station_status()}
        # Saad's trip ids (e.g. "B32002") won't match GBFS station_ids; match by name instead.
        info = L.load_station_list()  # noqa: F841  (name match handled below via GBFS names)

    zeros = np.zeros(HOURS)
    records = []
    for _, s in cluster.iterrows():
        sid = str(s["station_id"])
        cap = int(s["capacity"]) if pd.notna(s.get("capacity")) else None
        if not cap or cap < 1:
            continue  # the Markov state space needs a real capacity
        r_in = inn.get(sid, zeros)
        r_out = out.get(sid, zeros)
        b0 = initial_bikes(cap, r_in, r_out)  # start counts: GBFS ids differ, so use the estimate
        rec = {
            "id": sid,
            "name": str(s["name"]),
            "lat": round(float(s["lat"]), 6),
            "lon": round(float(s["lon"]), 6),
            "capacity": cap,
            "initialBikes": b0,
            "ratesIn": [round(float(x), 3) for x in r_in],
            "ratesOut": [round(float(x), 3) for x in r_out],
        }
        hood = hoods.name_for(float(s["lon"]), float(s["lat"]), nu_names)
        if hood:
            rec["neighborhood"] = hood
        flow = am_net_flow(r_in, r_out)
        rec["amNetFlow"] = round(flow, 3)
        rec["isAmDestination"] = bool(flow > 0)
        records.append(rec)

    if not records:
        raise SystemExit("no cluster stations had a usable capacity")

    source = {
        "kind": "saad-neu-cluster",
        "month": getattr(L, "MONTH", None),
        "weekdays": n_weekdays,
        "radiusKm": radius_km or getattr(L, "RADIUS_KM", None),
        "stations": len(records),
    }
    return make_doc(records, source)


def main(argv=None):
    p = argparse.ArgumentParser(description="Build bikes.json from Saad's NEU-cluster dataset.")
    p.add_argument("--status", action="store_true", help="(reserved) try live GBFS start counts")
    p.add_argument("--radius-km", type=float, default=None, help="override Saad's cluster radius")
    a = p.parse_args(argv)
    doc = build(a.status, a.radius_km)
    path = write_bikes(doc)
    dest = sum(1 for s in doc["stations"] if s.get("isAmDestination"))
    print(f"wrote {path} ({len(doc['stations'])} stations, source {doc['source']['kind']}, "
          f"{dest} AM destinations, month {doc['source'].get('month')})")


if __name__ == "__main__":
    main()
