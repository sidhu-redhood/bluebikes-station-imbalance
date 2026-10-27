"""Write web/data/bikes.json: per-station hourly rates for the station Markov model (bike_markov.py).

Run (real data, needs network once; files are cached under pipeline/data/raw/bluebikes/):
    uv run python -m coolboston.build_bikes --month 202509
    uv run python -m coolboston.build_bikes --month 202509 --status   # start counts from live GBFS
Run (no network, demo profiles; what the hosted demo ships with):
    uv run python -m coolboston.build_bikes --demo

Sources
  Trips:    Bluebikes monthly trip history, s3.amazonaws.com/hubway-data/{YYYYMM}-bluebikes-tripdata.zip
            (staff trips and trips under 60 s are already removed by Bluebikes)
  Stations: Bluebikes GBFS station_information.json (name, lat, lon, capacity)
  Counts:   Bluebikes GBFS station_status.json, optional, for the starting bike counts

Caveat (censoring): trip history records trips that happened, not trips riders wanted. When a
station sits empty its departures read as 0, so rates here UNDER-estimate true demand at stations
that often run out. The HMM step corrects this; until then treat the rates as a lower bound.

Schema v1
  version, stepMinutes (10), startMinute (300 = 05:00), source {kind, month, weekdays, generated}
  stations[]: id, name, lat, lon, capacity, initialBikes, ratesIn[24], ratesOut[24]
    rates are mean bikes per hour on WEEKDAYS (returns in, pickups out), clock hours 0..23 local.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.bike_markov import HOURS, STEP_MIN
from coolboston.neighborhoods import Neighborhoods

SCHEMA_VERSION = 1

# The cluster around Northeastern University (the "Neighborhoods Near NU" set).
# Kenmore is part of the Fenway polygon in our dataset, so it is covered by "Fenway".
NU_NEIGHBORHOODS = ["Fenway", "Back Bay", "Mission Hill", "South End", "Roxbury", "Jamaica Plain"]
START_MINUTE = 5 * 60
ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = ROOT / "web" / "data" / "bikes.json"
RAW = ROOT / "pipeline" / "data" / "raw" / "bluebikes"

GBFS_BASE = "https://gbfs.bluebikes.com/gbfs/en"
TRIPS_URL = "https://s3.amazonaws.com/hubway-data/{month}-bluebikes-tripdata.zip"

# Cluster centre: CoolBoston's default camera (web/lib/mapConfig.ts BOSTON_VIEW), Downtown Boston.
CLUSTER_CENTER = (-71.06, 42.3555)
CLUSTER_RADIUS_M = 2600
CLUSTER_SIZE = 45

STATION_FIELDS = ("id", "name", "lat", "lon", "capacity", "initialBikes", "ratesIn", "ratesOut")

# Trip CSV column names: current format (2023+) first, legacy (to early 2023) second.
_COLS = {
    "start": ("started_at", "starttime"),
    "end": ("ended_at", "stoptime"),
    "start_id": ("start_station_id", "start station id"),
    "end_id": ("end_station_id", "end station id"),
}


# ---------- fetching (thin, cached) ----------

def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path


def fetch_station_information(path: Path = RAW / "station_information.json") -> list[dict]:
    return json.loads(_download(f"{GBFS_BASE}/station_information.json", path).read_text())["data"]["stations"]


def fetch_station_status(path: Path | None = None) -> list[dict]:
    """Live snapshot. Not cached by default: it is meant to be current."""
    path = path or RAW / f"station_status_{dt.datetime.now():%Y%m%d_%H%M}.json"
    return json.loads(_download(f"{GBFS_BASE}/station_status.json", path).read_text())["data"]["stations"]


def fetch_trips(month: str, raw: Path = RAW) -> pd.DataFrame:
    if len(month) != 6 or not month.isdigit():
        raise ValueError(f"month must be YYYYMM, got {month!r}")
    z = _download(TRIPS_URL.format(month=month), raw / f"{month}-bluebikes-tripdata.zip")
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        return pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)


# ---------- pure logic (tested without network) ----------

def _col(df: pd.DataFrame, key: str) -> str:
    for c in _COLS[key]:
        if c in df.columns:
            return c
    raise KeyError(f"trip file has none of {_COLS[key]} (columns: {list(df.columns)[:12]}...)")


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def hourly_rates(trips: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Mean weekday pickups (out) and returns (in) per station per clock hour.
    Station keys are the trip file's station ids (GBFS short_name). Returns (out, in, n_weekdays)."""
    start = pd.to_datetime(trips[_col(trips, "start")])
    end = pd.to_datetime(trips[_col(trips, "end")])
    days = pd.Series(start.dt.normalize().unique())
    weekdays = days[days.dt.dayofweek < 5]
    n = len(weekdays)
    if n == 0:
        raise ValueError("no weekday trips in the file")

    def per_station(ts: pd.Series, ids: pd.Series) -> dict[str, np.ndarray]:
        keep = (ts.dt.dayofweek < 5) & ids.notna()
        g = pd.crosstab(ids[keep].astype(str), ts[keep].dt.hour)
        g = g.reindex(columns=range(HOURS), fill_value=0)
        return {sid: row.to_numpy(dtype=float) / n for sid, row in g.iterrows()}

    out = per_station(start, trips[_col(trips, "start_id")])
    inn = per_station(end, trips[_col(trips, "end_id")])
    return out, inn, n


# Morning commute window (weekday clock hours) used to classify destination behavior.
AM_WINDOW = range(6, 10)  # 06:00-09:59


def am_net_flow(rates_in: np.ndarray, rates_out: np.ndarray) -> float:
    """Net bikes arriving during the morning commute: returns in minus pickups out over AM_WINDOW.
    Positive = bikes pile up here in the morning (a commute DESTINATION). Negative = it drains
    (a residential ORIGIN)."""
    hours = list(AM_WINDOW)
    return float(np.sum(np.asarray(rates_in)[hours]) - np.sum(np.asarray(rates_out)[hours]))


def initial_bikes(capacity: int, rates_in: np.ndarray, rates_out: np.ndarray) -> int:
    """ASSUMPTION used when no live snapshot is given: operators pre-position bikes overnight, so a
    station that drains in the morning rush starts fuller. Start at half full, shifted by half the
    expected net drain over 05:00-10:00, kept off the empty and full states."""
    drain = float(np.sum(rates_out[5:10]) - np.sum(rates_in[5:10]))
    return int(np.clip(round(capacity / 2 + drain / 2), 1, capacity - 1))


def select_cluster(stations: list[dict], volume: dict[str, float], center=CLUSTER_CENTER,
                   radius_m: float = CLUSTER_RADIUS_M, size: int = CLUSTER_SIZE) -> list[dict]:
    """The `size` busiest stations within radius_m of center that have trips and a capacity."""
    near = [
        s for s in stations
        if s.get("capacity") and volume.get(s["_key"], 0) > 0
        and haversine_m(center[0], center[1], s["lon"], s["lat"]) <= radius_m
    ]
    near.sort(key=lambda s: -volume[s["_key"]])
    return near[:size]


def select_by_neighborhood(stations: list[dict], volume: dict[str, float], names: list[str],
                           hoods: Neighborhoods, size: int | None = None) -> list[dict]:
    """Stations that have trips, a capacity, and fall inside one of the named neighborhoods.
    Each returned station gets a `_neighborhood` tag. Busiest first; capped at `size` if given."""
    picked = []
    for s in stations:
        if not s.get("capacity") or volume.get(s["_key"], 0) <= 0:
            continue
        hood = hoods.name_for(float(s["lon"]), float(s["lat"]), names)
        if hood is None:
            continue
        s["_neighborhood"] = hood
        picked.append(s)
    picked.sort(key=lambda s: -volume[s["_key"]])
    return picked[:size] if size else picked


def build_records(stations: list[dict], out: dict, inn: dict, status: dict[str, int] | None = None) -> list[dict]:
    records = []
    zeros = np.zeros(HOURS)
    for s in stations:
        cap = int(s["capacity"])
        r_in, r_out = inn.get(s["_key"], zeros), out.get(s["_key"], zeros)
        b0 = status.get(s["station_id"]) if status else None
        b0 = initial_bikes(cap, r_in, r_out) if b0 is None else int(np.clip(b0, 0, cap))
        rec = {
            "id": str(s["station_id"]),
            "name": s["name"],
            "lat": round(float(s["lat"]), 6),
            "lon": round(float(s["lon"]), 6),
            "capacity": cap,
            "initialBikes": b0,
            "ratesIn": [round(float(x), 3) for x in r_in],
            "ratesOut": [round(float(x), 3) for x in r_out],
        }
        if s.get("_neighborhood"):
            rec["neighborhood"] = s["_neighborhood"]
        flow = am_net_flow(r_in, r_out)
        rec["amNetFlow"] = round(flow, 3)
        rec["isAmDestination"] = bool(flow > 0)
        records.append(rec)
    return records


def validate(doc: dict) -> dict:
    """Same rules as the web's parser: exact keys, finite non-negative numbers, 24 rates."""
    if doc.get("version") != SCHEMA_VERSION:
        raise ValueError(f"bikes.json: version must be {SCHEMA_VERSION}")
    if not doc["stations"]:
        raise ValueError("bikes.json: no stations")
    optional = {"neighborhood": str, "amNetFlow": (int, float), "isAmDestination": bool}
    for i, s in enumerate(doc["stations"]):
        where = f"stations[{i}] ({s.get('name')})"
        missing = [k for k in STATION_FIELDS if k not in s]
        if missing:
            raise ValueError(f"{where}: missing required fields {missing}")
        extra = [k for k in s if k not in STATION_FIELDS and k not in optional]
        if extra:
            raise ValueError(f"{where}: unexpected fields {extra}")
        for k, typ in optional.items():
            if k in s and not isinstance(s[k], typ):
                raise ValueError(f"{where}.{k}: must be {typ}")
        if not (isinstance(s["capacity"], int) and s["capacity"] >= 1):
            raise ValueError(f"{where}: capacity must be a positive integer")
        if not 0 <= s["initialBikes"] <= s["capacity"]:
            raise ValueError(f"{where}: initialBikes outside 0..capacity")
        for k in ("ratesIn", "ratesOut"):
            r = s[k]
            if len(r) != HOURS or not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in r):
                raise ValueError(f"{where}.{k}: expected 24 finite non-negative numbers")
    return doc


def make_doc(records: list[dict], source: dict) -> dict:
    return validate({
        "version": SCHEMA_VERSION,
        "stepMinutes": STEP_MIN,
        "startMinute": START_MINUTE,
        "source": {**source, "generated": dt.date.today().isoformat()},
        "stations": records,
    })


def write_bikes(doc: dict, path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validate(doc), allow_nan=False, separators=(",", ":")))
    return path


# ---------- demo model (no network) ----------

# Hand-placed demo stations: real Bluebikes station areas in the Downtown cluster, positions approximate
# to the nearest block, capacities typical. Replace with --month to use the real station list.
DEMO_STATIONS = [
    # name, lat, lon, capacity, role (d = AM destination, r = AM origin/residential, c = campus/mixed)
    ("South Station - 700 Atlantic Ave", 42.3523, -71.0551, 46, "d"),
    ("Post Office Sq - Congress St at Milk St", 42.3564, -71.0560, 23, "d"),
    ("Federal St at Franklin St", 42.3553, -71.0577, 19, "d"),
    ("Faneuil Hall - Union St at North St", 42.3602, -71.0566, 19, "d"),
    ("Government Center - Cambridge St at Court St", 42.3597, -71.0594, 19, "d"),
    ("Downtown Crossing - Washington St at Summer St", 42.3549, -71.0603, 23, "d"),
    ("Rowes Wharf - Atlantic Ave", 42.3569, -71.0507, 15, "d"),
    ("Seaport Blvd at Sleeper St", 42.3527, -71.0482, 19, "d"),
    ("Copley Square - Dartmouth St at Boylston St", 42.3500, -71.0773, 23, "d"),
    ("Back Bay T Stop - Dartmouth St at Stuart St", 42.3478, -71.0760, 23, "d"),
    ("Prudential Center - Belvidere St", 42.3459, -71.0825, 19, "d"),
    ("Boylston St at Arlington St", 42.3520, -71.0705, 19, "d"),
    ("MGH - Cambridge St at N Grove St", 42.3613, -71.0682, 19, "d"),
    ("North Station - Causeway St", 42.3657, -71.0610, 31, "d"),
    ("Tremont St at West St", 42.3547, -71.0626, 19, "d"),
    ("Chinatown Gate - Beach St at Surface Rd", 42.3513, -71.0594, 19, "d"),
    ("Charles St at Beacon St", 42.3561, -71.0702, 15, "r"),
    ("Charles Circle - Charles St at Cambridge St", 42.3608, -71.0710, 23, "r"),
    ("North End - Hanover St at Salem St", 42.3640, -71.0545, 15, "r"),
    ("Commercial St at Fleet St", 42.3646, -71.0519, 15, "r"),
    ("West End - Lomasney Way at Staniford St", 42.3638, -71.0650, 15, "r"),
    ("Columbus Ave at Massachusetts Ave", 42.3404, -71.0810, 19, "r"),
    ("Tremont St at E Berkeley St", 42.3446, -71.0699, 15, "r"),
    ("Washington St at Waltham St", 42.3425, -71.0690, 15, "r"),
    ("Shawmut Ave at Lenox St", 42.3378, -71.0782, 15, "r"),
    ("Commonwealth Ave at Dartmouth St", 42.3514, -71.0786, 15, "r"),
    ("Beacon St at Massachusetts Ave", 42.3509, -71.0898, 19, "r"),
    ("Newbury St at Hereford St", 42.3487, -71.0866, 19, "r"),
    ("Stuart St at Charles St S (Bay Village)", 42.3496, -71.0680, 15, "r"),
    ("W Broadway at D St", 42.3394, -71.0525, 19, "r"),
    ("Broadway T Stop", 42.3426, -71.0570, 19, "r"),
    ("Union Park St at Washington St", 42.3408, -71.0662, 15, "r"),
    ("Boylston St at Massachusetts Ave", 42.3474, -71.0874, 27, "c"),
    ("Christian Science Plaza - Mass Ave at Westland Ave", 42.3432, -71.0855, 19, "c"),
    ("Forsyth St at Huntington Ave", 42.3393, -71.0907, 23, "c"),
    ("Boston Common - Tremont St at Park St", 42.3567, -71.0623, 23, "c"),
]


def _bump(h: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return np.exp(-0.5 * ((h + 0.5 - mu) / sd) ** 2)


def demo_rates(role: str, capacity: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Commute-shaped weekday profiles, bikes/hour. Peak volume scales with station size."""
    h = np.arange(HOURS, dtype=float)
    am, pm, mid = _bump(h, 8.3, 1.0), _bump(h, 17.6, 1.3), _bump(h, 13.0, 2.8)
    night = np.where((h < 6) | (h >= 23), 0.15, 1.0)
    v = capacity * rng.uniform(0.24, 0.34)
    jitter = lambda: rng.uniform(0.85, 1.15, HOURS)  # noqa: E731
    if role == "d":     # fills in the morning, drains in the evening
        r_in = v * (1.0 * am + 0.30 * pm + 0.35 * mid + 0.05)
        r_out = v * (0.25 * am + 1.0 * pm + 0.35 * mid + 0.05)
    elif role == "r":   # drains in the morning, fills in the evening
        r_in = v * (0.25 * am + 0.95 * pm + 0.30 * mid + 0.05)
        r_out = v * (1.0 * am + 0.28 * pm + 0.30 * mid + 0.05)
    else:               # busy all day, roughly balanced
        r_in = v * (0.6 * am + 0.6 * pm + 0.75 * mid + 0.06)
        r_out = v * (0.55 * am + 0.65 * pm + 0.75 * mid + 0.06)
    return r_in * night * jitter(), r_out * night * jitter()


def demo_doc(seed: int = 4100) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    for k, (name, lat, lon, cap, role) in enumerate(DEMO_STATIONS):
        r_in, r_out = demo_rates(role, cap, rng)
        records.append({
            "id": f"demo-{k:02d}", "name": name, "lat": lat, "lon": lon, "capacity": cap,
            "initialBikes": initial_bikes(cap, r_in, r_out),
            "ratesIn": [round(float(x), 3) for x in r_in],
            "ratesOut": [round(float(x), 3) for x in r_out],
        })
    return make_doc(records, {"kind": "demo", "month": None, "weekdays": None})


# ---------- CLI ----------

def real_doc(month: str, use_status: bool, neighborhoods: list[str] | None = None) -> dict:
    stations = fetch_station_information()
    # Trip files identify stations by GBFS short_name (e.g. "A32000"); fall back to the name.
    for s in stations:
        s["_key"] = str(s.get("short_name") or s["name"])
    trips = fetch_trips(month)
    out, inn, n = hourly_rates(trips)
    known = set(out) | set(inn)
    for s in stations:
        if s["_key"] not in known and s["name"] in known:
            s["_key"] = s["name"]
    volume = {k: float(out.get(k, np.zeros(1)).sum() + inn.get(k, np.zeros(1)).sum()) for k in known}

    if neighborhoods:
        hoods = Neighborhoods.load()
        cluster = select_by_neighborhood(stations, volume, neighborhoods, hoods)
        if not cluster:
            raise ValueError(f"no stations with trips fell inside {neighborhoods}")
    else:
        cluster = select_cluster(stations, volume)
    if not cluster:
        raise ValueError("no stations matched the trip file: check that the month exists and ids line up")
    status = None
    if use_status:
        status = {s["station_id"]: int(s["num_bikes_available"]) for s in fetch_station_status()}
    source = {"kind": "bluebikes-trips", "month": month, "weekdays": n}
    if neighborhoods:
        source["neighborhoods"] = neighborhoods
    return make_doc(build_records(cluster, out, inn, status), source)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--month", help="trip-history month, YYYYMM (e.g. 202509)")
    g.add_argument("--demo", action="store_true", help="write the demo model (no network)")
    p.add_argument("--status", action="store_true", help="start counts from the live GBFS station_status feed")
    p.add_argument("--nu", action="store_true",
                   help="scope to the Northeastern-area neighborhoods (Fenway/Kenmore, Back Bay, "
                        "Mission Hill, South End, Roxbury, Jamaica Plain)")
    p.add_argument("--neighborhoods", nargs="+", metavar="NAME",
                   help="scope to these neighborhood names (overrides --nu)")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    a = p.parse_args(argv)
    hoods = a.neighborhoods or (NU_NEIGHBORHOODS if a.nu else None)
    if a.demo and hoods:
        p.error("--demo has no station coordinates to filter; use --month with --nu/--neighborhoods")
    doc = demo_doc() if a.demo else real_doc(a.month, a.status, hoods)
    path = write_bikes(doc, a.out)
    print(f"wrote {path} ({len(doc['stations'])} stations, source {doc['source']['kind']})")


if __name__ == "__main__":
    main()
