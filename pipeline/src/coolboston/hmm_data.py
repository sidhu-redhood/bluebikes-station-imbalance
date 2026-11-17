"""Inputs for the station HMM (station_hmm.py): trips and GBFS polls, cut into 10-minute slots.

What the HMM needs from each slot t and station s is small, so it is precomputed here once:

  net[t, s]   bikes in minus bikes out during the slot (from trips)
  lo[t, s]    lowest point the count reaches during the slot, relative to the slot start (<= 0)
  hi[t, s]    highest point, relative to the slot start (>= 0)
  dep, arr    number of pickups and returns in the slot

A station that starts the slot with b bikes can only have produced these trips if b + lo >= 0 (every
pickup found a bike) and b + hi <= capacity (every return found a dock).

  segments    the slot split at each trip: (slot, station, offset from slot start, minutes). With b
              bikes at the start the count during a segment is b + offset, so "minutes with a bike" and
              "minutes with a free dock" follow for every b. This is what the demand correction needs.
  polls       GBFS snapshots: (slot, station, offset of trips between slot start and the poll, bikes seen)

Times. Trip files are local wall-clock time (America/New_York) with no zone. The poller stores UTC epoch
seconds. Everything here is converted to local naive time, then to minutes since the window start.

Station ids are the trip-file ids ("B32062"). In the poller's database these are stations.short_name;
station_status.station_id is the GBFS id, so polls are joined through the stations table.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from coolboston.bike_markov import STEP_MIN

LOCAL_TZ = "America/New_York"


@dataclass
class SlotData:
    station_ids: list[str]
    names: list[str]
    capacity: np.ndarray      # (S,) docks
    start: pd.Timestamp       # local time at the start of slot 0
    step_min: int
    hour: np.ndarray          # (T,) clock hour of each slot's start
    weekend: np.ndarray       # (T,) True on Saturday and Sunday
    day: np.ndarray           # (T,) calendar day, 0 = the day of the first slot
    net: np.ndarray           # (T, S)
    lo: np.ndarray            # (T, S)
    hi: np.ndarray            # (T, S)
    dep: np.ndarray           # (T, S)
    arr: np.ndarray           # (T, S)
    seg_t: np.ndarray         # segments of constant count inside a slot
    seg_s: np.ndarray
    seg_level: np.ndarray
    seg_min: np.ndarray
    poll_t: np.ndarray        # polls
    poll_s: np.ndarray
    poll_offset: np.ndarray
    poll_y: np.ndarray

    @property
    def n_slots(self) -> int:
        return len(self.hour)

    @property
    def n_stations(self) -> int:
        return len(self.station_ids)

    @property
    def n_polls(self) -> int:
        return len(self.poll_t)

    def slot_times(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=self.n_slots, freq=f"{self.step_min}min")


# ---------------------------------------------------------------- slots from event lists (pure)

def build_slots(events: list[tuple[np.ndarray, np.ndarray]], capacity, start: pd.Timestamp, n_slots: int,
                step_min: int = STEP_MIN, polls: list[tuple[np.ndarray, np.ndarray]] | None = None,
                station_ids: list[str] | None = None, names: list[str] | None = None) -> SlotData:
    """events[s] = (minutes since `start`, +1 return / -1 pickup). polls[s] = (minutes, bikes seen).
    Events outside [0, n_slots * step_min) are dropped. Ties: returns before pickups (the order
    that needs the fewest bikes)."""
    capacity = np.asarray(capacity, dtype=int)
    S, T = len(events), int(n_slots)
    if len(capacity) != S:
        raise ValueError(f"{S} event lists but {len(capacity)} capacities")
    end = T * step_min
    net = np.zeros((T, S), dtype=int)
    lo = np.zeros((T, S), dtype=int)
    hi = np.zeros((T, S), dtype=int)
    dep = np.zeros((T, S), dtype=int)
    arr = np.zeros((T, S), dtype=int)
    seg = {k: [] for k in ("t", "s", "level", "min")}
    pol = {k: [] for k in ("t", "s", "offset", "y")}
    all_slots = np.arange(T)

    for s, (times, delta) in enumerate(events):
        times = np.asarray(times, dtype=float)
        delta = np.asarray(delta, dtype=int)
        keep = (times >= 0) & (times < end)
        times, delta = times[keep], delta[keep]
        order = np.lexsort((-delta, times))  # by time, returns first on ties
        times, delta = times[order], delta[order]
        slot = (times // step_min).astype(int)
        g0 = np.concatenate([[0], np.cumsum(delta)])  # g0[k] = net of the first k events

        # Slot-start segment for every slot: whole slot when no trips, else up to the first trip.
        first_time = np.full(T, float(step_min))
        if len(times):
            uniq, first = np.unique(slot, return_index=True)
            last = np.r_[first[1:], len(times)] - 1
            prefix = g0[1:] - g0[first.repeat(np.diff(np.r_[first, len(times)]))]
            net[uniq, s] = prefix[last]
            lo[uniq, s] = np.minimum(np.minimum.reduceat(prefix, first), 0)
            hi[uniq, s] = np.maximum(np.maximum.reduceat(prefix, first), 0)
            np.add.at(dep[:, s], slot[delta < 0], 1)
            np.add.at(arr[:, s], slot[delta > 0], 1)
            first_time[uniq] = times[first] - uniq * step_min
            # One segment after each trip, lasting until the next trip in the slot or the slot end.
            nxt = np.r_[times[1:], np.inf]
            same = np.r_[slot[1:] == slot[:-1], False]
            seg_end = np.where(same, nxt, (slot + 1) * step_min)
            seg["t"].append(slot)
            seg["s"].append(np.full(len(slot), s))
            seg["level"].append(prefix)
            seg["min"].append(seg_end - times)
        seg["t"].append(all_slots)
        seg["s"].append(np.full(T, s))
        seg["level"].append(np.zeros(T, dtype=int))
        seg["min"].append(first_time)

        if polls is not None and len(polls[s][0]):
            pt = np.asarray(polls[s][0], dtype=float)
            py = np.asarray(polls[s][1], dtype=int)
            ok = (pt >= 0) & (pt < end)
            pt, py = pt[ok], py[ok]
            ps = (pt // step_min).astype(int)
            # Trips strictly before the poll, minus trips before the slot start.
            off = g0[np.searchsorted(times, pt, side="left")] - g0[np.searchsorted(times, ps * step_min, side="left")]
            pol["t"].append(ps)
            pol["s"].append(np.full(len(ps), s))
            pol["offset"].append(off)
            pol["y"].append(py)

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    stamps = pd.date_range(start, periods=T, freq=f"{step_min}min")
    return SlotData(
        station_ids=list(station_ids) if station_ids is not None else [str(i) for i in range(S)],
        names=list(names) if names is not None else [str(i) for i in range(S)],
        capacity=capacity, start=pd.Timestamp(start), step_min=step_min,
        hour=np.asarray(stamps.hour), weekend=np.asarray(stamps.dayofweek >= 5),
        day=np.asarray((stamps.normalize() - stamps[0].normalize()).days),
        net=net, lo=lo, hi=hi, dep=dep, arr=arr,
        seg_t=cat(seg["t"], int), seg_s=cat(seg["s"], int), seg_level=cat(seg["level"], int),
        seg_min=cat(seg["min"], float),
        poll_t=cat(pol["t"], int), poll_s=cat(pol["s"], int), poll_offset=cat(pol["offset"], int),
        poll_y=cat(pol["y"], int),
    )


# ---------------------------------------------------------------- trips and stations (files)

def load_trips_csv(path: Path) -> pd.DataFrame:
    """A Bluebikes trip CSV (2024+ columns) or Saad's processed neu_trips_YYYY-MM.csv."""
    df = pd.read_csv(path, dtype={"start_station_id": str, "end_station_id": str}, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    sample = str(df["started_at"].dropna().iloc[0])
    if "-" not in sample and "/" not in sample:
        raise ValueError(f"started_at looks like {sample!r} with no date. Use the original CSV from the "
                         "Bluebikes zip, not a copy saved in Excel.")
    for c in ("started_at", "ended_at"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("start_station_id", "end_station_id"):
        df[c] = df[c].str.strip()
    return df


def load_stations_csv(path: Path) -> pd.DataFrame:
    """Saad's neu_stations.csv: station_id, name, lat, lon, capacity (others are kept but unused)."""
    st = pd.read_csv(path, dtype={"station_id": str})
    missing = st["capacity"].isna()
    if missing.any():
        print(f"[hmm] skipping {int(missing.sum())} stations with no capacity: {st.loc[missing, 'name'].tolist()}")
    st = st[~missing].copy()
    st["capacity"] = st["capacity"].astype(int)
    return st.reset_index(drop=True)


def events_from_trips(trips: pd.DataFrame, station_ids: list[str], start: pd.Timestamp,
                      end: pd.Timestamp) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per station: (minutes since start, delta). A pickup is at started_at, a return at ended_at."""
    out = []
    for sid in station_ids:
        d = trips.loc[trips["start_station_id"] == sid, "started_at"]
        a = trips.loc[trips["end_station_id"] == sid, "ended_at"]
        ts = pd.concat([d, a]).to_numpy()
        delta = np.r_[np.full(len(d), -1), np.full(len(a), 1)]
        ok = ~pd.isna(ts)
        ts, delta = ts[ok], delta[ok]
        mins = (pd.DatetimeIndex(ts) - start).total_seconds().to_numpy() / 60.0
        keep = (mins >= 0) & (mins < (end - start).total_seconds() / 60.0)
        out.append((mins[keep], delta[keep]))
    return out


# ---------------------------------------------------------------- polls (SQLite from polling/poll.py)

_POLL_SQL = """
SELECT st.short_name AS station_id,
       COALESCE(s.last_reported, s.fetched_at) AS t_utc,
       s.num_bikes_available AS bikes
FROM station_status AS s
JOIN stations AS st ON st.station_id = s.station_id
WHERE s.is_installed = 1 AND s.is_renting = 1 AND s.num_bikes_available IS NOT NULL
"""


def load_polls(db_path: Path, station_ids: list[str], start: pd.Timestamp, end: pd.Timestamp,
               tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Snapshots for our stations inside [start, end), local time. One row per (station, report time).
    The station's own last_reported time is used when present: the count is as of that moment."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(_POLL_SQL, conn)
    df = df[df["station_id"].isin(set(station_ids))].copy()
    df["ts"] = pd.to_datetime(df["t_utc"], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df = df[(df["ts"] >= start) & (df["ts"] < end)]
    df = df.drop_duplicates(["station_id", "ts"]).sort_values(["station_id", "ts"])
    return df[["station_id", "ts", "bikes"]].reset_index(drop=True)


def polls_by_station(polls: pd.DataFrame, station_ids: list[str],
                     start: pd.Timestamp) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for sid in station_ids:
        p = polls[polls["station_id"] == sid]
        mins = (pd.DatetimeIndex(p["ts"]) - start).total_seconds().to_numpy() / 60.0
        out.append((mins, p["bikes"].to_numpy(dtype=int)))
    return out


def month_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """'2026-08' -> (2026-08-01 00:00, 2026-09-01 00:00), local time."""
    start = pd.Timestamp(f"{month}-01")
    return start, start + pd.offsets.MonthBegin(1)


def load_month(trips_path: Path, stations_path: Path, month: str, polls_db: Path | None = None,
               step_min: int = STEP_MIN) -> SlotData:
    start, end = month_window(month)
    stations = load_stations_csv(stations_path)
    ids = stations["station_id"].tolist()
    trips = load_trips_csv(trips_path)
    events = events_from_trips(trips, ids, start, end)
    polls = None
    if polls_db is not None:
        p = load_polls(polls_db, ids, start, end)
        print(f"[hmm] {len(p):,} poll snapshots for {p['station_id'].nunique()} stations in {month}")
        polls = polls_by_station(p, ids, start)
    n_slots = int((end - start).total_seconds() // 60 // step_min)
    return build_slots(events, stations["capacity"].to_numpy(), start, n_slots, step_min, polls,
                       station_ids=ids, names=stations["name"].tolist())
