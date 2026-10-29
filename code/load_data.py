"""
Tasks 1-3: load one month of Bluebikes data around Northeastern University.

  Task 1: read the trip CSV and keep trips touching the NEU station cluster
  Task 2: read the station list (location + dock capacity)
  Task 3: read NOAA weather for the same month

Station IDs: 2024+ trips use codes like "A32042", but the station list only
has the old numeric IDs ("to match to historic system data"). So the station
table is built from the trips themselves, and dock capacity is attached from
the station list by matching names (falling back to the nearest station).

Expected folder layout (change the paths in CONFIG if yours differ):
  Blue Bikes Project/
    code/load_data.py
    data/raw/202608-bluebikes-tripdata.csv
    data/raw/current_bluebikes_stations.csv
    data/raw/noaa_boston_daily_aug.csv
Outputs go to data/processed/.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- CONFIG
MONTH = "2026-08"  # the month you are analyzing (YYYY-MM)

# Paths are relative to the project folder (the parent of code/), so the
# script works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed"

TRIPS_FILE = RAW / f"{MONTH.replace('-', '')}-bluebikes-tripdata.csv"
STATIONS_FILE = RAW / "current_bluebikes_stations.csv"
WEATHER_FILE = RAW / "noaa_boston_daily_aug.csv"

NEU_LAT, NEU_LON = 42.3398, -71.0892  # Northeastern, roughly Centennial Common
RADIUS_KM = 1.2  # tune this until you get 30-60 stations
NEAREST_MATCH_KM = 0.1  # max distance for matching a renamed station by location


# ---------------------------------------------------------------- helpers
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on numpy arrays / pandas Series."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def pick_column(df, candidates):
    """
    Return the first column in df that matches one of the candidates, either
    exactly or as a prefix (so "station id" matches
    "station id (to match to historic system data)").
    """
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        for col in df.columns:
            if col.startswith(c):
                return col
    raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")


def normalize_name(s):
    """Lowercase and collapse whitespace so small formatting differences still match."""
    return s.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


# ---------------------------------------------------------------- task 1
def load_trips(path=TRIPS_FILE):
    """Trip files from 2024 onward (ride_id, started_at, start_station_id, ...)."""
    df = pd.read_csv(path, dtype={"start_station_id": str, "end_station_id": str},
                     low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=["ride_id"])  # drop fully blank rows

    # Timestamps must include a date, e.g. "2026-08-01 07:45:12.345".
    # A file re-saved in Excel can show only "mm:ss.0", which loses the date.
    sample = str(df["started_at"].iloc[0])
    if "-" not in sample and "/" not in sample:
        raise ValueError(f"started_at looks like {sample!r} with no date. Use the original "
                         "CSV from the Bluebikes zip, not a copy saved in Excel.")

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
    df["start_station_id"] = df["start_station_id"].str.strip()
    df["end_station_id"] = df["end_station_id"].str.strip()
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

    n_before = len(df)
    df = df.dropna(subset=["started_at", "ended_at"])
    df = df[df["duration_min"] > 0]
    print(f"[trips] {n_before:,} rows read, {len(df):,} kept after removing bad timestamps")
    return df


def filter_trips_to_cluster(trips, cluster):
    """
    Keep a trip if it starts OR ends in the cluster: departures from cluster
    stations take bikes away, arrivals bring them in, and both matter for
    whether a station runs empty or full.
    """
    ids = set(cluster["station_id"])
    starts_in = trips["start_station_id"].isin(ids)
    ends_in = trips["end_station_id"].isin(ids)
    keep = starts_in | ends_in
    out = trips[keep].copy()
    out["starts_in_cluster"] = starts_in[keep]
    out["ends_in_cluster"] = ends_in[keep]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- task 2
def stations_from_trips(trips):
    """
    One row per station ID seen in the trips. Name = most common name for that
    ID; location = median coordinates (e-bike GPS points vary slightly).
    """
    parts = []
    for side in ["start", "end"]:
        p = trips[[f"{side}_station_id", f"{side}_station_name", f"{side}_lat", f"{side}_lng"]]
        p.columns = ["station_id", "name", "lat", "lon"]
        parts.append(p)
    allp = pd.concat(parts).dropna(subset=["station_id"])

    g = allp.groupby("station_id")
    st = pd.DataFrame({
        "name": g["name"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None),
        "lat": g["lat"].median(),
        "lon": g["lon"].median(),
        "n_trip_events": g.size(),
    }).reset_index()
    return st


def load_station_list(path=STATIONS_FILE):
    """
    Bluebikes' station CSV, used here for dock capacity. Some versions have a
    'Last Updated' line above the real header, so we search for the header row.
    """
    with open(path, encoding="utf-8-sig") as f:
        first_lines = [next(f, "") for _ in range(10)]
    header_row = next(i for i, line in enumerate(first_lines) if "lat" in line.lower())

    df = pd.read_csv(path, skiprows=header_row, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    sl = pd.DataFrame({
        "list_name": df[pick_column(df, ["name", "station name"])].str.strip(),
        "list_lat": pd.to_numeric(df[pick_column(df, ["lat", "latitude"])], errors="coerce"),
        "list_lon": pd.to_numeric(df[pick_column(df, ["long", "lon", "lng", "longitude"])],
                                  errors="coerce"),
        "capacity": pd.to_numeric(df[pick_column(df, ["total docks", "total_docks", "capacity"])],
                                  errors="coerce"),
        "legacy_id": df[pick_column(df, ["station id", "number"])].str.strip(),
    })
    for col, new in [("municipality", "municipality"), ("district", "municipality"),
                     ("seasonal status", "seasonal_status")]:
        if col in df.columns and new not in sl.columns:
            sl[new] = df[col].str.strip()
    return sl.dropna(subset=["list_lat", "list_lon"]).reset_index(drop=True)


def build_stations(trips, station_list):
    """Trip-based station table with capacity attached from the station list."""
    st = stations_from_trips(trips)
    sl = station_list.copy()
    sl["key"] = normalize_name(sl["list_name"])
    sl = sl.drop_duplicates("key")
    st["key"] = normalize_name(st["name"].fillna(""))

    # 1) match by name
    merged = st.merge(sl, on="key", how="left")
    merged["match"] = np.where(merged["capacity"].notna(), "name", None)

    # 2) unmatched: nearest station in the list, if close enough
    info_cols = [c for c in sl.columns if c != "key"]
    for i in merged.index[merged["match"].isna()]:
        d = haversine_km(merged.at[i, "lat"], merged.at[i, "lon"],
                         sl["list_lat"].values, sl["list_lon"].values)
        j = int(np.argmin(d))
        if d[j] <= NEAREST_MATCH_KM:
            for c in info_cols:
                merged.at[i, c] = sl.iloc[j][c]
            merged.at[i, "match"] = "nearest"

    merged = merged.drop(columns="key")
    merged["dist_to_neu_km"] = haversine_km(NEU_LAT, NEU_LON, merged["lat"], merged["lon"])
    counts = merged["match"].value_counts(dropna=False).to_dict()
    print(f"[stations] {len(merged)} stations in trips; capacity matched: {counts}")
    return merged.sort_values("dist_to_neu_km").reset_index(drop=True)


def neu_cluster(stations, radius_km=RADIUS_KM):
    return stations[stations["dist_to_neu_km"] <= radius_km].reset_index(drop=True)


# ---------------------------------------------------------------- task 3
def load_weather(path=WEATHER_FILE, month=MONTH):
    """
    NOAA Climate Data Online daily summaries (GHCN-Daily), e.g. Boston Logan
    station USW00014739. Units depend on what you chose when downloading
    (Standard = inches / deg F, Metric = mm / deg C).
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()
    df["DATE"] = pd.to_datetime(df["DATE"])

    keep = [c for c in ["DATE", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "TAVG", "AWND"]
            if c in df.columns]
    df = df[keep]
    df = df[df["DATE"].dt.strftime("%Y-%m") == month].sort_values("DATE")

    if df.empty:
        raise ValueError(f"No weather rows for {month}. Check the file's date range.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- main
def check_files_exist():
    missing = [p for p in (TRIPS_FILE, STATIONS_FILE, WEATHER_FILE) if not p.exists()]
    if missing:
        found = sorted(f.name for f in RAW.glob("*")) if RAW.exists() else "folder does not exist"
        raise FileNotFoundError(
            "Missing input files:\n  " + "\n  ".join(str(p) for p in missing)
            + f"\nFiles currently in {RAW}: {found}")


def main():
    check_files_exist()
    OUT.mkdir(parents=True, exist_ok=True)

    trips = load_trips()
    stations = build_stations(trips, load_station_list())
    cluster = neu_cluster(stations)
    print(f"[cluster] {len(cluster)} stations within {RADIUS_KM} km of NEU")
    print(cluster[["station_id", "name", "capacity", "match", "dist_to_neu_km"]]
          .to_string(index=False))
    if cluster["capacity"].isna().any():
        print("[warn] some cluster stations have no capacity; check them by hand")

    neu_trips = filter_trips_to_cluster(trips, cluster)
    print(f"[trips] {len(neu_trips):,} trips touch the NEU cluster "
          f"({neu_trips['starts_in_cluster'].sum():,} departures, "
          f"{neu_trips['ends_in_cluster'].sum():,} arrivals)")

    weather = load_weather()
    print(f"[weather] {len(weather)} days loaded for {MONTH}")
    print(weather.head().to_string(index=False))

    cluster.to_csv(OUT / "neu_stations.csv", index=False)
    neu_trips.to_csv(OUT / f"neu_trips_{MONTH}.csv", index=False)
    weather.to_csv(OUT / f"weather_{MONTH}.csv", index=False)
    print(f"[done] saved to {OUT}/")


if __name__ == "__main__":
    main()