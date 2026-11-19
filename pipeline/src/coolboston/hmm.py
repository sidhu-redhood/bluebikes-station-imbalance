"""HMM for the bike count at one station, used to correct demand censoring.

Hidden state: true bikes docked, 0..C, every 10 minutes.
Transitions: the hourly 10-minute matrices from bike_markov.py.
Observations: the poller's bike count. It is right with probability 1 - eps,
otherwise off by one bike either way. A missed poll is None.
"""

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from coolboston.bike_markov import HOURS, STEP_MIN, StationModel, _hour_at, hourly_transitions, stationary_occupancy

EPS = 0.05
MIN_AVAILABLE = 0.2


# ---------- emissions ----------

def emission_matrix(capacity, eps=EPS):
    n = capacity + 1
    B = np.zeros((n, n))
    for b in range(n):
        neighbors = [o for o in (b - 1, b + 1) if 0 <= o <= capacity]
        B[b, b] = 1 - eps
        for o in neighbors:
            B[b, o] = eps / len(neighbors)
    return B


def obs_likelihood(o, B):
    if o is None:
        return np.ones(B.shape[0])
    o = min(max(o, 0), B.shape[0] - 1)
    return B[:, o]


# ---------- forward / backward ----------

def forward(pi, mats, obs, B, start_min):
    T, n = len(obs), len(pi)
    alpha = np.zeros((T, n))
    scale = np.zeros(T)
    a = pi * obs_likelihood(obs[0], B)
    for t in range(T):
        if t > 0:
            A = mats[_hour_at(start_min, t - 1, STEP_MIN)]
            a = (alpha[t - 1] @ A) * obs_likelihood(obs[t], B)
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    return alpha, scale


def backward(mats, obs, B, start_min, scale):
    T, n = len(obs), B.shape[0]
    beta = np.ones((T, n))
    for t in range(T - 2, -1, -1):
        A = mats[_hour_at(start_min, t, STEP_MIN)]
        beta[t] = A @ (obs_likelihood(obs[t + 1], B) * beta[t + 1]) / scale[t + 1]
    return beta


def posterior(model, obs, start_min=0, eps=EPS):
    mats = hourly_transitions(model)
    pi = stationary_occupancy(model, _hour_at(start_min, 0, STEP_MIN))
    B = emission_matrix(model.capacity, eps)
    alpha, scale = forward(pi, mats, obs, B, start_min)
    beta = backward(mats, obs, B, start_min, scale)
    gamma = alpha * beta
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    loglik = np.log(scale).sum()
    return gamma, loglik


# ---------- demand censoring ----------

def hourly_empty_full(gamma, start_min=0):
    empty, full, count = np.zeros(HOURS), np.zeros(HOURS), np.zeros(HOURS)
    for t in range(len(gamma)):
        h = _hour_at(start_min, t, STEP_MIN)
        empty[h] += gamma[t, 0]
        full[h] += gamma[t, -1]
        count[h] += 1
    count[count == 0] = np.nan
    return empty / count, full / count


def corrected_rates(model, p_empty, p_full):
    p_empty = np.nan_to_num(p_empty)
    p_full = np.nan_to_num(p_full)
    rates_out = model.rates_out / np.maximum(1 - p_empty, MIN_AVAILABLE)
    rates_in = model.rates_in / np.maximum(1 - p_full, MIN_AVAILABLE)
    return rates_in, rates_out


# ---------- loading data ----------

def boston_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        return timezone(timedelta(hours=-5))


def load_model(bikes_json, station):
    data = json.loads(Path(bikes_json).read_text())
    for s in data["stations"]:
        if s["id"] == station:
            return StationModel(s["capacity"], np.array(s["ratesIn"]), np.array(s["ratesOut"]))
    raise KeyError(station + " not in " + str(bikes_json))


def load_day(db_path, station, day):
    start = int(datetime(day.year, day.month, day.day, tzinfo=boston_tz()).timestamp())
    step = STEP_MIN * 60
    slots = 24 * 60 // STEP_MIN
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    rows = conn.execute(
        """SELECT s.fetched_at, s.num_bikes_available
             FROM station_status s JOIN stations st ON st.station_id = s.station_id
            WHERE st.short_name = ? AND s.fetched_at >= ? AND s.fetched_at < ?
              AND s.num_bikes_available IS NOT NULL
            ORDER BY s.fetched_at""",
        (station, start, start + slots * step),
    ).fetchall()
    conn.close()
    obs = [None] * slots
    for fetched_at, bikes in rows:
        obs[(fetched_at - start) // step] = bikes
    return obs


# ---------- main ----------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--station", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--bikes-json", default="web/data/bikes.json")
    args = p.parse_args(argv)

    model = load_model(args.bikes_json, args.station)
    first = date.fromisoformat(args.start)
    empty_sum, full_sum, n_days = np.zeros(HOURS), np.zeros(HOURS), 0
    for i in range(args.days):
        obs = load_day(args.db, args.station, first + timedelta(days=i))
        if all(o is None for o in obs):
            continue
        gamma, _ = posterior(model, obs)
        e, f = hourly_empty_full(gamma)
        empty_sum += e
        full_sum += f
        n_days += 1
    if n_days == 0:
        print(args.station + ": no poller data for those days")
        return 1

    p_empty, p_full = empty_sum / n_days, full_sum / n_days
    rates_in, rates_out = corrected_rates(model, p_empty, p_full)
    print(f"{args.station}: {n_days} days of poller data")
    print("hour  P(empty)  P(full)   out -> out*    in -> in*")
    for h in range(HOURS):
        print(f"{h:4d}  {p_empty[h]:8.3f}  {p_full[h]:7.3f}  {model.rates_out[h]:5.1f} -> {rates_out[h]:5.1f}"
              f"  {model.rates_in[h]:5.1f} -> {rates_in[h]:5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
