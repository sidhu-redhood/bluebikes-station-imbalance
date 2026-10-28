# 🚲 Bluebikes Station Imbalance

CS5100 Foundations of AI, Northeastern University
Predicting where Boston's Bluebikes stations run empty or full, and learning how an operator should respond.

## Overview

Commutes drain residential stations and fill downtown and campus ones, then flip in the evening. We forecast demand, estimate hidden bike counts, and compare responses (vans, rider incentives, app nudges) in a simulator on a cluster of 30 to 60 stations. All methods are implemented from scratch.

## Setup

```bash
git clone https://github.com/sidhu-redhood/bluebikes-station-imbalance.git
cd bluebikes-station-imbalance
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS the command is `python3`; inside the activated venv, `python` and `pip` work as usual. Run `source .venv/bin/activate` each time you open a new terminal.

## Data

Data files are not stored in this repo (`data/` is in `.gitignore`). Each person downloads them into `data/raw/`:

```bash
mkdir -p data/raw
```

| File | Where to get it | Save as |
|---|---|---|
| Trip history (one month) | [bluebikes.com/system-data](https://bluebikes.com/system-data): download `YYYYMM-bluebikes-tripdata.zip` and unzip | `data/raw/202608-bluebikes-tripdata.csv` |
| Station list (dock capacity) | Same page, current station list CSV | `data/raw/current_bluebikes_stations.csv` |
| Daily weather | [NOAA Climate Data Online](https://www.ncei.noaa.gov/cdo-web/): Daily Summaries, station Boston Logan International Airport (USW00014739), same month, CSV | `data/raw/noaa_boston_daily_aug.csv` |

**Do not open and save the trip CSV in Excel.** Excel rewrites the timestamps and deletes the date and hour. Inspect files in Python (`df.head()`) instead.

## Reproducing Results

**1. Data loading.** Builds the station cluster around Northeastern and filters trips and weather to it:

```bash
python code/load_data.py
```

This writes three files to `data/processed/`:

- `neu_stations.csv`: cluster stations with ID, name, location, dock capacity, and distance to NEU
- `neu_trips_YYYY-MM.csv`: trips that start or end in the cluster
- `weather_YYYY-MM.csv`: daily weather for the month

The month, cluster radius, and file names are set in the CONFIG block at the top of `code/load_data.py`.

Later steps: coming soon.

## Code Organization

```
bluebikes-station-imbalance/
├── code/
│   └── load_data.py      # load trips, stations, weather; build the NEU cluster
├── data/                 # not tracked by git
│   ├── raw/              # downloaded files (see Data)
│   └── processed/        # outputs of load_data.py
├── requirements.txt
└── README.md
```

## Team

Sidharthan Rajarajan, Muhammad Saad Malik, Nishant Naravarajula, Samarth Srinivas