"""Build the 1-hour-ahead PM2.5 training table from station_hour.csv.

Robust version: handles a BOM, different separators, different capitalisation of
column names, and prints a clear message if the file is not the expected dataset.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/station_hour.csv")
OUT = Path("data/processed/train_pm25_1h.csv.gz")
OUT.parent.mkdir(parents=True, exist_ok=True)

HORIZON = 1        # hours ahead to predict
MAX_FILL_H = 2     # interpolate gaps of at most this many hours
PM_CAP = 500       # top of the Indian AQI scale; larger values are treated as sensor errors

# normalised column name -> our name
ALIASES = {
    "stationid": "station", "station": "station", "stationcode": "station",
    "siteid": "station", "city": "station",
    "datetime": "time", "date": "time", "timestamp": "time", "time": "time",
    "pm25": "pm25", "no2": "no2", "co": "co",
}


def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# ---- 1. Inspect the file ----
if not RAW.exists():
    sys.exit(f"File not found: {RAW.resolve()}\nPut the dataset there and name it station_hour.csv")

with open(RAW, "r", encoding="utf-8-sig", errors="replace") as f:
    header_line = f.readline()
sep = max([",", ";", "\t", "|"], key=header_line.count)
columns = [c.strip() for c in header_line.strip().split(sep)]
print("Detected separator:", repr(sep))
print("Columns found:", columns)

mapping = {}
for c in columns:
    key = ALIASES.get(norm(c))
    if key and key not in mapping.values():
        mapping[c] = key
missing = [k for k in ("station", "time", "pm25") if k not in mapping.values()]
if missing:
    sys.exit(f"\nThis file does not look like the expected dataset. Missing: {missing}\n"
             f"Expected columns like: StationId, Datetime, PM2.5, NO2, CO.\n"
             f"Copy the correct file (station_hour.csv) into data/raw/ and run again.")

# ---- 2. Load only the needed columns ----
print("Loading", RAW)
df = pd.read_csv(RAW, sep=sep, encoding="utf-8-sig", low_memory=False,
                 usecols=lambda c: c.strip() in mapping)
df.columns = [mapping[c.strip()] for c in df.columns]
for g in ("no2", "co"):
    if g not in df.columns:
        df[g] = np.nan
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["time"])
print("Raw rows:", len(df), "| stations:", df["station"].nunique())

# ---- 3. Check that the data really is hourly ----
sample = df[df["station"] == df["station"].iloc[0]].sort_values("time")
step = sample["time"].diff().median()
print("Typical time step:", step)
if step > pd.Timedelta(minutes=90):
    sys.exit("\nThis data is not hourly (typical step is %s).\n"
             "It looks like a daily file such as city_day.csv. Use station_hour.csv." % step)

# ---- 4. Cleaning ----
for c in ("pm25", "no2", "co"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df.loc[df["pm25"] >= 999, "pm25"] = np.nan          # 999/1000 are sentinel / error values
df["pm25"] = df["pm25"].clip(upper=PM_CAP)
for g in ("no2", "co"):
    if df[g].notna().any():
        df[g] = df[g].clip(upper=df[g].quantile(0.999))   # remove absurd spikes

# ---- 5. Features per station ----
parts = []
for station, d in df.groupby("station"):
    d = d.drop_duplicates("time").set_index("time").sort_index()
    d = d[["pm25", "no2", "co"]].asfreq("1h")        # strict hourly grid; gaps become NaN
    d = d.interpolate(limit=MAX_FILL_H, limit_area="inside")

    p = d["pm25"]
    f = pd.DataFrame(index=d.index)
    f["pm25"] = p
    for k in (1, 2, 3, 6, 12, 24):
        f[f"pm25_lag{k}"] = p.shift(k)
    f["pm25_delta1"] = p - p.shift(1)
    f["pm25_delta3"] = p - p.shift(3)
    f["pm25_roll3"] = p.rolling(3).mean()
    f["pm25_roll6"] = p.rolling(6).mean()
    f["pm25_roll24"] = p.rolling(24).mean()
    f["pm25_std6"] = p.rolling(6).std()
    f["hour_sin"] = np.sin(2 * np.pi * f.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.index.hour / 24)
    f["month_sin"] = np.sin(2 * np.pi * f.index.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * f.index.month / 12)
    f["no2"] = d["no2"]
    f["co"] = d["co"]
    f["pm25_future"] = p.shift(-HORIZON)             # the target
    f["station"] = station

    pm_cols = [c for c in f.columns if c.startswith("pm25")]
    parts.append(f.dropna(subset=pm_cols))           # gas columns may stay NaN

if not parts:
    sys.exit("No usable rows after cleaning. Check the dataset.")
out = pd.concat(parts)
out.index.name = "time"
print("Final rows:", len(out), "| stations:", out["station"].nunique())
print("Gas columns missing: no2 %.0f%%, co %.0f%%" %
      (out["no2"].isna().mean() * 100, out["co"].isna().mean() * 100))
print(out[["pm25", "pm25_future"]].describe().round(1))
out.astype({c: "float32" for c in out.columns if c != "station"}).to_csv(OUT)
print("Saved", OUT)