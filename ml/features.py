"""Feature builder shared by training and the live backend.

The live system must compute features exactly like ml/preprocess.py does,
otherwise the model receives different inputs than it was trained on.
"""
import numpy as np

BASE = ["pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6", "pm25_lag12",
        "pm25_lag24", "pm25_delta1", "pm25_delta3", "pm25_roll3", "pm25_roll6",
        "pm25_roll24", "pm25_std6"]
TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
MIN_HISTORY = 25          # current hour + 24 earlier hours


def hourly_features(history, ts):
    """history: hourly PM2.5 values, oldest first, last value = current hour.
    ts: datetime of the current hour. Returns a dict, or None if history is too short."""
    p = np.asarray(history, dtype=float)
    if len(p) < MIN_HISTORY or np.isnan(p[-MIN_HISTORY:]).any():
        return None
    f = {"pm25": p[-1]}
    for k in (1, 2, 3, 6, 12, 24):
        f[f"pm25_lag{k}"] = p[-1 - k]
    f["pm25_delta1"] = p[-1] - p[-2]
    f["pm25_delta3"] = p[-1] - p[-4]
    f["pm25_roll3"] = p[-3:].mean()
    f["pm25_roll6"] = p[-6:].mean()
    f["pm25_roll24"] = p[-24:].mean()
    f["pm25_std6"] = p[-6:].std(ddof=1)
    f["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    f["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
    f["month_cos"] = np.cos(2 * np.pi * ts.month / 12)
    return f