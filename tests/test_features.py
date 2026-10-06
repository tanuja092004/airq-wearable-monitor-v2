"""Check that live-style features match the training features exactly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from features import BASE, TIME, hourly_features  # noqa: E402

df = pd.read_csv("data/processed/train_pm25_1h.csv.gz", index_col=0, parse_dates=True)
cols = BASE + TIME
checked, worst = 0, 0.0
for station, d in df.groupby("station"):
    d = d.sort_index()
    gaps = d.index.to_series().diff() != pd.Timedelta(hours=1)
    run = gaps.cumsum()
    for _, seg in d.groupby(run.values):
        if len(seg) < 60:
            continue
        pm = seg["pm25"].values
        for i in (30, 45, len(seg) - 1):
            f = hourly_features(pm[: i + 1], seg.index[i])
            row = seg.iloc[i]
            diff = max(abs(f[c] - row[c]) for c in cols)
            worst = max(worst, diff)
            checked += 1
        break
    if checked >= 150:
        break
print("Rows checked:", checked, "| largest difference:", worst)
print("PASS" if worst < 1e-3 else "FAIL")