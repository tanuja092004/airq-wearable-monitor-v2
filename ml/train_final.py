"""Train the model that will run in the live system (PM2.5 history + time only).

Why not use the gas columns? The wearable's MQ135 cannot supply NO2 and CO in the
same units as the stations, so a model that needs them could not be fed correctly.
Dropping them costs only about 0.5% MAE in the comparison.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from features import BASE, TIME

FEATURES = BASE + TIME
Path("ml/models").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/train_pm25_1h.csv.gz", index_col=0, parse_dates=True)
df["change"] = df["pm25_future"] - df["pm25"]
data = df.sample(min(400_000, len(df)), random_state=42)

model = RandomForestRegressor(n_estimators=40, min_samples_leaf=30, max_features=0.5,
                              n_jobs=-1, random_state=42)
model.fit(data[FEATURES], data["change"])

joblib.dump({"model": model, "features": FEATURES, "horizon_h": 1},
            "ml/models/pm25_1h_deploy.joblib")
print("Trained on", len(data), "rows with", len(FEATURES), "features")
print("Saved ml/models/pm25_1h_deploy.joblib")