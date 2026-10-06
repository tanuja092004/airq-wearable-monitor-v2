"""Compare models for 1-hour-ahead PM2.5 using leave-stations-out validation."""
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

SAMPLE = 60_000           # training rows per fold (keeps runtime reasonable)
TEST_SAMPLE = 80_000      # test rows per fold
N_FOLDS = 5
Path("ml/models").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/train_pm25_1h.csv.gz", index_col=0, parse_dates=True)
df["change"] = df["pm25_future"] - df["pm25"]          # target = change over 1 hour
print("Rows:", len(df), "| stations:", df["station"].nunique())

BASE = ["pm25", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6", "pm25_lag12",
        "pm25_lag24", "pm25_delta1", "pm25_delta3", "pm25_roll3", "pm25_roll6",
        "pm25_roll24", "pm25_std6"]
TIME = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
GAS = ["no2", "co"]
FEATURE_SETS = {"pm": BASE, "pm+time": BASE + TIME, "pm+time+gas": BASE + TIME + GAS}

# CPCB-style PM2.5 bands -> category index (used to measure alert-level accuracy)
BANDS = [30, 60, 90, 120, 250]
def band(x):
    return np.digitize(x, BANDS)

def make_models():
    return {
        "Ridge": Ridge(alpha=10.0),
        "RandomForest": RandomForestRegressor(n_estimators=40, min_samples_leaf=30,
                                              max_features=0.5, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=40, min_samples_leaf=30,
                                          max_features=0.7, n_jobs=-1, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=80, max_depth=4,
                                                      learning_rate=0.08, subsample=0.7,
                                                      random_state=42),
        "HistGradBoost (no SHAP)": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=200, min_samples_leaf=100,
            random_state=42),
    }

def score(y_true, pred, now):
    return {"mae": mean_absolute_error(y_true, pred),
            "rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
            "band_acc": float(np.mean(band(y_true) == band(pred)))}

gkf = GroupKFold(n_splits=N_FOLDS)
folds = []
for tr, te in gkf.split(df, groups=df["station"]):
    te = np.random.RandomState(42).choice(te, min(TEST_SAMPLE, len(te)), replace=False)
    folds.append((tr, te))

# ---- persistence baseline ("same as now") ----
pers = [score(df.iloc[te]["pm25_future"].values, df.iloc[te]["pm25"].values, None)
        for _, te in folds]
pers_mae = np.mean([p["mae"] for p in pers])
print(f"\nPersistence: MAE {pers_mae:.2f} | RMSE {np.mean([p['rmse'] for p in pers]):.2f} "
      f"| band accuracy {np.mean([p['band_acc'] for p in pers]) * 100:.1f}%")
print("Per-fold persistence MAE:", [round(p["mae"], 1) for p in pers], "\n")

rows = []
t0 = time.time()
for fs_name, feats in FEATURE_SETS.items():
    for mname in make_models():
        res = []
        for tr, te in folds:
            train = df.iloc[tr].sample(min(SAMPLE, len(tr)), random_state=42)
            test = df.iloc[te]
            Xtr, Xte = train[feats], test[feats]
            if mname == "Ridge" or "Forest" in mname or "Extra" in mname or mname == "GradientBoosting":
                med = Xtr.median()
                Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)     # these models need no NaN
            m = make_models()[mname]
            m.fit(Xtr, train["change"])
            pred = np.clip(test["pm25"].values + m.predict(Xte), 0, 500)
            res.append(score(test["pm25_future"].values, pred, None))
        rows.append({"features": fs_name, "model": mname,
                     "MAE": np.mean([r["mae"] for r in res]),
                     "RMSE": np.mean([r["rmse"] for r in res]),
                     "band_acc_%": np.mean([r["band_acc"] for r in res]) * 100,
                     "folds_beating_persistence": sum(r["mae"] < p["mae"] for r, p in zip(res, pers))})
        print(f"done {fs_name:12s} {mname:24s} MAE {rows[-1]['MAE']:.2f}  ({time.time() - t0:.0f}s)")

table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
table["vs_persistence_%"] = (1 - table["MAE"] / pers_mae) * 100
print("\n", table.round(2).to_string())

# ---- choose the best model that SHAP can explain, and retrain it on all stations ----
shap_ok = table[~table["model"].str.contains("no SHAP") & (table["model"] != "Ridge")]
best = shap_ok.iloc[0]
feats = FEATURE_SETS[best["features"]]
final = make_models()[best["model"]]
data = df.sample(min(400_000, len(df)), random_state=42)
Xall = data[feats].fillna(df[feats].median())
final.fit(Xall, data["change"])
joblib.dump({"model": final, "features": feats, "horizon_h": 1,
             "fill_values": df[feats].median().to_dict()},
            "ml/models/pm25_1h_model.joblib")
print(f"\nBEST (SHAP-compatible): {best['model']} with '{best['features']}' | "
      f"MAE {best['MAE']:.2f} ({best['vs_persistence_%']:.1f}% vs persistence)")
print("Saved ml/models/pm25_1h_model.joblib")