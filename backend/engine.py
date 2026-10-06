"""Core logic: hourly aggregation, forecast, risk score, explanation.
Kept separate from the web layer (main.py) so it can be tested on its own."""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from features import MIN_HISTORY, hourly_features  # noqa: E402

import db  # noqa: E402
from nlp_explain import explain_sentence  # noqa: E402
from risk_score import personal_factor, respiratory_risk, risk_breakdown  # noqa: E402

bundle = joblib.load(ROOT / "ml" / "models" / "pm25_1h_deploy.joblib")
model, FEATURES = bundle["model"], bundle["features"]

try:
    import shap
    explainer = shap.TreeExplainer(model)
except Exception as e:                      # the API still works without SHAP
    print("SHAP not available:", e)
    explainer = None

IST = timezone(timedelta(hours=5, minutes=30))   # the training data is in Indian time
HOURS = MIN_HISTORY                              # 25 hourly values are needed
MIN_PER_BIN = 3                                  # readings needed in each hourly window
KEEP_S = (HOURS + 1) * 3600

buffers = {}        # device_id -> {"ts": [...], "pm": [...]}  (raw readings, last 26 h)


def _get_buffer(device_id, now_ts):
    if device_id not in buffers:
        rows = db.recent_pm(device_id, now_ts - KEEP_S)   # rebuild after a restart
        buffers[device_id] = {"ts": [r[0] for r in rows], "pm": [r[1] for r in rows]}
    return buffers[device_id]


def _trim(buf, now_ts):
    cutoff = now_ts - KEEP_S
    if buf["ts"] and min(buf["ts"][0], buf["ts"][-1]) < cutoff:
        keep = [i for i, t in enumerate(buf["ts"]) if t >= cutoff]
        buf["ts"] = [buf["ts"][i] for i in keep]
        buf["pm"] = [buf["pm"][i] for i in keep]


def hourly_series(buf, t):
    """Mean PM2.5 of each of the last 25 rolling hours before time t, oldest first.
    Hours with too few readings are NaN."""
    ts = np.asarray(buf["ts"], dtype=float)
    pm = np.asarray(buf["pm"], dtype=float)
    age = t - ts
    ok = (age >= 0) & (age < HOURS * 3600)
    k = (age[ok] // 3600).astype(int)
    sums = np.bincount(k, weights=pm[ok], minlength=HOURS)
    cnt = np.bincount(k, minlength=HOURS)
    means = np.where(cnt >= MIN_PER_BIN, sums / np.maximum(cnt, 1), np.nan)
    return means[::-1]


def _forecast(buf, t, gas, temp, hum, factor, want_shap=False):
    series = hourly_series(buf, t)
    f = hourly_features(series, datetime.fromtimestamp(t, tz=IST))
    if f is None:
        span_h = (t - min(buf["ts"])) / 3600 if buf["ts"] else 0
        return None, {"forecast": "warming_up",
                      "hours_collected": round(span_h, 1), "hours_needed": HOURS}
    x = pd.DataFrame([[f[n] for n in FEATURES]], columns=FEATURES)
    change = float(model.predict(x)[0])
    now_h = float(f["pm25"])
    pred = float(np.clip(now_h + change, 0, 500))
    f_score, f_level = respiratory_risk(pred, gas, temp, hum, factor)
    fc = {"horizon_h": 1, "pm25_last_hour": round(now_h, 1),
          "pm25_expected_1h": round(pred, 1), "change": round(change, 1),
          "risk_score": f_score, "level": f_level}
    shap_info = None
    if want_shap and explainer is not None:
        sv = explainer.shap_values(x)[0]
        shap_info = {"change_ugm3": change,
                     "base_value": float(np.ravel(explainer.expected_value)[0]),
                     "contributions": dict(zip(FEATURES, map(float, sv)))}
    return shap_info, fc


def factor_for(device_id):
    u = db.get_user_by_device(device_id)
    if not u:
        return 1.0, None
    return personal_factor(bool(u["asthma"]), u["age"], u["sensitivity"]), u["name"]


def process(device_id, pm25, gas, temp, hum, ts=None):
    """Handle one reading: update history, score, forecast, store."""
    ts = float(ts) if ts is not None else time.time()
    buf = _get_buffer(device_id, ts)
    buf["ts"].append(ts)
    buf["pm"].append(pm25)
    _trim(buf, ts)

    factor, user_name = factor_for(device_id)
    score, level, parts = risk_breakdown(pm25, gas, temp, hum, factor)
    out = {"device_id": device_id, "user": user_name, "personal_factor": factor,
           "risk_score": score, "level": level,
           "explanation": explain_sentence(level, parts)}
    _, fc = _forecast(buf, ts, gas, temp, hum, factor)
    out["forecast"] = fc
    db.save(device_id, ts, pm25, gas, temp, hum, out)
    return out


def explain(device_id, pm25, gas, temp, hum, ts=None):
    """Full explanation with SHAP. Does not store the reading."""
    ts = float(ts) if ts is not None else time.time()
    real = _get_buffer(device_id, ts)
    buf = {"ts": list(real["ts"]) + [ts], "pm": list(real["pm"]) + [pm25]}

    factor, user_name = factor_for(device_id)
    score, level, parts = risk_breakdown(pm25, gas, temp, hum, factor)
    shap_info, fc = _forecast(buf, ts, gas, temp, hum, factor, want_shap=True)
    return {"user": user_name, "personal_factor": factor,
            "risk_score": score, "level": level, "risk_contributions": parts,
            "forecast": fc,
            "forecast_shap": shap_info if shap_info else
            ("shap_not_installed" if explainer is None else "warming_up"),
            "sentence": explain_sentence(level, parts, shap_info)}