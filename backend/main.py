import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import shap
from fastapi import FastAPI
from pydantic import BaseModel

import db
from nlp_explain import explain_sentence
from risk_score import personal_factor, respiratory_risk, risk_breakdown

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "ml" / "models" / "final_co2_model.joblib")
model, FEATURES = bundle["model"], bundle["features"]
explainer = shap.TreeExplainer(model)

app = FastAPI(title="Wearable Air Quality API")

# Per device: {minute_number: last co2 value in that minute}
buffers = defaultdict(dict)


class Reading(BaseModel):
    device_id: str
    co2: float
    pm25: float = 0
    voc: float = 0
    temp: float = 25
    hum: float = 50
    timestamp: Optional[float] = None    # unix seconds; defaults to now


class UserIn(BaseModel):
    name: str
    age: Optional[int] = None
    asthma: bool = False
    sensitivity: str = "normal"          # normal | high
    device_id: str


def factor_for(device_id):
    """Look up the user linked to this device and return (factor, name)."""
    u = db.get_user_by_device(device_id)
    if not u:
        return 1.0, None
    return personal_factor(bool(u["asthma"]), u["age"], u["sensitivity"]), u["name"]


def build_features(buf, m):
    """Needs a CO2 value for each of the last 16 minutes (m-15 ... m)."""
    if not all((m - k) in buf for k in range(16)):
        return None
    c = lambda k: buf[m - k]
    last10 = [c(k) for k in range(10)]
    f = {
        "co2": c(0),
        "co2_delta5": c(0) - c(5),
        "co2_lag5": c(5),
        "co2_std10": float(np.std(last10, ddof=1)),
        "co2_delta15": c(0) - c(15),
    }
    return np.array([[f[name] for name in FEATURES]])


@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}


@app.post("/ingest")
def ingest(r: Reading):
    ts = r.timestamp if r.timestamp is not None else time.time()
    m = int(ts // 60)

    buf = buffers[r.device_id]
    buf[m] = r.co2
    for old in [k for k in buf if k < m - 30]:
        del buf[old]

    factor, user_name = factor_for(r.device_id)
    score, level, parts = risk_breakdown(r.pm25, r.co2, r.voc, r.temp, r.hum, factor)
    out = {"device_id": r.device_id, "risk_score": score, "level": level,
           "explanation": explain_sentence(level, parts)}
    out["user"] = user_name
    out["personal_factor"] = factor

    x = build_features(buf, m)
    if x is None:
        out["forecast"] = "warming_up"
        out["minutes_collected"] = len(buf)
    else:
        change = float(model.predict(x)[0])
        co2_future = r.co2 + change
        f_score, f_level = respiratory_risk(r.pm25, co2_future, r.voc,
                                            r.temp, r.hum, factor)
        out["forecast"] = {
            "horizon_min": 30,
            "co2_now": r.co2,
            "co2_predicted": round(co2_future, 1),
            "risk_score": f_score,
            "level": f_level,
        }
    db.save(r, ts, out)
    return out


@app.post("/explain")
def explain(r: Reading):
    """Full explanation. Does not store the reading."""
    ts = r.timestamp if r.timestamp is not None else time.time()
    m = int(ts // 60)
    buf = dict(buffers[r.device_id])
    buf[m] = r.co2

    factor, user_name = factor_for(r.device_id)
    score, level, parts = risk_breakdown(r.pm25, r.co2, r.voc, r.temp, r.hum, factor)
    shap_info = None
    x = build_features(buf, m)
    if x is not None:
        sv = explainer.shap_values(x)[0]
        base = float(np.ravel(explainer.expected_value)[0])
        shap_info = {
            "change_ppm": float(model.predict(x)[0]),
            "base_value": base,
            "contributions": dict(zip(FEATURES, map(float, sv))),
        }
    return {
        "user": user_name,
        "personal_factor": factor,
        "risk_score": score,
        "level": level,
        "risk_contributions": parts,
        "forecast_shap": shap_info if shap_info else "warming_up",
        "sentence": explain_sentence(level, parts, shap_info),
    }


# ---------- User profile endpoints ----------

@app.post("/users")
def add_user(u: UserIn):
    try:
        uid = db.create_user(u.name, u.age, u.asthma, u.sensitivity, u.device_id)
    except Exception as e:
        return {"error": f"Could not create user (device already linked?): {e}"}
    return {"user_id": uid, **u.model_dump()}


@app.get("/users")
def get_users():
    return db.list_users()


@app.put("/users/{user_id}")
def edit_user(user_id: int, fields: dict):
    allowed = {"name", "age", "asthma", "sensitivity", "device_id"}
    db.update_user(user_id, {k: v for k, v in fields.items() if k in allowed})
    return {"updated": user_id}


@app.delete("/users/{user_id}")
def remove_user(user_id: int):
    db.delete_user(user_id)
    return {"deleted": user_id}