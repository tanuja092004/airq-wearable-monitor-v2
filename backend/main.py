from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

import db
import engine

app = FastAPI(title="Wearable Air Quality API (v2: PM2.5, 1-hour forecast)")


class Reading(BaseModel):
    device_id: str
    pm25: float
    gas: Optional[float] = None      # 0-100 gas index from the MQ135
    voc: Optional[float] = None      # old name, still accepted
    temp: float = 25
    hum: float = 50
    timestamp: Optional[float] = None    # unix seconds; defaults to now

    def gas_value(self):
        if self.gas is not None:
            return self.gas
        return self.voc if self.voc is not None else 0.0


class UserIn(BaseModel):
    name: str
    age: Optional[int] = None
    asthma: bool = False
    sensitivity: str = "normal"          # normal | high
    device_id: str


@app.get("/health")
def health():
    return {"status": "ok", "features": engine.FEATURES,
            "shap": engine.explainer is not None}


@app.post("/ingest")
def ingest(r: Reading):
    return engine.process(r.device_id, r.pm25, r.gas_value(), r.temp, r.hum, r.timestamp)


@app.post("/explain")
def explain(r: Reading):
    return engine.explain(r.device_id, r.pm25, r.gas_value(), r.temp, r.hum, r.timestamp)


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