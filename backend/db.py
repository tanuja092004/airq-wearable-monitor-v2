import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "airq_v2.db"
DB.parent.mkdir(exist_ok=True)

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS readings (
        device_id TEXT, ts REAL, pm25 REAL, gas REAL, temp REAL, hum REAL,
        risk REAL, level TEXT,
        pm25_pred REAL, pred_risk REAL, pred_level TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_readings_dev_ts ON readings (device_id, ts)",
    """CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        asthma INTEGER DEFAULT 0,
        sensitivity TEXT DEFAULT 'normal',
        device_id TEXT UNIQUE)""",
]


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def _run(sql, params=()):
    conn = _conn()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _query(sql, params=()):
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


for s in SCHEMA:
    _run(s)


def save(device_id, ts, pm25, gas, temp, hum, out):
    fc = out["forecast"] if isinstance(out["forecast"], dict) else {}
    _run("INSERT INTO readings VALUES (?,?,?,?,?,?,?,?,?,?,?)",
         (device_id, ts, pm25, gas, temp, hum, out["risk_score"], out["level"],
          fc.get("pm25_expected_1h"), fc.get("risk_score"), fc.get("level")))


def recent_pm(device_id, since_ts):
    """Stored PM2.5 readings after since_ts, oldest first (used to rebuild the
    history after a restart)."""
    rows = _query("SELECT ts, pm25 FROM readings WHERE device_id = ? AND ts >= ? "
                  "ORDER BY ts", (device_id, since_ts))
    return [(r["ts"], r["pm25"]) for r in rows]


def create_user(name, age, asthma, sensitivity, device_id):
    return _run("INSERT INTO users (name, age, asthma, sensitivity, device_id) "
                "VALUES (?,?,?,?,?)", (name, age, int(asthma), sensitivity, device_id))


def get_user_by_device(device_id):
    rows = _query("SELECT * FROM users WHERE device_id = ?", (device_id,))
    return rows[0] if rows else None


def list_users():
    return _query("SELECT * FROM users")


def update_user(user_id, fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    _run(f"UPDATE users SET {cols} WHERE user_id = ?", (*fields.values(), user_id))


def delete_user(user_id):
    _run("DELETE FROM users WHERE user_id = ?", (user_id,))