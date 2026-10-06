import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

DB = Path(__file__).resolve().parent.parent / "data" / "airq.db"
st.set_page_config(page_title="Air Quality Monitor", layout="wide")
st.title("Wearable Air Quality Monitor")

if not DB.exists():
    st.warning("No data yet. Start the API, gateway and simulator.")
    st.stop()

conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM readings ORDER BY ts DESC LIMIT 300", conn)
conn.close()

if df.empty:
    st.warning("No readings yet.")
    st.stop()

df = df.sort_values("ts")
df["time"] = pd.to_datetime(df["ts"], unit="s")
last = df.iloc[-1]

# Look up the user linked to the latest device
conn = sqlite3.connect(DB)
user = conn.execute(
    "SELECT name, age, asthma, sensitivity FROM users WHERE device_id = ?",
    (last["device_id"],),
).fetchone()
conn.close()

if user:
    name, age, asthma, sensitivity = user
    st.caption(
        f"User: {name} | Age: {age if age is not None else 'n/a'} | "
        f"Asthma: {'yes' if asthma else 'no'} | Sensitivity: {sensitivity} "
        f"| Device: {last['device_id']}"
    )
else:
    st.caption(f"Device: {last['device_id']} | No user profile linked")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk now", f"{last['risk']:.0f}", last["level"], delta_color="off")
c2.metric("PM2.5", f"{last['pm25']:.1f} ug/m3")
c3.metric("CO2", f"{last['co2']:.0f} ppm")
c4.metric("Temp / Humidity", f"{last['temp']:.1f} C / {last['hum']:.0f}%")

if last["level"] in ("High", "Severe"):
    st.error(f"Unsafe exposure: risk level {last['level']}")
if pd.notna(last["pred_level"]):
    st.info(f"Forecast in 30 min: CO2 about {last['co2_pred']:.0f} ppm, "
            f"risk level {last['pred_level']}")

left, right = st.columns(2)
left.subheader("Respiratory risk score")
left.line_chart(df.set_index("time")["risk"])
right.subheader("CO2 and PM2.5")
right.line_chart(df.set_index("time")[["co2", "pm25"]])

st.subheader("Latest readings")
st.dataframe(df.tail(15).drop(columns=["ts"]).iloc[::-1], use_container_width=True)

time.sleep(3)
st.rerun()