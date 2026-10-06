import csv
import json
import time
from datetime import datetime
from pathlib import Path

import paho.mqtt.client as mqtt

BROKER, PORT = "localhost", 1883
TOPIC = "airq/+/data"
LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = ["recv_ts", "recv_time", "device_ts", "device_id",
           "co2", "pm25", "voc", "temp", "hum"]
count = 0


def log_row(row):
    """Append one row to today's file, writing the header if the file is new."""
    path = LOG_DIR / f"{datetime.now():%Y-%m-%d}.csv"
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(COLUMNS)
        w.writerow(row)


def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected. Logging {TOPIC} to {LOG_DIR}")
    client.subscribe(TOPIC)


def on_message(client, userdata, msg):
    global count
    try:
        d = json.loads(msg.payload)
        now = time.time()
        log_row([
            round(now, 3),
            datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            d.get("timestamp", ""),                  # only set by the fast simulator
            d.get("device_id", msg.topic.split("/")[1]),
            d.get("co2", ""), d.get("pm25", ""), d.get("voc", ""),
            d.get("temp", ""), d.get("hum", ""),
        ])
        count += 1
        if count % 20 == 0:
            print(f"{count} readings logged")
    except Exception as e:
        print("Skipped a bad message:", e)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.loop_forever()