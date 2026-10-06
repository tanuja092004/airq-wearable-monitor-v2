import json
import urllib.request

import paho.mqtt.client as mqtt

API = "http://127.0.0.1:8000/ingest"
CONSECUTIVE = 3          # High readings in a row before alerting
streak = {}


def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected to broker, listening on airq/+/data")
    client.subscribe("airq/+/data")


def on_message(client, userdata, msg):
    try:
        body = json.loads(msg.payload)
        body.setdefault("device_id", msg.topic.split("/")[1])
        req = urllib.request.Request(API, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=5).read())

        fc = r["forecast"]
        fc_text = fc if isinstance(fc, str) else \
            f"in 30 min: CO2 {fc['co2_predicted']} -> {fc['level']}"
        print(f"{r['device_id']} | risk {r['risk_score']} ({r['level']}) | {fc_text}")
        print("   ", r["explanation"])

        dev = r["device_id"]
        if r["level"] in ("High", "Severe"):
            streak[dev] = streak.get(dev, 0) + 1
        else:
            streak[dev] = 0
        if streak[dev] == CONSECUTIVE:
            print("   *** ALERT: sustained unsafe exposure ***")
    except Exception as e:
        print("Error:", e)


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)
client.loop_forever()