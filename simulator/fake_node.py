import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

p = argparse.ArgumentParser()
p.add_argument("--device", default="device1")
p.add_argument("--interval", type=float, default=5, help="seconds between readings")
p.add_argument("--fast", action="store_true",
               help="fake time: each reading is 1 minute apart, sent every second")
args = p.parse_args()

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883)
client.loop_start()
topic = f"airq/{args.device}/data"

co2, pm25 = 550.0, 25.0
spike_left = 0
start = time.time() - 30 * 60
i = 0
print(f"Publishing to {topic} (Ctrl+C to stop)")

while True:
    if spike_left == 0 and random.random() < 0.04:
        spike_left = 20                      # a pollution event
    if spike_left > 0:
        co2 += 30
        pm25 += 8
        spike_left -= 1
    else:
        co2 += (550 - co2) * 0.03            # slowly return to normal
        pm25 += (25 - pm25) * 0.10
    co2 += random.gauss(0, 8)
    pm25 += random.gauss(0, 1)
    co2, pm25 = max(co2, 400), max(pm25, 1)

    msg = {
        "device_id": args.device,
        "co2": round(co2, 1),
        "pm25": round(pm25, 1),
        "voc": round(random.uniform(10, 40), 1),
        "temp": round(random.gauss(27, 0.3), 1),
        "hum": round(random.gauss(55, 1.0), 1),
    }
    if args.fast:
        msg["timestamp"] = start + i * 60    # fake clock, 1 minute per message
    client.publish(topic, json.dumps(msg))
    print(msg)

    i += 1
    time.sleep(1 if args.fast else args.interval)