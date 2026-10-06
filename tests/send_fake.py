import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/ingest"
start = time.time() - 20 * 60

for i in range(20):
    co2 = 500 + i * 25            # CO2 slowly rising
    body = {"device_id": "device1", "co2": co2, "pm25": 40, "voc": 20,
            "temp": 28, "hum": 55, "timestamp": start + i * 60}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    print(i, resp)