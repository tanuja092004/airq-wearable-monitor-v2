import json
import urllib.request
from collections import Counter

BASE = "https://data.sensor.community/airrohr/v1/filter/"

TESTS = [
    ("CONTROL Germany, SDS011", "country=DE&type=SDS011"),
    ("India, any sensor", "country=IN"),
    ("India, SDS011", "country=IN&type=SDS011"),
    ("India, DHT22", "country=IN&type=DHT22"),
    ("Pune area (100 km)", "area=18.52,73.86,100"),
    ("Delhi area (100 km)", "area=28.61,77.21,100"),
    ("Mumbai area (100 km)", "area=19.07,72.88,100"),
    ("Bengaluru area (100 km)", "area=12.97,77.59,100"),
]

for name, query in TESTS:
    try:
        with urllib.request.urlopen(BASE + query, timeout=90) as r:
            rows = json.loads(r.read())
    except Exception as e:
        print(f"{name:28s} ERROR: {e}")
        continue
    locs = {x["location"]["id"] for x in rows}
    types = Counter(x["sensor"]["sensor_type"]["name"] for x in rows)
    print(f"{name:28s} readings={len(rows):6d}  locations={len(locs):5d}  types={dict(types.most_common(4))}")