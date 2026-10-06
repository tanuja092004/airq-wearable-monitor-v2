import json
import urllib.request

BASE = "https://data.sensor.community/airrohr/v1/filter/country=IN&type="


def fetch(sensor_type):
    with urllib.request.urlopen(BASE + sensor_type, timeout=60) as r:
        return json.loads(r.read())


def by_location(rows):
    out = {}
    for r in rows:
        loc = r["location"]
        out[loc["id"]] = (r["sensor"]["id"], loc["latitude"], loc["longitude"])
    return out


pm = by_location(fetch("SDS011"))
dht = by_location(fetch("DHT22"))
both = sorted(set(pm) & set(dht))

print("India locations with SDS011 (PM2.5):", len(pm))
print("India locations with DHT22:", len(dht))
print("Locations with BOTH:", len(both))
print()
for loc in both:
    print(f"location {loc} | PM sensor {pm[loc][0]} | DHT sensor {dht[loc][0]} "
          f"| lat/lon {pm[loc][1]}, {pm[loc][2]}")
    