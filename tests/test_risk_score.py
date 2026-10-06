import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from risk_score import (level_of, personal_factor, respiratory_risk,  # noqa: E402
                        risk_breakdown, sub_index)

# every value maps to a sub-index; no gaps between ranges (old code returned 500 for 30.5)
assert abs(sub_index(30.5) - 50.83) < 0.1, sub_index(30.5)
assert sub_index(0) == 0 and sub_index(30) == 50 and sub_index(60) == 100
assert sub_index(600) == 500
prev = -1
for v in [x / 2 for x in range(0, 1100)]:          # monotonic over 0-550
    s = sub_index(v)
    assert s >= prev
    prev = s

score, level = respiratory_risk(20, 10, 25, 50)
assert level == "Low", (score, level)
score, level = respiratory_risk(150, 50, 25, 50)
assert level in ("High", "Severe"), (score, level)

base, _, _ = risk_breakdown(80, 20, 25, 50)
asth, _, _ = risk_breakdown(80, 20, 25, 50, personal_factor(asthma=True))
assert abs(asth / base - 1.2) < 0.01
assert level_of(99.9) == "Low" and level_of(100) == "Moderate"
_, _, parts = risk_breakdown(90, 30, 40, 80)
assert set(parts) == {"PM2.5", "Gas", "Temp/humidity"}
print("All risk score tests passed")