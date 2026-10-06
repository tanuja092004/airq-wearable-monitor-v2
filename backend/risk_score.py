def sub_index(value, breakpoints):
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= value <= c_hi:
            return i_lo + (value - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    return 500

# CPCB-style PM2.5 breakpoints (verify against the official CPCB table)
PM25_BP = [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
           (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)]


def comfort_penalty(temp, hum):
    p = 0
    if hum < 30 or hum > 70:
        p += 10
    if temp > 35 or temp < 10:
        p += 10
    return p


def personal_factor(asthma=False, age=None, sensitivity="normal"):
    """Multiplier applied to the score. These values are design choices,
    not clinical constants, so justify and tune them in your report."""
    f = 1.0
    if asthma:
        f *= 1.2
    if age is not None and (age < 12 or age >= 60):
        f *= 1.1
    if sensitivity == "high":
        f *= 1.1
    return f


def risk_breakdown(pm25, co2_ppm, voc_index, temp, hum, factor=1.0):
    """Returns (score, level, contributions). Contributions sum to the score."""
    pm = sub_index(pm25, PM25_BP)
    co2 = min(max((co2_ppm - 400) / 16, 0), 100)    # 400-2000 ppm -> 0-100
    voc = min(max(voc_index, 0), 100)
    parts = {
        "PM2.5": 0.55 * pm * factor,
        "CO2": 0.20 * co2 * factor,
        "VOC": 0.15 * voc * factor,
        "Temp/humidity": comfort_penalty(temp, hum) * factor,
    }
    score = min(sum(parts.values()), 500)
    level = ("Low" if score < 100 else "Moderate" if score < 200
             else "High" if score < 300 else "Severe")
    return round(score, 1), level, {k: round(v, 1) for k, v in parts.items()}


def respiratory_risk(pm25, co2_ppm, voc_index, temp, hum, factor=1.0):
    score, level, _ = risk_breakdown(pm25, co2_ppm, voc_index, temp, hum, factor)
    return score, level