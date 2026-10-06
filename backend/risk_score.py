"""Respiratory Risk Score: PM2.5 + gas index + comfort penalty, personalised by profile."""

# CPCB-style PM2.5 breakpoints (ug/m3 -> sub-index). Ranges are contiguous so every
# value maps to a sub-index. Verify against the official CPCB table before final report.
PM25_BP = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
           (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)]

# Design choices (not clinical constants): justify and tune them in the report.
W_PM25 = 0.70
W_GAS = 0.30


def sub_index(value, breakpoints=PM25_BP):
    if value <= 0:
        return 0.0
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if value <= c_hi:
            return i_lo + (value - c_lo) * (i_hi - i_lo) / (c_hi - c_lo)
    return 500.0


def comfort_penalty(temp, hum):
    p = 0
    if hum < 30 or hum > 70:
        p += 10
    if temp > 35 or temp < 10:
        p += 10
    return p


def personal_factor(asthma=False, age=None, sensitivity="normal"):
    """Multiplier applied to the score (design choices, not clinical constants)."""
    f = 1.0
    if asthma:
        f *= 1.2
    if age is not None and (age < 12 or age >= 60):
        f *= 1.1
    if sensitivity == "high":
        f *= 1.1
    return f


def level_of(score):
    return ("Low" if score < 100 else "Moderate" if score < 200
            else "High" if score < 300 else "Severe")


def risk_breakdown(pm25, gas_index, temp, hum, factor=1.0):
    """Returns (score, level, contributions). Contributions sum to the score
    (before the 500 cap). gas_index is a 0-100 value from the MQ135 proxy."""
    gas = min(max(gas_index, 0), 100)
    parts = {
        "PM2.5": W_PM25 * sub_index(pm25) * factor,
        "Gas": W_GAS * gas * factor,
        "Temp/humidity": comfort_penalty(temp, hum) * factor,
    }
    score = min(sum(parts.values()), 500)
    return round(score, 1), level_of(score), {k: round(v, 1) for k, v in parts.items()}


def respiratory_risk(pm25, gas_index, temp, hum, factor=1.0):
    score, level, _ = risk_breakdown(pm25, gas_index, temp, hum, factor)
    return score, level