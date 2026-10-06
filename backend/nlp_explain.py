"""Turns risk contributions and SHAP values into a plain-language sentence."""

RISK_PHRASES = {
    "PM2.5": "elevated PM2.5 (fine dust)",
    "Gas": "a high gas / VOC level",
    "Temp/humidity": "uncomfortable temperature or humidity",
}

FEATURE_LABELS = {
    "pm25": "the current PM2.5 level",
    "pm25_lag1": "PM2.5 one hour ago",
    "pm25_lag2": "PM2.5 two hours ago",
    "pm25_lag3": "PM2.5 three hours ago",
    "pm25_lag6": "PM2.5 six hours ago",
    "pm25_lag12": "PM2.5 twelve hours ago",
    "pm25_lag24": "PM2.5 at the same time yesterday",
    "pm25_delta1": "the PM2.5 change over the last hour",
    "pm25_delta3": "the PM2.5 trend of the last three hours",
    "pm25_roll3": "the 3-hour PM2.5 average",
    "pm25_roll6": "the 6-hour PM2.5 average",
    "pm25_roll24": "the 24-hour PM2.5 average",
    "pm25_std6": "how much PM2.5 fluctuated recently",
    "hour_sin": "the time of day",
    "hour_cos": "the time of day",
    "month_sin": "the season",
    "month_cos": "the season",
}


def explain_sentence(level, parts, shap_info=None):
    total = sum(parts.values())
    top = max(parts, key=parts.get)
    share = parts[top] / total * 100 if total > 0 else 0

    if level == "Low":
        text = (f"Air quality is fine. The largest contributor is "
                f"{RISK_PHRASES[top]} ({share:.0f}% of the score).")
    else:
        text = (f"Your risk is {level.lower()}, mainly due to "
                f"{RISK_PHRASES[top]} (about {share:.0f}% of the score).")

    if shap_info:
        change = shap_info["change_ugm3"]
        contrib = shap_info["contributions"]
        top_f = max(contrib, key=lambda k: abs(contrib[k]))
        if abs(change) < 2:
            text += " PM2.5 is expected to stay about the same over the next hour."
        else:
            direction = "rise" if change > 0 else "fall"
            text += (f" Over the next hour PM2.5 is expected to {direction} by about "
                     f"{abs(change):.0f} ug/m3, driven mostly by "
                     f"{FEATURE_LABELS.get(top_f, top_f)}.")
    return text