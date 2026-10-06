RISK_PHRASES = {
    "PM2.5": "elevated PM2.5 (fine dust)",
    "CO2": "high CO2 (poor ventilation)",
    "VOC": "high VOC/gas levels",
    "Temp/humidity": "uncomfortable temperature or humidity",
}

FEATURE_LABELS = {
    "co2": "the current CO2 level",
    "co2_delta5": "the CO2 trend of the last 5 minutes",
    "co2_lag5": "the CO2 level 5 minutes ago",
    "co2_std10": "recent CO2 fluctuation",
    "co2_delta15": "the CO2 trend of the last 15 minutes",
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
        change = shap_info["change_ppm"]
        top_f = max(shap_info["contributions"],
                    key=lambda k: abs(shap_info["contributions"][k]))
        direction = "rise" if change > 0 else "fall"
        text += (f" In 30 minutes CO2 is expected to {direction} by about "
                 f"{abs(change):.0f} ppm, driven mostly by "
                 f"{FEATURE_LABELS.get(top_f, top_f)}.")
    return text