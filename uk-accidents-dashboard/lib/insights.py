from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.overview_data import (
    load_monthly_by_year,
    load_hour_dow_by_year,
    load_severity_by_year,
)
from lib.assets import asset_exists, read_csv


# --- label maps (same as Trends) ---
WEATHER_MAP = {
    1: "Fine (no high winds)",
    2: "Raining (no high winds)",
    3: "Snowing (no high winds)",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Missing",
}

LIGHT_MAP = {
    1: "Daylight",
    4: "Dark (lights lit)",
    5: "Dark (lights unlit)",
    6: "Dark (no lighting)",
    7: "Dark (lighting unknown)",
    -1: "Missing",
}

ROAD_SURFACE_MAP = {
    1: "Dry",
    2: "Wet / damp",
    3: "Snow",
    4: "Frost / ice",
    5: "Flood (over 3cm)",
    6: "Oil / diesel",
    7: "Mud",
    9: "Unknown",
    -1: "Missing",
}


def _rate_to_pct(series: pd.Series) -> pd.Series:
    mx = series.max()
    return series * 100.0 if mx <= 1.0 else series


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _top_factor_from_csv(csv_name: str, key_col: str, mapping: dict[int, str] | None):
    """
    Returns: (label, risk_pct, n)
    """
    if not asset_exists("data", "factor_tables", csv_name):
        return None

    df = read_csv("data", "factor_tables", csv_name).copy()
    if "serious_or_fatal_rate" not in df.columns or key_col not in df.columns:
        return None

    df["risk_pct"] = _rate_to_pct(df["serious_or_fatal_rate"])
    df = df.dropna(subset=["risk_pct"]).sort_values("risk_pct", ascending=False)

    if df.empty:
        return None

    r = df.iloc[0]
    code = _safe_int(_safe_float(r[key_col]))
    label = mapping.get(code, f"Code {code}") if (mapping and code is not None) else str(r[key_col])
    risk = float(r["risk_pct"])
    n = int(r["n"]) if "n" in df.columns and pd.notna(r.get("n")) else None
    return label, risk, n


@st.cache_data(show_spinner=False)
def overview_insights(selected_year: int | None) -> dict:
    """
    Computes lightweight insights based on precomputed overview tables (and teammate factor tables).
    Returns dict of insight values.
    """
    out: dict = {}

    # -----------------------
    # Peak month
    # -----------------------
    monthly = load_monthly_by_year().copy()
    if selected_year is not None:
        monthly = monthly[monthly["year"] == int(selected_year)]
    if not monthly.empty:
        r = monthly.sort_values("accidents", ascending=False).iloc[0]
        out["peak_month_name"] = str(r["month_name"])
        out["peak_month_accidents"] = int(r["accidents"])

    # -----------------------
    # Peak day+hour (heatmap)
    # -----------------------
    hd = load_hour_dow_by_year().copy()
    if selected_year is not None:
        hd = hd[hd["year"] == int(selected_year)]
    if not hd.empty:
        r = hd.sort_values("accidents", ascending=False).iloc[0]
        out["peak_dow"] = str(r["dow"])
        out["peak_hour"] = int(r["hour"])
        out["peak_dow_hour_accidents"] = int(r["accidents"])

    # -----------------------
    # Serious/Fatal share
    # -----------------------
    sev = load_severity_by_year().copy()
    if selected_year is not None:
        sev = sev[sev["year"] == int(selected_year)]
    if not sev.empty:
        total = float(sev["accidents"].sum())
        sf = float(sev[sev["severity"].isin(["Serious", "Fatal"])]["accidents"].sum())
        if total > 0:
            out["serious_fatal_share_pct"] = round((sf / total) * 100.0, 2)

    # -----------------------
    # Top risk factor (overall, not year-specific)
    # (The factor tables are global aggregates from teammate.
    #  If later you generate factor tables by year, we can make this filter-aware.)
    # -----------------------
    w = _top_factor_from_csv("weather_conditions_severity_rates.csv", "weather_conditions", WEATHER_MAP)
    l = _top_factor_from_csv("light_conditions_severity_rates.csv", "light_conditions", LIGHT_MAP)
    rs = _top_factor_from_csv("road_surface_conditions_severity_rates.csv", "road_surface_conditions", ROAD_SURFACE_MAP)

    out["top_weather"] = w  # (label, risk_pct, n)
    out["top_light"] = l
    out["top_road_surface"] = rs

    return out
