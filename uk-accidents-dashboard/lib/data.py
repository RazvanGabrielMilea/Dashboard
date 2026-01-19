from __future__ import annotations

from pathlib import Path
import os
import json
import pandas as pd
import streamlit as st

# Big parquet (fallback only)
DEFAULT_DATA_PATH = Path("data/traffic_enriched.parquet")
ENV_PATH = os.getenv("UK_ACCIDENTS_PARQUET")

# Fast cache outputs (preferred)
CACHE_DIR = Path("data/dashboard_cache")
FAST_COLLISIONS = CACHE_DIR / "collisions_fast.parquet"
META_JSON = CACHE_DIR / "metadata.json"

SEVERITY_MAP = {
    1: "Fatal", 2: "Serious", 3: "Slight",
    1.0: "Fatal", 2.0: "Serious", 3.0: "Slight",
}

def _resolved_big_path() -> Path:
    p = Path(ENV_PATH) if ENV_PATH else DEFAULT_DATA_PATH
    return p

@st.cache_data(show_spinner=False)
def get_sidebar_metadata() -> dict:
    """
    Loads precomputed metadata (years / severities / top LSOAs).
    Falls back to a small computation on the FAST collisions file if metadata is missing.
    """
    if META_JSON.exists():
        return json.loads(META_JSON.read_text(encoding="utf-8"))

    # fallback: build from fast collisions parquet (still much smaller than full file)
    if not FAST_COLLISIONS.exists():
        # last resort: minimal defaults
        return {"years": [2021], "severity_labels": ["Slight", "Serious", "Fatal"], "top_lsoas": ["All"]}

    df = pd.read_parquet(FAST_COLLISIONS, columns=["accident_date", "collision_severity", "lsoa_of_casualty"])
    df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")

    years = sorted([int(x) for x in df["accident_date"].dt.year.dropna().unique()])
    sev = pd.Series(df["collision_severity"]).map(SEVERITY_MAP).dropna().astype(str).unique().tolist()
    order = ["Slight", "Serious", "Fatal"]
    sev = [x for x in order if x in sev] + [x for x in sorted(sev) if x not in order]

    top_lsoas = ["All"] + df["lsoa_of_casualty"].dropna().astype(str).value_counts().head(200).index.tolist()

    meta = {"years": years, "severity_labels": sev or order, "top_lsoas": top_lsoas}
    return meta

@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    """
    Loads FAST collision-level data if available.
    Otherwise falls back to loading the huge parquet (slower).
    """
    if FAST_COLLISIONS.exists():
        df = pd.read_parquet(FAST_COLLISIONS)
    else:
        # fallback (slow): load only what we need, not all columns
        path = _resolved_big_path()
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found: {path.resolve()}")

        cols = [
            "collision_index", "accident_date", "accident_hour",
            "collision_severity", "casualty_severity", "lsoa_of_casualty"
        ]
        df = pd.read_parquet(path, columns=cols)

        # Create collision-level quickly (no extra metadata file)
        df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")
        df["accident_hour"] = pd.to_numeric(df["accident_hour"], errors="coerce")
        df["_killed"] = (df["casualty_severity"] == 1).astype("int8")
        df = df.groupby("collision_index", sort=False).agg(
            accident_date=("accident_date", "first"),
            accident_hour=("accident_hour", "first"),
            collision_severity=("collision_severity", "first"),
            lsoa_of_casualty=("lsoa_of_casualty", "first"),
            persons_involved=("collision_index", "size"),
            persons_killed=("_killed", "sum"),
        ).reset_index()

    df.columns = [c.strip() for c in df.columns]

    # Derived fields
    df["_date"] = pd.to_datetime(df["accident_date"], errors="coerce")
    df["_hour"] = pd.to_numeric(df["accident_hour"], errors="coerce")
    df["_year"] = df["_date"].dt.year.astype("Int64")
    df["_month"] = df["_date"].dt.to_period("M").astype(str)
    df["_day"] = df["_date"].dt.date
    df["_dow"] = df["_date"].dt.day_name()

    df["_collision_sev"] = pd.Series(df["collision_severity"]).map(SEVERITY_MAP).fillna(df["collision_severity"].astype(str))
    return df

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df

    year = filters.get("year")
    if year is not None:
        out = out[out["_year"] == int(year)]

    lsoa = filters.get("lsoa", "All")
    if lsoa != "All" and "lsoa_of_casualty" in out.columns:
        out = out[out["lsoa_of_casualty"].astype(str) == str(lsoa)]

    severity = filters.get("severity", [])
    if severity:
        out = out[out["_collision_sev"].astype(str).isin([str(s) for s in severity])]

    return out

def compute_kpis(df_f: pd.DataFrame) -> dict:
    total_accidents = int(len(df_f))  # collision-level => 1 row per accident
    fatal_accidents = int((df_f["collision_severity"] == 1).sum()) if "collision_severity" in df_f.columns else None

    persons_involved = int(df_f["persons_involved"].sum()) if "persons_involved" in df_f.columns else None
    persons_killed = int(df_f["persons_killed"].sum()) if "persons_killed" in df_f.columns else None

    avg_accidents_per_day = None
    if df_f["_day"].notna().any():
        days = int(df_f["_day"].nunique())
        avg_accidents_per_day = round(total_accidents / days, 2) if days else None

    return {
        "total_accidents": total_accidents,
        "fatal_accidents": fatal_accidents,
        "persons_involved": persons_involved,
        "persons_killed": persons_killed,
        "avg_accidents_per_day": avg_accidents_per_day,
    }
