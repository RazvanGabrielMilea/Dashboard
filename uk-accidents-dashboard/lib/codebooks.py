from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st


# Fallback maps (STATS19-style)
FALLBACK = {
    "weather_conditions": {
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
    },
    "light_conditions": {
        1: "Daylight",
        4: "Dark (lights lit)",
        5: "Dark (lights unlit)",
        6: "Dark (no lighting)",
        7: "Dark (lighting unknown)",
        -1: "Missing",
    },
    "road_surface_conditions": {
        1: "Dry",
        2: "Wet / damp",
        3: "Snow",
        4: "Frost / ice",
        5: "Flood (over 3cm)",
        6: "Oil / diesel",
        7: "Mud",
        9: "Unknown",
        -1: "Missing",
    },
}

# Where to look for teammate mapping CSVs (put them in any of these folders)
SEARCH_DIRS = [
    Path("assets/data/codebooks"),
    Path("assets/data/lookups"),
    Path("assets/data/mappings"),
    Path("assets/data/factor_tables"),  # if they placed dictionaries here
]

# Accepted column name variants
CODE_COLS = ["code", "value", "id", "key"]
LABEL_COLS = ["label", "meaning", "description", "name", "title"]


def _find_codebook_file(factor_key: str) -> Path | None:
    candidates = [
        f"{factor_key}.csv",
        f"{factor_key}_codes.csv",
        f"{factor_key}_mapping.csv",
        f"{factor_key}_lookup.csv",
        f"{factor_key}_dictionary.csv",
    ]
    for d in SEARCH_DIRS:
        for c in candidates:
            p = d / c
            if p.exists():
                return p
    return None


def _pick_col(cols: list[str], options: list[str]) -> str | None:
    cols_l = [c.lower() for c in cols]
    for opt in options:
        if opt in cols_l:
            return cols[cols_l.index(opt)]
    return None


@st.cache_data(show_spinner=False)
def get_codebook(factor_key: str) -> dict[int, str]:
    """
    Returns a dict: {code_int: label}.
    1) If a CSV codebook exists in assets/, use it.
    2) Otherwise fallback to built-in STATS19 mappings (for known fields).
    """
    p = _find_codebook_file(factor_key)
    if p is not None:
        df = pd.read_csv(p)
        code_col = _pick_col(list(df.columns), CODE_COLS)
        label_col = _pick_col(list(df.columns), LABEL_COLS)
        if code_col and label_col:
            out = {}
            for _, r in df[[code_col, label_col]].dropna().iterrows():
                try:
                    out[int(float(r[code_col]))] = str(r[label_col])
                except Exception:
                    continue
            if out:
                return out

    return FALLBACK.get(factor_key, {})
