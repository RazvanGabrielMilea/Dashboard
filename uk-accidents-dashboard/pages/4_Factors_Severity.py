import streamlit as st
import plotly.express as px

from lib.assets import read_csv, asset_exists

"""
Factors & Severity Page for UK Accidents Dashboard.

This Streamlit page displays pre-aggregated factor analysis showing how different
conditions (weather, light, road surface, speed limit) correlate with accident severity.
It uses precomputed tables for fast loading and visualization without querying the full dataset.
"""

# Small mappings for nicer labels (only where we’re confident)
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


def _to_int_or_none(x):
    """
    Safely convert a value to integer or return None if conversion fails.

    Parameters:
    - x: The value to convert.

    Returns:
    - int or None: The integer value or None if conversion fails.
    """
    try:
        return int(float(x))
    except Exception:
        return None


def load_factor_table(csv_name: str):
    """
    Load a factor analysis table from the assets directory.

    Parameters:
    - csv_name (str): The name of the CSV file in assets/data/factor_tables/.

    Returns:
    - pd.DataFrame: The loaded factor table.
    """
    # These exist in: assets/data/factor_tables/
    return read_csv("data", "factor_tables", csv_name)


def add_label(df, col_name: str, mapping: dict | None, label_name: str):
    """
    Add a human-readable label column to a DataFrame based on a mapping dictionary.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - col_name (str): The name of the column to map.
    - mapping (dict | None): Dictionary mapping codes to labels. If None, uses string conversion.
    - label_name (str): The name of the new label column.

    Returns:
    - pd.DataFrame: The DataFrame with the new label column added.
    """
    if mapping is None:
        df[label_name] = df[col_name].astype(str)
        return df

    def _map(v):
        vi = _to_int_or_none(v)
        if vi is None:
            return str(v)
        return mapping.get(vi, f"Code {vi}")

    df[label_name] = df[col_name].map(_map)
    return df


def ensure_pct(df, col):
    """
    Ensure a rate column is in percentage format (0-100).

    If the maximum value is <= 1.0, assumes it's in decimal form and multiplies by 100.
    Otherwise, assumes it's already in percent.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - col (str): The name of the rate column.

    Returns:
    - pd.DataFrame: The DataFrame with a new column '{col}_pct' in percentage format.
    """
    # Some files store rates in [0..1], others already in percent
    if col not in df.columns:
        return df
    mx = df[col].max()
    if mx <= 1.0:
        df[col + "_pct"] = df[col] * 100.0
    else:
        df[col + "_pct"] = df[col]
    return df


st.set_page_config(layout="wide")

# Page header and description
st.markdown("<div class='dash-title'>Factors & Severity</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='dash-sub'>Fast analysis from pre-aggregated factor tables (no parquet load)</div>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Check for required factor table assets
needed = [
    ("weather_conditions_severity_rates.csv", "weather_conditions"),
    ("light_conditions_severity_rates.csv", "light_conditions"),
    ("road_surface_conditions_severity_rates.csv", "road_surface_conditions"),
    ("speed_limit_severity_rates.csv", "speed_limit"),
]
missing = [f for f, _ in needed if not asset_exists("data", "factor_tables", f)]
if missing:
    st.error(
        "Missing factor tables in assets/data/factor_tables:\n- " + "\n- ".join(missing)
    )
    st.stop()

# User control for number of top categories to display
top_n = st.slider("Top categories", 5, 30, 12)

# Create columns for factor charts
c1, c2 = st.columns(2)

# Chart 1: Weather conditions severity analysis
with c1:
    st.markdown(
        "<div class='panel-title'>Weather conditions</div>", unsafe_allow_html=True
    )
    df = load_factor_table("weather_conditions_severity_rates.csv")
    df = ensure_pct(df, "serious_or_fatal_rate")
    df = ensure_pct(df, "fatal_rate")
    df = ensure_pct(df, "serious_rate")
    df = add_label(df, "weather_conditions", WEATHER_MAP, "Weather")

    df = df.sort_values("serious_or_fatal_rate_pct", ascending=False).head(top_n)

    fig = px.bar(
        df,
        x="Weather",
        y="serious_or_fatal_rate_pct",
        hover_data={
            "n": ":,",
            "fatal_rate_pct": ":.2f",
            "serious_rate_pct": ":.2f",
        },
        labels={
            "serious_or_fatal_rate_pct": "Serious/Fatal rate (%)",
            "Weather": "Weather",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Serious/Fatal rate (%)")
    fig.update_traces(
        hovertemplate="Weather: %{x}<br>Serious/Fatal: %{y:.2f}%<br>Fatal: %{customdata[1]:.2f}%<br>Serious: %{customdata[2]:.2f}%<br>n: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 2: Light conditions severity analysis
with c2:
    st.markdown(
        "<div class='panel-title'>Light conditions</div>", unsafe_allow_html=True
    )
    df = load_factor_table("light_conditions_severity_rates.csv")
    df = ensure_pct(df, "serious_or_fatal_rate")
    df = ensure_pct(df, "fatal_rate")
    df = ensure_pct(df, "serious_rate")
    df = add_label(df, "light_conditions", LIGHT_MAP, "Light")

    df = df.sort_values("serious_or_fatal_rate_pct", ascending=False).head(top_n)

    fig = px.bar(
        df,
        x="Light",
        y="serious_or_fatal_rate_pct",
        hover_data={"n": ":,", "fatal_rate_pct": ":.2f", "serious_rate_pct": ":.2f"},
        labels={
            "serious_or_fatal_rate_pct": "Serious/Fatal rate (%)",
            "Light": "Light",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Serious/Fatal rate (%)")
    fig.update_traces(
        hovertemplate="Light: %{x}<br>Serious/Fatal: %{y:.2f}%<br>Fatal: %{customdata[1]:.2f}%<br>Serious: %{customdata[2]:.2f}%<br>n: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

# Create second row of columns for remaining factors
c3, c4 = st.columns(2)

# Chart 3: Road surface conditions severity analysis
with c3:
    st.markdown(
        "<div class='panel-title'>Road surface conditions</div>", unsafe_allow_html=True
    )
    df = load_factor_table("road_surface_conditions_severity_rates.csv")
    df = ensure_pct(df, "serious_or_fatal_rate")
    df = ensure_pct(df, "fatal_rate")
    df = ensure_pct(df, "serious_rate")
    df = add_label(df, "road_surface_conditions", ROAD_SURFACE_MAP, "Surface")

    df = df.sort_values("serious_or_fatal_rate_pct", ascending=False).head(top_n)

    fig = px.bar(
        df,
        x="Surface",
        y="serious_or_fatal_rate_pct",
        hover_data={"n": ":,", "fatal_rate_pct": ":.2f", "serious_rate_pct": ":.2f"},
        labels={
            "serious_or_fatal_rate_pct": "Serious/Fatal rate (%)",
            "Surface": "Surface",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Serious/Fatal rate (%)")
    fig.update_traces(
        hovertemplate="Surface: %{x}<br>Serious/Fatal: %{y:.2f}%<br>Fatal: %{customdata[1]:.2f}%<br>Serious: %{customdata[2]:.2f}%<br>n: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 4: Speed limit severity analysis (line chart)
with c4:
    st.markdown("<div class='panel-title'>Speed limit</div>", unsafe_allow_html=True)
    df = load_factor_table("speed_limit_severity_rates.csv")
    df = ensure_pct(df, "serious_or_fatal_rate")
    df = ensure_pct(df, "fatal_rate")
    df = ensure_pct(df, "serious_rate")

    # Keep top N speed limits by count, then order by speed
    df = df.sort_values("n", ascending=False).head(top_n)
    df["speed_limit"] = df["speed_limit"].astype(float)
    df = df.sort_values("speed_limit")

    fig = px.line(
        df,
        x="speed_limit",
        y="serious_or_fatal_rate_pct",
        markers=True,
        hover_data={"n": ":,", "fatal_rate_pct": ":.2f", "serious_rate_pct": ":.2f"},
        labels={
            "speed_limit": "Speed limit",
            "serious_or_fatal_rate_pct": "Serious/Fatal rate (%)",
        },
    )
    fig.update_layout(xaxis_title="Speed limit", yaxis_title="Serious/Fatal rate (%)")
    fig.update_traces(
        hovertemplate="Speed: %{x:.0f}<br>Serious/Fatal: %{y:.2f}%<br>Fatal: %{customdata[1]:.2f}%<br>Serious: %{customdata[2]:.2f}%<br>n: %{customdata[0]:,}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)
