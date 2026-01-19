import pandas as pd
import plotly.express as px
import streamlit as st

from lib.state import get_filters
from lib.overview_data import (
    load_monthly_by_year,
    load_hour_dow_by_year,
    load_severity_by_year,
)
from lib.assets import read_csv, asset_exists

# ----------------------------
# Fallback codebooks (STATS19-style)
# ----------------------------
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

CODEBOOKS = {
    "weather_conditions": WEATHER_MAP,
    "light_conditions": LIGHT_MAP,
    "road_surface_conditions": ROAD_SURFACE_MAP,
}


def _to_int_or_none(v):
    try:
        return int(float(v))
    except Exception:
        return None


def _rate_to_pct(series: pd.Series) -> pd.Series:
    # Handles both [0..1] rates and [0..100] already-in-percent values
    mx = series.max()
    return series * 100.0 if mx <= 1.0 else series


st.set_page_config(layout="wide")

filters = get_filters()
selected_year = filters.get("year")  # can be None

st.markdown("<div class='panel-title'>Trends</div>", unsafe_allow_html=True)

# ----------------------------
# Section 1: Monthly comparison (multi-year)
# ----------------------------
monthly = load_monthly_by_year().copy()
monthly["month_name"] = monthly["month_name"].astype(str)

years_all = sorted(monthly["year"].dropna().unique().astype(int).tolist())
default_years = (
    [int(selected_year)]
    if selected_year is not None
    else years_all[-5:] if len(years_all) > 5 else years_all
)

years_pick = st.multiselect(
    "Years to compare (Monthly)",
    options=years_all,
    default=default_years,
)

m = monthly[monthly["year"].isin(years_pick)].copy()
m = m.sort_values(["year", "month_num"])
month_order = (
    m.drop_duplicates(["month_num", "month_name"])
    .sort_values("month_num")["month_name"]
    .tolist()
)

left, right = st.columns([1.35, 1])

with left:
    st.markdown(
        "<div class='panel-title'>Seasonality: Accidents by Month (Year comparison)</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        if m.empty:
            st.info("No data for selected years.")
        else:
            fig = px.line(
                m,
                x="month_name",
                y="accidents",
                color="year",
                markers=True,
                category_orders={"month_name": month_order},
            )
            fig.update_layout(
                xaxis_title="Month", yaxis_title="Accidents", legend_title="Year"
            )
            fig.update_yaxes(tickformat=",.0f", hoverformat=",.0f")
            fig.update_traces(
                hovertemplate="Month: %{x}<br>Accidents: %{y:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Section 2: Severity share over years
# ----------------------------
sev = load_severity_by_year().copy()
sev["year"] = sev["year"].astype(int)

total_by_year = sev.groupby("year")["accidents"].sum().rename("total")
sev = sev.merge(total_by_year, on="year", how="left")
sev["share_pct"] = (sev["accidents"] / sev["total"]) * 100.0

with right:
    st.markdown(
        "<div class='panel-title'>Severity share by Year</div>", unsafe_allow_html=True
    )
    with st.container(border=True):
        if sev.empty:
            st.info("No severity data.")
        else:
            order = ["Slight", "Serious", "Fatal"]
            sev["severity"] = pd.Categorical(
                sev["severity"], categories=order, ordered=True
            )
            sev = sev.sort_values(["year", "severity"])

            fig = px.bar(
                sev, x="year", y="share_pct", color="severity", barmode="stack"
            )
            fig.update_layout(
                xaxis_title="Year", yaxis_title="Share (%)", legend_title="Severity"
            )
            fig.update_yaxes(tickformat=",.0f", hoverformat=",.2f")
            fig.update_traces(
                hovertemplate="Year: %{x}<br>Share: %{y:.2f}%<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("")

# ----------------------------
# Section 3: Hour x Day heatmap (year or sum)
# ----------------------------
hour_dow = load_hour_dow_by_year().copy()

st.markdown(
    "<div class='panel-title'>Traffic pattern: Hour × Day</div>", unsafe_allow_html=True
)
with st.container(border=True):

    mode = st.radio(
        "Heatmap mode", ["Selected year", "All selected years (sum)"], horizontal=True
    )

    if selected_year is None and mode == "Selected year":
        st.info(
            "Select a year from the sidebar to see that year’s heatmap, or switch to sum mode."
        )
    else:
        if mode == "Selected year" and selected_year is not None:
            h = hour_dow[hour_dow["year"] == int(selected_year)].copy()
        else:
            h = hour_dow[hour_dow["year"].isin(years_pick)].copy()

        if h.empty:
            st.info("No heatmap data for this selection.")
        else:
            pivot = h.pivot_table(
                index="dow",
                columns="hour",
                values="accidents",
                aggfunc="sum",
                fill_value=0,
            )
            pivot.index.name = "Day"
            pivot.columns.name = "Hour"

            order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            pivot = pivot.reindex([d for d in order if d in pivot.index])

            try:
                pivot.columns = pivot.columns.astype(int)
                pivot = pivot.reindex(sorted(pivot.columns), axis=1)
            except Exception:
                pass

            fig = px.imshow(pivot, aspect="auto")
            fig.update_layout(xaxis_title="Hour", yaxis_title="Day")
            fig.update_traces(
                hovertemplate="Day: %{y}<br>Hour: %{x}<br>Accidents: %{z:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("")

# ----------------------------
# Section 4: Factor risk (from teammate CSVs)
# ----------------------------
# Analyze how different factors correlate with accident severity
st.markdown(
    "<div class='panel-title'>What increases severity? (Factor risk)</div>",
    unsafe_allow_html=True,
)
with st.container(border=True):

    factor_opts = [
        ("Weather", "weather_conditions_severity_rates.csv", "weather_conditions"),
        ("Light", "light_conditions_severity_rates.csv", "light_conditions"),
        (
            "Road surface",
            "road_surface_conditions_severity_rates.csv",
            "road_surface_conditions",
        ),
        ("Speed limit", "speed_limit_severity_rates.csv", "speed_limit"),
    ]
    names = [x[0] for x in factor_opts]
    choice = st.selectbox("Factor", names, index=0)
    _, fname, key = next(x for x in factor_opts if x[0] == choice)

    if not asset_exists("data", "factor_tables", fname):
        st.warning(f"Missing factor table: assets/data/factor_tables/{fname}")
        st.stop()

    df = read_csv("data", "factor_tables", fname).copy()

    # Ensure percent
    df["risk_pct"] = _rate_to_pct(df["serious_or_fatal_rate"])

    # Build readable labels for categorical coded factors
    if key in CODEBOOKS and key in df.columns:
        mapping = CODEBOOKS[key]

        def label_for(v):
            vi = _to_int_or_none(v)
            if vi is None:
                return str(v)
            return mapping.get(vi, f"Code {vi}")

        df["Category"] = df[key].map(label_for)
    else:
        df["Category"] = df[key].astype(str)

    # Keep top rows for bar charts
    top_n = st.slider("Top categories (for categorical factors)", 5, 30, 15)

    if key == "speed_limit":
        # Speed limit is numeric; show as a curve
        df = df.sort_values("n", ascending=False).head(
            30
        )  # keep the most common speed limits
        df[key] = df[key].astype(float)
        df = df.sort_values(key)

        fig = px.line(df, x=key, y="risk_pct", markers=True)
        fig.update_layout(
            xaxis_title="Speed limit", yaxis_title="Serious/Fatal rate (%)"
        )
        fig.update_traces(
            hovertemplate="Speed: %{x:.0f}<br>Serious/Fatal: %{y:.2f}%<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show table"):
            st.dataframe(df, use_container_width=True)

    else:
        df_plot = df.sort_values("risk_pct", ascending=False).head(top_n)

        fig = px.bar(df_plot, x="Category", y="risk_pct")
        fig.update_layout(xaxis_title=choice, yaxis_title="Serious/Fatal rate (%)")
        fig.update_traces(
            hovertemplate=f"{choice}: %{{x}}<br>Serious/Fatal: %{{y:.2f}}%<extra></extra>"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show table"):
            st.dataframe(df_plot, use_container_width=True)
