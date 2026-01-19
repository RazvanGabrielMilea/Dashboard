import pandas as pd
import plotly.express as px
import streamlit as st

from lib.insights import overview_insights
from lib.state import get_filters
from lib.data import load_data, apply_filters, compute_kpis
from lib.overview_data import (
    load_monthly_by_year,
    load_hour_dow_by_year,
    load_severity_by_year,
)

"""
Overview Page for UK Accidents Dashboard.

This Streamlit page provides a high-level overview of accident data with key metrics,
insights, and visualizations. It displays KPIs, temporal patterns, severity distributions,
and traffic patterns using precomputed data for fast loading.
"""

st.set_page_config(layout="wide")

# Get current dashboard filters
filters = get_filters()
year = filters.get("year")

# Load filtered data and compute KPIs
df = load_data()
df_f = apply_filters(df, filters)
k = compute_kpis(df_f)

# ---- KPI row ----
# Display key performance indicators
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Accidents", f"{k['total_accidents']:,}")
c2.metric(
    "Fatal Accidents",
    "—" if k["fatal_accidents"] is None else f"{k['fatal_accidents']:,}",
)
c3.metric(
    "Persons involved",
    "—" if k["persons_involved"] is None else f"{k['persons_involved']:,}",
)
c4.metric(
    "Avg Accidents / Day",
    "—" if k["avg_accidents_per_day"] is None else f"{k['avg_accidents_per_day']:.2f}",
)

st.markdown("")

# ----------------------------
# Insights (fast, from precomputed tables)
# ----------------------------
# Generate quick insights from precomputed data
ins = overview_insights(year)

st.markdown("<div class='panel-title'>Quick Insights</div>", unsafe_allow_html=True)
with st.container(border=True):
    i1, i2, i3, i4 = st.columns(4)

    if "peak_month_name" in ins:
        i1.metric(
            "Peak month",
            ins["peak_month_name"],
            help=f"Accidents: {ins['peak_month_accidents']:,}",
        )
    else:
        i1.metric("Peak month", "—")

    if "peak_dow" in ins:
        i2.metric(
            "Peak time",
            f"{ins['peak_dow']} @ {ins['peak_hour']:02d}:00",
            help=f"Accidents: {ins['peak_dow_hour_accidents']:,}",
        )
    else:
        i2.metric("Peak time", "—")

    sf = ins.get("serious_fatal_share_pct")
    i3.metric("Serious+Fatal share", "—" if sf is None else f"{sf:.2f}%")

    # Pick the highest-risk among weather/light/road-surface
    candidates = [
        x
        for x in [
            ins.get("top_weather"),
            ins.get("top_light"),
            ins.get("top_road_surface"),
        ]
        if x
    ]
    if candidates:
        label, risk, n = sorted(candidates, key=lambda t: t[1], reverse=True)[0]
        suffix = f" (n={n:,})" if n is not None else ""
        i4.metric("Highest risk factor", f"{risk:.2f}%", help=f"{label}{suffix}")
    else:
        i4.metric("Highest risk factor", "—")

# ---- Load precomputed tables ----
# Load overview data for visualizations
monthly = load_monthly_by_year()
hour_dow = load_hour_dow_by_year()
severity = load_severity_by_year()

# Filter by year (most common case)
if year is not None:
    monthly = monthly[monthly["year"] == int(year)]
    hour_dow = hour_dow[hour_dow["year"] == int(year)]
    severity = severity[severity["year"] == int(year)]

# Create layout columns for charts
left, right = st.columns([1.35, 1])

# =========================
# Accidents Over Time
# =========================
# Display monthly accident trends
with left:
    st.markdown(
        "<div class='panel-title'>Accidents Over Time</div>", unsafe_allow_html=True
    )
    with st.container(border=True):
        if monthly.empty:
            st.info("No data for selected filters.")
        else:
            # Keep Jan..Dec order
            monthly = monthly.sort_values("month_num")
            fig = px.line(monthly, x="month_name", y="accidents", markers=True)
            fig.update_layout(xaxis_title="Month", yaxis_title="Accidents")
            fig.update_xaxes(
                categoryorder="array", categoryarray=monthly["month_name"].tolist()
            )

            # raw numbers, no k
            fig.update_yaxes(tickformat=",.0f", hoverformat=",.0f")
            fig.update_traces(
                hovertemplate="Month: %{x}<br>Accidents: %{y:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================
# Severity split (quick extra panel)
# =========================
# Display accident severity distribution
with right:
    st.markdown("<div class='panel-title'>Severity Split</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if severity.empty:
            st.info("No data for selected filters.")
        else:
            order = ["Slight", "Serious", "Fatal"]
            severity["severity"] = pd.Categorical(
                severity["severity"], categories=order, ordered=True
            )
            severity = severity.sort_values("severity")

            fig = px.bar(severity, x="severity", y="accidents")
            fig.update_layout(xaxis_title="Severity", yaxis_title="Accidents")
            fig.update_yaxes(tickformat=",.0f", hoverformat=",.0f")
            fig.update_traces(
                hovertemplate="Severity: %{x}<br>Accidents: %{y:,.0f}<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)

st.markdown("")

# =========================
# Heatmap Hour x Day
# =========================
# Display traffic pattern heatmap
st.markdown(
    "<div class='panel-title'>Accidents by Hour of Day and Day of Week</div>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    if hour_dow.empty:
        st.info("No data for selected filters.")
    else:
        # Pivot into matrix for imshow
        pivot = hour_dow.pivot_table(
            index="dow", columns="hour", values="accidents", fill_value=0
        )

        pivot.index.name = "Day"
        pivot.columns.name = "Hour"

        # Ensure hour ordering
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
