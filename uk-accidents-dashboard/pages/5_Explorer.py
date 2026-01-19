from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from lib.state import get_filters
from lib.data import load_data, apply_filters
from lib.overview_data import (
    load_monthly_by_year,
    load_hour_dow_by_year,
    load_severity_by_year,
)
from lib.assets import asset_exists, read_csv

st.set_page_config(layout="wide")

"""
Explorer Page for UK Accidents Dashboard.

This Streamlit page provides tools for exploring and exporting the accident data.
It includes data sampling, quality profiling, and various export options for
filtered datasets and precomputed tables. The page respects the current dashboard
filters and allows users to inspect data quality and download subsets.
"""

filters = get_filters()
year = filters.get("year")
lsoa = filters.get("lsoa", "All")
severity = filters.get("severity", [])

df = load_data()  # collision-level (fast cache)
df_f = apply_filters(df, filters)  # filtered collision-level


# ----------------------------
# Helpers
# ----------------------------
def _existing(cols: list[str]) -> list[str]:
    """
    Filter a list of column names to only include those present in the filtered DataFrame.

    Parameters:
    - cols (list[str]): List of column names to check.

    Returns:
    - list[str]: Subset of column names that exist in df_f.
    """
    return [c for c in cols if c in df_f.columns]


def _bytes_csv(dfx: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to CSV bytes for download.

    Parameters:
    - dfx (pd.DataFrame): The DataFrame to convert.

    Returns:
    - bytes: UTF-8 encoded CSV data.
    """
    return dfx.to_csv(index=False).encode("utf-8")


def _bytes_csv_from_df(dfx: pd.DataFrame, name: str):
    """
    Create a download button for a DataFrame as CSV.

    Parameters:
    - dfx (pd.DataFrame): The DataFrame to download.
    - name (str): Base name for the file and button label.
    """
    b = _bytes_csv(dfx)
    st.download_button(
        f"Download {name} (CSV)",
        data=b,
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


@st.cache_data(show_spinner=False)
def compute_quality_table(
    dfx: pd.DataFrame, cols: list[str], sample_n: int | None
) -> pd.DataFrame:
    """
    Computes basic quality statistics for the selected columns in a DataFrame.

    This function analyzes data quality by calculating null percentages, unique value counts,
    and min/max values for numeric columns. It can optionally sample the data for performance.

    Parameters:
    - dfx (pd.DataFrame): The DataFrame to analyze.
    - cols (list[str]): List of column names to include in the analysis.
    - sample_n (int | None): Number of rows to sample for analysis. If None, uses full dataset.

    Returns:
    - pd.DataFrame: A DataFrame with quality statistics for each column, sorted by null % and unique count.
    """
    if dfx.empty or not cols:
        return pd.DataFrame()

    work = dfx[cols].copy()

    if sample_n is not None and len(work) > sample_n:
        work = work.sample(sample_n, random_state=42)

    out_rows = []
    n = len(work)

    for c in cols:
        s = work[c]
        null_count = int(s.isna().sum())
        null_pct = (null_count / n * 100.0) if n else 0.0
        nunique = int(s.nunique(dropna=True))

        row = {
            "column": c,
            "dtype": str(s.dtype),
            "rows_analyzed": n,
            "null_count": null_count,
            "null_pct": round(null_pct, 2),
            "unique_count": nunique,
        }

        # numeric min/max
        if pd.api.types.is_numeric_dtype(s):
            row["min"] = None if s.dropna().empty else float(s.min())
            row["max"] = None if s.dropna().empty else float(s.max())
        else:
            row["min"] = None
            row["max"] = None

        out_rows.append(row)

    q = pd.DataFrame(out_rows).sort_values(
        ["null_pct", "unique_count"], ascending=[False, False]
    )
    return q


# ----------------------------
# Page header
# ----------------------------
st.markdown("<div class='panel-title'>Explorer</div>", unsafe_allow_html=True)
with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered accidents", f"{len(df_f):,}")
    c2.metric("Year", "All" if year is None else str(year))
    c3.metric("LSOA", "All" if lsoa == "All" else str(lsoa))
    c4.metric(
        "Severity",
        (
            "All"
            if not severity
            else ", ".join(map(str, severity))[:35]
            + ("…" if len(", ".join(map(str, severity))) > 35 else "")
        ),
    )

st.markdown("")

# Create tabs for different exploration functionalities
tabs = st.tabs(["Sample (Filtered)", "Data Quality", "Downloads"])

# =====================================================================
# TAB 1: Sample (Filtered)
# =====================================================================
with tabs[0]:
    # Section: Display a sample of the filtered accident data
    st.markdown("<div class='panel-title'>Sample rows</div>", unsafe_allow_html=True)
    with st.container(border=True):
        a, b, c = st.columns([1, 1, 2])

        sample_mode = a.selectbox("Mode", ["Head", "Random sample"], index=1)
        n_rows = b.slider("Rows", 50, 2000, 200, step=50)
        seed = c.number_input(
            "Random seed", min_value=0, max_value=999999, value=42, step=1
        )

        default_cols = _existing(
            [
                "collision_index",
                "accident_date",
                "accident_hour",
                "_dow",
                "_month",
                "_collision_sev",
                "persons_involved",
                "persons_killed",
                "lsoa_of_casualty",
            ]
        )

        cols = st.multiselect(
            "Columns to display",
            options=list(df_f.columns),
            default=default_cols if default_cols else list(df_f.columns)[:10],
        )

        if df_f.empty:
            st.info("No rows match the current filters.")
        else:
            view = df_f[cols] if cols else df_f

            if sample_mode == "Random sample":
                n_take = min(int(n_rows), len(view))
                view = view.sample(n_take, random_state=int(seed))
            else:
                view = view.head(int(n_rows))

            st.dataframe(view, use_container_width=True, height=520)

# =====================================================================
# TAB 2: Data Quality
# =====================================================================
with tabs[1]:
    # Section: Analyze data quality and profiling for selected columns
    st.markdown(
        "<div class='panel-title'>Data quality & profiling</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.write(
            "Compute quick statistics for selected columns (null %, uniques, min/max)."
        )

        col_left, col_right = st.columns([2, 1])

        with col_left:
            # Suggested “useful” columns first
            suggested = _existing(
                [
                    "accident_date",
                    "accident_hour",
                    "_year",
                    "_month",
                    "_dow",
                    "collision_severity",
                    "_collision_sev",
                    "persons_involved",
                    "persons_killed",
                    "lsoa_of_casualty",
                ]
            )
            selected_cols = st.multiselect(
                "Columns to analyze",
                options=list(df_f.columns),
                default=suggested if suggested else list(df_f.columns)[:12],
            )

        with col_right:
            mode = st.radio(
                "Computation mode",
                ["Quick (sample rows)", "Full (may be slower)"],
                index=0,
            )
            sample_n = st.number_input(
                "Sample size (quick mode)",
                min_value=10_000,
                max_value=500_000,
                value=100_000,
                step=10_000,
                disabled=(mode != "Quick (sample rows)"),
            )

        run = st.button(
            "Compute quality table", type="primary", use_container_width=True
        )

    if run:
        with st.spinner("Computing quality stats..."):
            q = compute_quality_table(
                df_f,
                selected_cols,
                sample_n=None if mode.startswith("Full") else int(sample_n),
            )

        if q.empty:
            st.info("Nothing to analyze (empty selection or filtered dataset).")
        else:
            st.dataframe(q, use_container_width=True, height=420)

            # Top values for a chosen column (categorical)
            cat_cols = [
                c for c in selected_cols if not pd.api.types.is_numeric_dtype(df_f[c])
            ]
            if cat_cols:
                st.markdown(
                    "<div class='panel-title'>Top values (categorical)</div>",
                    unsafe_allow_html=True,
                )
                with st.container(border=True):
                    chosen = st.selectbox("Column", options=cat_cols, index=0)
                    topk = st.slider("Top K", 5, 30, 10)

                    work = df_f[chosen]
                    if mode.startswith("Quick") and len(df_f) > int(sample_n):
                        work = work.sample(int(sample_n), random_state=42)

                    vc = (
                        work.astype(str)
                        .value_counts(dropna=False)
                        .head(int(topk))
                        .reset_index()
                    )
                    vc.columns = ["value", "count"]
                    st.dataframe(vc, use_container_width=True)

# =====================================================================
# TAB 3: Downloads
# =====================================================================
with tabs[2]:
    # Section: Export options for data and precomputed tables
    st.markdown("<div class='panel-title'>Exports</div>", unsafe_allow_html=True)

    # ----------------------------
    # A) Export filtered accidents (collision-level)
    # ----------------------------
    # Subsection: Export the filtered collision-level accident data
    st.markdown(
        "<div class='panel-title'>Filtered accidents (collision-level)</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.write(
            "To keep the app responsive, export is prepared only when you click the button."
        )
        max_rows = st.number_input(
            "Max rows to export",
            min_value=1_000,
            max_value=500_000,
            value=100_000,
            step=10_000,
        )
        export_mode = st.selectbox("Export mode", ["Head", "Random sample"], index=0)
        export_seed = st.number_input(
            "Export seed (for random sample)",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

        export_cols_default = _existing(
            [
                "collision_index",
                "accident_date",
                "accident_hour",
                "_dow",
                "_month",
                "_year",
                "_collision_sev",
                "persons_involved",
                "persons_killed",
                "lsoa_of_casualty",
            ]
        )
        export_cols = st.multiselect(
            "Columns to export",
            options=list(df_f.columns),
            default=(
                export_cols_default if export_cols_default else list(df_f.columns)[:15]
            ),
        )

        if st.button("Prepare CSV export", use_container_width=True, type="primary"):
            if df_f.empty:
                st.warning("No rows to export for the current filters.")
            else:
                dfx = df_f[export_cols] if export_cols else df_f

                n_take = min(int(max_rows), len(dfx))
                if export_mode == "Random sample":
                    dfx = dfx.sample(n_take, random_state=int(export_seed))
                else:
                    dfx = dfx.head(n_take)

                st.session_state["export_filtered_csv"] = _bytes_csv(dfx)
                st.session_state["export_filtered_name"] = (
                    f"filtered_accidents_{'all' if year is None else year}"
                )

        if "export_filtered_csv" in st.session_state:
            st.download_button(
                "Download prepared CSV",
                data=st.session_state["export_filtered_csv"],
                file_name=f"{st.session_state.get('export_filtered_name','filtered_accidents')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("")

    # ----------------------------
    # B) Export precomputed overview tables
    # ----------------------------
    # Subsection: Export precomputed tables used by dashboard charts
    st.markdown(
        "<div class='panel-title'>Precomputed tables used by charts</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        include_all_years = st.checkbox(
            "Export all years (ignore sidebar year)", value=False
        )

        monthly = load_monthly_by_year().copy()
        hour_dow = load_hour_dow_by_year().copy()
        sev = load_severity_by_year().copy()

        if not include_all_years and year is not None:
            monthly = monthly[monthly["year"] == int(year)]
            hour_dow = hour_dow[hour_dow["year"] == int(year)]
            sev = sev[sev["year"] == int(year)]

        c1, c2, c3 = st.columns(3)
        with c1:
            _bytes_csv_from_df(
                monthly,
                f"monthly_by_year_{'all' if include_all_years or year is None else year}",
            )
        with c2:
            _bytes_csv_from_df(
                hour_dow,
                f"hour_dow_by_year_{'all' if include_all_years or year is None else year}",
            )
        with c3:
            _bytes_csv_from_df(
                sev,
                f"severity_by_year_{'all' if include_all_years or year is None else year}",
            )

    st.markdown("")

    # ----------------------------
    # C) Export teammate factor tables (as-is)
    # ----------------------------    # Subsection: Export factor analysis tables created by team members    
    st.markdown("<div class='panel-title'>Factor tables (team exports)</div>",
        unsafe_allow_html=True,
    )
    with st.container(border=True):

        factor_files = [
            "weather_conditions_severity_rates.csv",
            "light_conditions_severity_rates.csv",
            "road_surface_conditions_severity_rates.csv",
            "speed_limit_severity_rates.csv",
        ]

        available = [
            f for f in factor_files if asset_exists("data", "factor_tables", f)
        ]
        missing = [f for f in factor_files if f not in available]

        if missing:
            st.caption(
                "Missing (not found in assets/data/factor_tables): "
                + ", ".join(missing)
            )

        if available:
            chosen = st.selectbox("Choose factor table", options=available, index=0)
            df_factor = read_csv("data", "factor_tables", chosen)
            st.dataframe(df_factor.head(50), use_container_width=True)

            st.download_button(
                "Download selected factor table (CSV)",
                data=_bytes_csv(df_factor),
                file_name=chosen,
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No factor tables found in assets/data/factor_tables.")
