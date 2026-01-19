"""
Precompute overview tables for dashboard performance.

This script processes collision data to create pre-aggregated tables that are used
by the dashboard overview page. It generates three key summary tables:
- Monthly accident counts by year
- Hourly accident patterns by day of week and year
- Accident severity distribution by year

These precomputed tables improve dashboard loading speed by avoiding
real-time aggregations on large datasets.
"""

from pathlib import Path
import pandas as pd

# Define input file path for collision data (processed collision-level data)
IN_COLLISIONS = Path("data/dashboard_cache/collisions_fast.parquet")

# Define output directory for overview tables (organized within assets)
OUT_DIR = Path("assets/data/overview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define output file paths for each precomputed table
OUT_MONTHLY = OUT_DIR / "monthly_by_year.csv"
OUT_HOUR_DOW = OUT_DIR / "hour_dow_by_year.csv"
OUT_SEVERITY = OUT_DIR / "severity_by_year.csv"

# Mapping dictionary to convert numeric severity codes to human-readable labels
SEVERITY_MAP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
    1.0: "Fatal",
    2.0: "Serious",
    3.0: "Slight",
}

print("Reading collisions_fast.parquet ...")
# Load collision data with only the columns needed for overview calculations
df = pd.read_parquet(
    IN_COLLISIONS,
    columns=["collision_index", "accident_date", "accident_hour", "collision_severity"],
)

# Ensure date and hour columns are properly typed
df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")
df["accident_hour"] = pd.to_numeric(df["accident_hour"], errors="coerce")

# Extract temporal features for grouping and analysis
df["year"] = df["accident_date"].dt.year.astype("Int64")
df["month_num"] = df["accident_date"].dt.month.astype("Int64")
df["month_name"] = df["accident_date"].dt.strftime("%B")
df["dow"] = df["accident_date"].dt.day_name()
df["hour"] = df["accident_hour"].astype("Int64")

# ---- Monthly by year ----
print("Computing monthly_by_year ...")
# Aggregate accidents by year and month, counting unique collision_index values
monthly = (
    df.dropna(subset=["year", "month_num"])
    .groupby(["year", "month_num", "month_name"])["collision_index"]
    .nunique()
    .reset_index(name="accidents")
    .sort_values(["year", "month_num"])
)
monthly.to_csv(OUT_MONTHLY, index=False)
print("Wrote:", OUT_MONTHLY)

# ---- Hour x Day of week by year ----
print("Computing hour_dow_by_year ...")
# Define the order for days of the week (Monday first)
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# Filter data and prepare for hour-by-day analysis
df2 = df.dropna(subset=["year", "dow", "hour"]).copy()
df2 = df2[df2["hour"].between(0, 23)]
df2["dow"] = pd.Categorical(df2["dow"], categories=order, ordered=True)

# Aggregate accidents by year, day of week, and hour
hour_dow = (
    df2.groupby(["year", "dow", "hour"])["collision_index"]
    .nunique()
    .reset_index(name="accidents")
    .sort_values(["year", "dow", "hour"])
)
hour_dow.to_csv(OUT_HOUR_DOW, index=False)
print("Wrote:", OUT_HOUR_DOW)

# ---- Severity split by year ----
print("Computing severity_by_year ...")
# Prepare data for severity analysis
df3 = df.dropna(subset=["year", "collision_severity"]).copy()
# Map numeric severity codes to readable labels, fallback to string conversion
df3["severity"] = (
    df3["collision_severity"]
    .map(SEVERITY_MAP)
    .fillna(df3["collision_severity"].astype(str))
)

# Aggregate accidents by year and severity level
severity = (
    df3.groupby(["year", "severity"])["collision_index"]
    .nunique()
    .reset_index(name="accidents")
    .sort_values(["year", "severity"])
)
severity.to_csv(OUT_SEVERITY, index=False)
print("Wrote:", OUT_SEVERITY)

print("Done ✅")
