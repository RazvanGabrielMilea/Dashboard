from pathlib import Path
import json
import pandas as pd

"""
Build Fast Cache Script.

This script creates optimized cache files for the dashboard from the enriched traffic data.
It aggregates casualty-level data to collision-level and generates metadata for fast filtering.
The resulting cache enables quick dashboard loading without processing the full dataset.
"""

# Input parquet (change if needed)
IN_PARQUET = Path("data/traffic_enriched.parquet")

# Output cache folder
OUT_DIR = Path("data/dashboard_cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COLLISIONS = OUT_DIR / "collisions_fast.parquet"
OUT_META = OUT_DIR / "metadata.json"

SEVERITY_MAP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
    1.0: "Fatal",
    2.0: "Serious",
    3.0: "Slight",
}

# Only the columns we need for the dashboard
COLS = [
    "collision_index",
    "accident_date",
    "accident_hour",
    "collision_severity",
    "casualty_severity",
    "lsoa_of_casualty",
]

# Load and preprocess data
print("Reading parquet (only needed columns)...")
df = pd.read_parquet(IN_PARQUET, columns=COLS)
df.columns = [c.strip() for c in df.columns]

# Convert data types for consistency
df["accident_date"] = pd.to_datetime(df["accident_date"], errors="coerce")
df["accident_hour"] = pd.to_numeric(df["accident_hour"], errors="coerce")

# Precompute killed flag for faster aggregation
df["_killed"] = (df["casualty_severity"] == 1).astype("int8")

# Aggregate to collision-level data
print("Aggregating to collision-level table (1 row per collision)...")
coll = (
    df.groupby("collision_index", sort=False)
    .agg(
        accident_date=("accident_date", "first"),
        accident_hour=("accident_hour", "first"),
        collision_severity=("collision_severity", "first"),
        lsoa_of_casualty=("lsoa_of_casualty", "first"),
        persons_involved=(
            "collision_index",
            "size",
        ),  # number of rows in original (casualty-level)
        persons_killed=("_killed", "sum"),
    )
    .reset_index()
)

coll.to_parquet(OUT_COLLISIONS, index=False)
print("Wrote:", OUT_COLLISIONS, "rows:", len(coll), "cols:", coll.shape[1])

# Build metadata for fast dropdowns and filters
print("Building metadata for fast dropdowns...")
years = sorted([int(x) for x in coll["accident_date"].dt.year.dropna().unique()])

sev_labels = (
    pd.Series(coll["collision_severity"])
    .map(SEVERITY_MAP)
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
# keep nice order
order = ["Slight", "Serious", "Fatal"]
sev_labels = [x for x in order if x in sev_labels] + [
    x for x in sorted(sev_labels) if x not in order
]

top_lsoas = ["All"]
if "lsoa_of_casualty" in coll.columns:
    top_lsoas += (
        coll["lsoa_of_casualty"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(200)
        .index.tolist()
    )

meta = {"years": years, "severity_labels": sev_labels or order, "top_lsoas": top_lsoas}
OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Wrote:", OUT_META)
print("Done.")
