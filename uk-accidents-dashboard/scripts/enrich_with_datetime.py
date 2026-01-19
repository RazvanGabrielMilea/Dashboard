"""
Enrich accident data with datetime information.

This script takes the raw accident CSV file containing date and time columns,
parses and extracts datetime features (accident_date and accident_hour),
and merges this information with the main accident facts parquet file.
The result is an enriched dataset with proper datetime columns for analysis.
"""

from pathlib import Path
import pandas as pd

# Define file paths for input and output data
FACT_PARQUET = Path("data/uk_accidents.parquet")  # Main accident facts parquet file (large dataset ~10M rows)
ACC_CSV = Path(
    "data/dft-road-casualty-statistics-collision-1979-latest-published-year.csv"
)  # Raw CSV containing date and time information
OUT_PARQUET = Path("data/traffic_enriched.parquet")  # Output file with enriched datetime data

# Load only the necessary columns from the accidents CSV to optimize memory usage
acc = pd.read_csv(ACC_CSV, usecols=["collision_index", "date", "time"])

# Parse date/time columns safely, handling UK date format (day/month/year)
# Using dayfirst=True for UK date format, errors='coerce' converts invalid dates to NaT
acc["accident_date"] = pd.to_datetime(acc["date"], errors="coerce", dayfirst=True)

# Extract hour from time strings (format like "12:10" -> 12)
# Convert time column to string, strip whitespace, split on colon, take first part
t = acc["time"].astype(str).str.strip()
acc["accident_hour"] = pd.to_numeric(t.str.split(":").str[0], errors="coerce")

# Keep only essential columns and remove duplicate collision_index entries
# This ensures clean data for merging with the main dataset
acc = acc[["collision_index", "accident_date", "accident_hour"]].drop_duplicates(
    "collision_index"
)

# Load the main accident facts parquet file
df = pd.read_parquet(FACT_PARQUET)

# Merge datetime information with main dataset using left join
# This preserves all rows from the main dataset while adding datetime columns
df = df.merge(acc, on="collision_index", how="left")

# Save the enriched dataset to parquet format for efficient storage and loading
df.to_parquet(OUT_PARQUET, index=False)

# Print completion status with file info and data quality metrics
print("Done:", OUT_PARQUET, df.shape, "null dates:", df["accident_date"].isna().mean())
