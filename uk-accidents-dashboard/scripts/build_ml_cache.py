from pathlib import Path
import pandas as pd

"""
Build ML Cache Script.

This script creates a sampled dataset optimized for machine learning tasks from the
enriched traffic data. It extracts relevant features and targets for casualty severity
prediction, sampling the data for manageable size while preserving key patterns.
"""

# Change this if your big enriched parquet is elsewhere
SRC = Path("data/traffic_enriched.parquet")  # or data/traffic_full.parquet (folder)
OUT = Path("data/ml_cache.parquet")

# Columns used in your team notebooks + a couple optional helpful ones
COLS = [
    # target
    "casualty_severity",
    # numeric
    "latitude",
    "longitude",
    "speed_limit",
    "age_of_vehicle",
    "age_of_driver",
    "engine_capacity_cc",
    "age_of_casualty",
    # categorical
    "weather_conditions",
    "road_surface_conditions",
    "light_conditions",
    "vehicle_type",
    "sex_of_driver",
    "propulsion_code",
    "journey_purpose_of_driver",
    "casualty_class",
    "sex_of_casualty",
    # optional (if present, helps filtering later)
    "accident_date",
]


def main(sample_n: int = 250_000):
    """
    Build the ML cache by extracting and sampling relevant columns from the enriched dataset.

    This function reads the specified columns from the source parquet file, drops rows
    with missing target values, samples the data if needed, and saves the result
    as a new parquet file for ML training.

    Parameters:
    - sample_n (int): Maximum number of rows to sample. Default is 250,000.

    Returns:
    None: The function saves the processed data to disk and prints status information.
    """
    # Check if the source parquet file exists
    if not SRC.exists():
        raise FileNotFoundError(f"Source parquet not found: {SRC.resolve()}")

    # Read only the needed columns from the parquet file for efficiency
    df = pd.read_parquet(
        SRC,
        columns=[
            c
            for c in COLS
            if c in pd.read_parquet(SRC, engine="pyarrow", columns=None).columns
        ],
    )  # optional check
    # If the line above is too fancy for your setup, use this instead:
    # df = pd.read_parquet(SRC)  # then select COLS below

    # If you used the simpler read_parquet, uncomment:
    # df = df[[c for c in COLS if c in df.columns]]

    # Remove rows where the target variable (casualty_severity) is missing
    df = df.dropna(subset=["casualty_severity"]).copy()

    # Sample the dataset if it exceeds the specified sample size for faster processing
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)

    # Create the output directory if it doesn't exist
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Save the processed dataframe to parquet format
    df.to_parquet(OUT, index=False)
    # Print confirmation with file path and dimensions
    print("Saved:", OUT.resolve(), "rows:", len(df), "cols:", len(df.columns))


if __name__ == "__main__":
    main()
