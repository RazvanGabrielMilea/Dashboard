import pandas as pd
import dask.dataframe as dd
import pyarrow as pa

df = dd.read_csv(
    "vehicles_1979_latest.csv",
    assume_missing=True
)

print(df.head())

print(df.columns)

print("\nValori lipsă pe coloană:")
print(df.isna().sum().compute())

if "collision_index" in df.columns:
    df = df.dropna(subset=["collision_index"])

if "vehicle_reference" in df.columns:
    df = df.dropna(subset=["vehicle_reference"])

for col in [
    "vehicle_type",
    "propulsion_code",
    "journey_purpose_of_driver",
]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

for col in [
    "age_of_vehicle",
    "age_of_driver",
    "engine_capacity_cc",
]:
    if col in df.columns:
        df[col] = df[col].fillna(-1)

important_cols = [
    "collision_index",
    "vehicle_reference",
    "vehicle_type",
    "age_of_vehicle",
    "age_of_driver",
    "sex_of_driver",
    "engine_capacity_cc",
    "propulsion_code",
    "journey_purpose_of_driver",
]

existing_cols = [c for c in important_cols if c in df.columns]
df = df[existing_cols]

print(existing_cols)

df.to_parquet(
    "vehicles_clean.parquet",
    engine="pyarrow",
    write_index=False
)
