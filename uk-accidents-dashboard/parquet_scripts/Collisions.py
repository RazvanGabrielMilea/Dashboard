import pandas as pd
import dask.dataframe as dd
import pyarrow as pa

df = dd.read_csv("collisions_1979_latest.csv", assume_missing=True)
print(df.head())
df.info()
print(df.isna().sum())

df.isna().sum().compute()

df = df.dropna(subset=["latitude", "longitude"])
df = df.dropna(subset=['collision_severity'])
df["weather_conditions"] = df["weather_conditions"].fillna("Unknown")
df["road_surface_conditions"] = df["road_surface_conditions"].fillna("Unknown")
df["light_conditions"] = df["light_conditions"].fillna("Unknown")

df = df[
[
"collision_index",
"latitude",
"longitude",
"speed_limit",
"weather_conditions",
"road_surface_conditions",
"light_conditions",
"collision_severity"
]
]

print(df.dtypes)
df["speed_limit"] = dd.to_numeric(df["speed_limit"], errors="coerce")
df = df.dropna(subset=["speed_limit"])


df.to_parquet(
"collisions_cleaned.parquet",
engine="pyarrow",
write_index=False
)
