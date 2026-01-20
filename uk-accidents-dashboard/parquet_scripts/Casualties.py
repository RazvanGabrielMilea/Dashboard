import pandas as pd
import dask.dataframe as dd
import pyarrow as pa

df = dd.read_csv(
    "casualty_1979_latest.csv",
    assume_missing=True,
    dtype={"lsoa_of_casualty": "object"}
)

df = df.dropna(subset=["collision_index"])
df["sex_of_casualty"] = df["sex_of_casualty"].fillna(-1)
df["age_of_casualty"] = df["age_of_casualty"].fillna(-1)
df["casualty_severity"] = df["casualty_severity"].fillna(-1)

df = df[
[
"collision_index",
"casualty_reference",
"casualty_class",
"sex_of_casualty",
"age_of_casualty",
"casualty_severity",
"lsoa_of_casualty"
]
]

print(df.head())

df.to_parquet(
    "casualty_clean.parquet", 
    engine="pyarrow",
    write_index=False
    )
