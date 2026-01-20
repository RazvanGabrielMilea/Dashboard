import dask.dataframe as dd

collisions = dd.read_parquet("collisions_cleaned.parquet")
vehicles = dd.read_parquet("vehicles_clean.parquet")
casualties = dd.read_parquet("casualty_clean.parquet")

print("Collisions sample:")

coll_veh = collisions.merge(
    vehicles,
    on="collision_index",
    how="inner"
)

print("Collisions + Vehicles sample:")

full = coll_veh.merge(
    casualties,
    on=["collision_index"],
    how="left"
)


full.to_parquet("traffic_full.parquet", engine="pyarrow")

print("Fisierul final traffic_full.parquet a fost creat.")
