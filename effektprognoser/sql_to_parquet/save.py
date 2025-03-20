import geopandas as gpd
import pandas as pd
import os
from effektprognoser.paths import CSV_DIR, GEOJSON_DIR, PARQUET_DIR
from ..utils import verify_gdf


def make_path(dir, region, table, prefix):
    output_filename = f"{table}{prefix}"
    output_dir = os.path.join(dir, region)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)


def as_csv(gdf: pd.DataFrame | gpd.GeoDataFrame, region: str, table: str):
    df_for_csv = gdf.drop(columns=["geometry"])
    output_filepath = make_path(CSV_DIR, region, table, ".csv")
    df_for_csv.to_csv(output_filepath, index=False)


def as_geojson(gdf: pd.DataFrame | gpd.GeoDataFrame, region: str, table: str):
    gdf = verify_gdf(gdf)

    if "lp" in gdf.columns:
        gdf = gdf.drop(columns=["lp"])

    output_filepath = make_path(GEOJSON_DIR, region, table, ".geojson")
    gdf.to_file(output_filepath, driver="GeoJSON")


def as_parquet(gdf, region, table):
    gdf = verify_gdf(gdf)
    output_filepath = make_path(PARQUET_DIR, region, table, ".parquet")
    gdf.to_parquet(output_filepath)
