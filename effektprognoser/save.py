import geopandas as gpd
import pandas as pd
import os
from effektprognoser.paths import CSV_DIR, GEOJSON_DIR, PARQUET_DIR


def _verify_gdf(gdf):
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:3006")

    if not gdf.crs == "EPSG:3006":
        gdf = gdf.set_crs("EPSG:3006")

    return gdf


def save_table_as_csv(gdf: pd.DataFrame | gpd.GeoDataFrame, region: str, table: str):
    df_for_csv = gdf.drop(columns=["geometry"])

    output_filename = f"{table}.csv"
    output_dir = os.path.join(CSV_DIR, region)

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, output_filename)

    df_for_csv.to_csv(output_filepath, index=False)


def save_table_as_geojson(
    gdf: pd.DataFrame | gpd.GeoDataFrame, region: str, table: str
):
    gdf = _verify_gdf(gdf)

    if "lp" in gdf.columns:
        gdf = gdf.drop(columns=["lp"])

    output_filename = f"{table}.geojson"
    output_dir = os.path.join(GEOJSON_DIR, region)

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, output_filename)

    gdf.to_file(output_filepath, driver="GeoJSON")


def save_table_as_parquet(gdf, region, table):
    gdf = _verify_gdf(gdf)

    output_filename = f"{table}.parquet"
    output_dir = os.path.join(PARQUET_DIR, region)

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, output_filename)

    gdf.to_parquet(output_filepath)
