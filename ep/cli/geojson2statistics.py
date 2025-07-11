"""
This script processes GeoJSON files and compares them to a reference GeoDataFrame.
It merges the data for each category and year, and saves the results as GeoJSON files.

The following processes are performed:
1. Load GeoJSON files for a specific region.
2. For each category, filter the filenames and load the corresponding GeoDataFrame.
3. Merge the GeoDataFrame with a reference GeoDataFrame based on 'rid'.
4. Calculate differences and percent changes for 'eb' and 'ea' columns.
5. Drop unnecessary columns and rename the remaining ones.
6. Save the processed GeoDataFrame as a new GeoJSON file.

The script is designed to work with the output of the parquet2geojson.py script.
"""

import sys
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from ep.cli.parquet2geojson import save_geojson
from ep.config import GEOJSON_DIR, GEOJSON_TMP_DIR, raps_categories


def geojson_tmp_filenames(region: str, tmp: bool = False) -> list[str]:
    """Get the list of parquet filenames for a given region."""
    if tmp:
        region_path = Path(GEOJSON_TMP_DIR) / region
    else:
        region_path = Path(GEOJSON_DIR) / region

    if not region_path.exists():
        raise FileNotFoundError(f"Region path {region_path} does not exist")

    filenames = [
        f.name for f in region_path.iterdir() if f.is_file() and f.suffix == ".geojson"
    ]

    if not filenames:
        raise FileNotFoundError(f"No GeoJSON files found in {region_path}")

    return sorted(filenames)


def load_geojson(
    filename: str, region: str, cols: Optional[list[str]] = None, tmp: bool = False
) -> gpd.GeoDataFrame:
    """Load a GeoJSON file into a GeoDataFrame."""

    if tmp:
        file_path = Path(GEOJSON_TMP_DIR) / region / filename
    else:
        file_path = Path(GEOJSON_DIR) / region / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist")
    gdf = gpd.read_file(file_path)
    return gdf[cols] if cols else gdf


def merge_keep_unmutuals(
    gdf: gpd.GeoDataFrame, ref: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Outer merge keeps all rows, including non-matching 'rid's
    merged = gdf.merge(ref, on="rid", how="left", suffixes=("_gdf", "_ref"))

    # Compute differences and percent changes, allowing NaN to propagate
    merged["ebd"] = merged["eb_gdf"] - merged["eb_ref"]
    merged["ebp"] = ((merged["eb_gdf"] - merged["eb_ref"]) / merged["eb_ref"]) * 100

    merged["ead"] = merged["ea_gdf"] - merged["ea_ref"]
    merged["eap"] = ((merged["ea_gdf"] - merged["ea_ref"]) / merged["ea_ref"]) * 100

    merged = merged.drop(
        columns=[c for c in merged.columns if "_ref" in c or "lp" in c]
    )

    merged = merged.rename(
        columns={
            "eb_gdf": "eb",
            "ea_gdf": "ea",
            "geometry_gdf": "geometry",
            "kn_gdf": "kn",
            "kk_gdf": "kk",
            "natbolag_gdf": "natbolag",
            "filename_gdf": "filename",
            "category_gdf": "category",
        }
    )

    return merged


def replace_negative_val(
    df: pd.DataFrame | gpd.GeoDataFrame, col: str, to_val: int | float
) -> pd.DataFrame | gpd.GeoDataFrame:
    df[col] = df[col].astype(float)
    df.loc[df[col] < 0, col] = to_val
    return df


def main(region):
    tqdm.write(f"Processing region: {region}")

    filenames = geojson_tmp_filenames(region, tmp=True)

    for category in raps_categories.keys():
        filenames_filtered = [f for f in filenames if category in f]
        if len(filenames_filtered) == 0:
            tqdm.write(f"No filenames for category {category}. Skipping...")
            continue

        for filename in tqdm(
            filenames_filtered, desc="Filenames", position=1, leave=False
        ):
            # tqdm.write(filename)
            gdf = load_geojson(filename, region, tmp=True)

            year = filename.split("_")[0]
            if year == "2022":
                ref = gdf.copy(deep=True)

            merged = merge_keep_unmutuals(gdf, ref)
            if category == "total":
                for idx, row in merged.iterrows():
                    fn = row.filename.replace("_V1.parquet", "")
                    # tqdm.write(fn)
                    if any(sub in fn for sub in ["Smahus", "Flerbostadshus"]):
                        # tqdm.write(fn)
                        merged.loc[idx, "ebp"] = 10e6
                        merged.loc[idx, "eap"] = 10e6
                        merged.loc[idx, "ebd"] = 10e6
                        merged.loc[idx, "ead"] = 10e6
                    elif any(sub in fn for sub in ["TT", "PB", "LL"]):
                        merged.loc[idx, "ebp"] = 10e6
                        merged.loc[idx, "eap"] = 10e6
                        merged.loc[idx, "ebd"] = row["eb"]

            else:
                if category == "bostader":
                    merged["ebp"] = merged["ebp"].fillna(10e6)
                    merged["eap"] = merged["eap"].fillna(10e6)
                    merged["ebd"] = merged["ebd"].fillna(10e6)
                    merged["ead"] = merged["ead"].fillna(10e6)
                if category == "transport":
                    merged["ebp"] = merged["ebp"].fillna(10e6)
                    merged["eap"] = merged["eap"].fillna(10e6)
                    merged["ebd"] = merged["ebd"].fillna(merged["eb"])
                    merged = replace_negative_val(merged, "ebd", 0.1)

            merged = merged.to_crs("EPSG:4326")

            save_geojson(merged, region, year, category, tmp=False)


if __name__ == "__main__":
    tqdm.write("geojson2statistics")

    from ep.config import regions_alla

    # regions = ["10"]
    for region in tqdm(regions_alla, desc="Regions", position=0, leave=False):
        main(region)
