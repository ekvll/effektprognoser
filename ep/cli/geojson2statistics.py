import geopandas as gpd

from typing import Optional

from tqdm import tqdm
from pathlib import Path

from ep.config import GEOJSON_TMP_DIR, GEOJSON_DIR, raps_categories
from ep.cli.parquet2geojson import save_geojson


"""
This script processes GeoJSON files and compares them to a reference GeoDataFrame.
It merges the data for each category and year, and saves the results as GeoJSON files.


The script is designed to work with the output of the parquet2geojson.py script.
"""


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
        columns={"eb_gdf": "eb", "ea_gdf": "ea", "geometry_gdf": "geometry"}
    )

    return merged


def main(region):
    tqdm.write(f"Processing region: {region}")

    filenames = geojson_tmp_filenames(region, tmp=True)

    for category in raps_categories.keys():
        filenames_filtered = [f for f in filenames if category in f]
        if len(filenames_filtered) == 0:
            tqdm.write(f"No filenames for category {category}. Skipping...")
            continue

        for filename in filenames_filtered:
            tqdm.write(f"Processing file: {filename}")
            gdf = load_geojson(filename, region, tmp=True)

            year = filename.split("_")[0]
            if year == "2022":
                ref = gdf.copy(deep=True)

            merged = merge_keep_unmutuals(gdf, ref)

            merged.fillna(10e6, inplace=True)

            merged = merged.to_crs("EPSG:4326")
            save_geojson(merged, region, year, category, tmp=False)


if __name__ == "__main__":
    regions = ["10"]
    for region in regions:
        main(region)
