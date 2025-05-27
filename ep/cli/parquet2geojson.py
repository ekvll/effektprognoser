import os
import numpy as np
import geopandas as gpd

from tqdm import tqdm

from ep.config import (
    GEOJSON_TMP_DIR,
    raps_categories,
    default_years,
    GEOJSON_DIR,
)
from ep.cli.sql2parquet import parquet_filenames, load_parquet

"""
This script processes parquet files and converts them into GeoJSON format.
It groups the parquet files into categories based on their filenames,
merges the data for each category and year, and saves the results as GeoJSON files.
"""


def extract_raps_from_filename(filename: str) -> str:
    """
    Extract the category from filename.

    Example:
        filename = "EF_2022_RAPS_16_V1.parquet"
        category = extract_raps_from_filename(filename)
        print(category)  # Output: "RAPS 16"

    Args:
        filename (str): Filename to extract category from.

    Returns:
        str: The category extracted from filename.
    """
    category = " ".join(filename.split("_")[2:-1])
    return category


def group_raps_into_categories_from_filenames(filenames: list[str]) -> dict:
    categories = {}

    for key, values in raps_categories.items():
        if key not in categories:
            categories[key] = []

        for file in filenames:
            category = extract_raps_from_filename(file)
            if category in values:
                categories[key].append(file)
    return categories


def process_raps_grouped(raps_grouped: dict):
    # Iterate over each category (bostader, transport, etc.) and all filenames associated with it
    for category, filenames in raps_grouped.items():
        # Check if the category have any files
        if len(filenames) == 0:
            tqdm.write(f"Category '{category}' contain no files. Skipping...")
            continue

        # Iterate over each year and filter the filenames for that year
        for year in default_years:
            # tqdm.write(f"Processing category '{category}' for year {year}.")

            filenames_filtered = [f for f in filenames if str(year) in f]

            # Check if there are any filenames for the current year
            if len(filenames_filtered) == 0:
                tqdm.write(f"No filenames for year {year}. Skipping...")
                continue

            # Process the filtered filenames
            yield category, year, filenames_filtered


def merge_filenames_per_year(region, filenames: list[str], year) -> dict:
    result = {}

    for filename in filenames:
        gdf = load_parquet(filename, region)

        if gdf.empty:
            tqdm.write(f"DataFrame is empty for {filename}. Skipping...")
            continue

        unique_ids = gdf.rid.unique()

        for rid in unique_ids:
            gdf_rid = gdf.loc[gdf.rid == rid]

            if len(gdf_rid) != 1:
                # Each 'rid' is expected to have one row
                raise ValueError

            if rid not in result:
                result[rid] = {
                    "lp": np.zeros(8784 if year == "2040" else 8760),
                    "geometry": gdf_rid.geometry,
                }

            result[rid]["lp"] += gdf_rid.lp.to_numpy()[0]
    return result


def restructure_merged_filenames(merged_filenames: dict) -> gpd.GeoDataFrame:
    records = []
    for rid, loadprofile in merged_filenames.items():
        records.append(
            {
                "rid": rid,
                "lp": loadprofile["lp"],
                "eb": loadprofile["lp"].max(),
                "ea": loadprofile["lp"].sum(),
                "geometry": loadprofile["geometry"].values[0],
            }
        )
    return gpd.GeoDataFrame(records, crs="EPSG:3006")


def save_geojson(
    gdf: gpd.GeoDataFrame,
    region: str,
    year: str | int,
    category: str,
    tmp: bool = False,
) -> None:
    """Save the GeoDataFrame as a GeoJSON file."""
    if tmp:
        path_output = os.path.join(GEOJSON_TMP_DIR, region)
    else:
        path_output = os.path.join(GEOJSON_DIR, region)
    filename_output = f"{year}_{category}.geojson"
    gdf.to_file(os.path.join(path_output, filename_output), driver="GeoJSON")


def main(region: str) -> None:
    """Main function to process the parquet files and convert them to GeoJSON."""

    tqdm.write(f"Processing region: {region}")
    filenames = parquet_filenames(region)
    raps_grouped = group_raps_into_categories_from_filenames(filenames)
    for category, year, filenames in process_raps_grouped(raps_grouped):
        # tqdm.write(f"Category: {category}, Year: {year}, Filenames: {filenames}")
        tqdm.write(f"Category: {category}, Year: {year}")

        merged_filenames = merge_filenames_per_year(region, filenames, year)
        gdf = restructure_merged_filenames(merged_filenames)
        save_geojson(gdf, region, year, category, tmp=True)


if __name__ == "__main__":
    tqdm.write("ep.cli.parquet2geojson")

    from ep.config import regions_alla

    for region in regions_alla:
        main(region)
