import os
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from eptools.utils.paths import PARQUET_DIR
from eptools.utils.groups import category_groups
from eptools.processing.transform import get_category

"""
This script extracts the category from each parquet file. Then groups corresponding categories into groups: bostader, transport, total, and so on.

Thereafter this script performs processing on each group and saves the output as GeoJSON, ready for the webmap.
"""


def group_files(files: list[str]) -> dict:
    categories = {}
    for key, values in category_groups.items():
        if key not in categories:
            categories[key] = []
        for file in files:
            category = get_category(file)
            if category in values:
                categories[key].append(file)
    for key, values in categories.items():
        print("\n", key)
        print(values)
    return categories


def get_years(files: list[str]) -> list[str]:
    """
    Extract and sort unique years from filenames.
    """
    years = []
    for file in files:
        # Assume year is in the second position of the filename
        year = file.split("_")[1]

        if year not in years:
            years.append(year)

    years_sorted = sorted(years)
    tqdm.write(f"Found years {years_sorted}")

    return years_sorted


def process(categories: dict, region: str):
    for category, files in categories.items():
        tqdm.write(f"\nCategory {category}")
        if len(files) == 0:
            tqdm.write(f"Category {category} contain no files. Skipping...")
            continue

        # Extract years from current category's files
        years = get_years(files)

        for year in years:
            tqdm.write(f"Year {year}")

            # Only process files for this year
            files_filtered = [f for f in files if year in f]

            if len(files_filtered) == 0:
                tqdm.write(f"No files for year {year}")
                continue

            # Will store accumulated values per 'rid'
            data = {}

            for file in files_filtered:
                # print(file)
                path = os.path.join(PARQUET_DIR, region, file)
                gdf = gpd.read_parquet(path)

                if gdf.empty:
                    tqdm.write(f"DataFrame is empty {path}. Skipping...")
                    continue

                # Unique RutIDs (rid) in thus file
                unique_ids = gdf.rid.unique()

                for rid in unique_ids:
                    gdf_rid = gdf.loc[gdf.rid == rid]

                    # Initialize array of zeros for each 'rid' if not already done
                    if rid not in data:
                        data[rid] = {
                            "lp": np.zeros(8784 if year == "2040" else 8760),
                            "geometry": gdf_rid.geometry,
                        }

                    if gdf_rid.shape[0] != 1:
                        # Each 'rid' is expected to have one row
                        raise ValueError

                    # Add the 'lp' array to cumulative sum for this 'rid'
                    data[rid]["lp"] += gdf_rid.lp.to_numpy()[0]

            records = []
            for rid, values in data.items():
                records.append(
                    {
                        "rid": rid,
                        "lp": values["lp"],
                        "geometry": values["geometry"].values[0],
                    }
                )
            gdf_out = gpd.GeoDataFrame(records, crs="EPSG:3006")
            print(gdf_out.head())


def run(region):
    tqdm.write(f"Processing region {region}")
    path: str = os.path.join(PARQUET_DIR, region)
    files: list[str] = os.listdir(path)

    categories: dict = group_files(files)
    process(categories, region)


if __name__ == "__main__":
    region = "10"
    run(region)
