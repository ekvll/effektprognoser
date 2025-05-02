import os
import ast
import re
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from eptools.utils.paths import PARQUET_DIR, GEOJSON_TMP_DIR, GEOJSON_DIR
from eptools.utils.groups import category_groups, categories
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
        years: list[str] = get_years(files)

        for year in years:
            tqdm.write(f"Year {year}")

            # Only process files for this year
            files_filtered: list[str] = [f for f in files if year in f]

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

            # Re-structure the data dict to fit into a DataFrame
            records = []
            for rid, values in data.items():
                records.append(
                    {
                        "rid": rid,
                        "lp": values["lp"],
                        "eb": values["lp"].max(),
                        "ea": values["lp"].sum(),
                        "geometry": values["geometry"].values[0],
                    }
                )
            gdf_out: gpd.GeoDataFrame = gpd.GeoDataFrame(records, crs="EPSG:3006")

            # Save as GeoJSON
            path_output: str = os.path.join(GEOJSON_TMP_DIR, region)
            os.makedirs(path_output, exist_ok=True)
            filename_output: str = f"{year}_{category}.geojson"
            gdf_out.to_file(
                os.path.join(path_output, filename_output), driver="GeoJSON"
            )


def run(region: str):
    tqdm.write(f"Processing region {region}")
    path: str = os.path.join(PARQUET_DIR, region)
    files: list[str] = os.listdir(path)

    categories: dict = group_files(files)
    process(categories, region)


def percent_change(new_value, old_value):
    change = ((new_value - old_value) / old_value) * 100
    return change


def merge_keep_unmutuals(gdf, ref):
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


def merge_drop_unmutuals(gdf, ref):
    # Assume gdf and ref both have a 'rid' column and a 'value' column
    merged = gdf.merge(ref, on="rid", suffixes=("_gdf", "_ref"))

    merged["ebd"] = merged["eb_gdf"] - merged["eb_ref"]
    merged["ebp"] = ((merged["eb_gdf"] - merged["eb_ref"]) / merged["eb_ref"]) * 100

    merged["ead"] = merged["ea_gdf"] - merged["ea_ref"]
    merged["eap"] = ((merged["ea_gdf"] - merged["ea_ref"]) / merged["ea_ref"]) * 100

    # merged = merged.drop(
    #    columns=[col for col in merged.columns if "_gdf" in col or "_ref" in col]
    # )
    return merged


def postprocess(region: str):
    path = os.path.join(GEOJSON_TMP_DIR, region)
    out_path = os.path.join(GEOJSON_DIR, region)
    os.makedirs(out_path, exist_ok=True)
    files = os.listdir(path)
    for category in categories:
        print(f"\nCategory {category}")
        files_filtered = sorted([f for f in files if category in f])
        for file in files_filtered:
            gdf = gpd.read_file(os.path.join(path, file))
            year = file.split("_")[0]
            if year == "2022":
                ref = gdf.copy(deep=True)
            # Merge with the reference DataFrame
            merged = merge_keep_unmutuals(gdf, ref)
            print(merged.columns)
            print(merged.head())

            merged.to_file(
                os.path.join(out_path, f"{year}_{category}.geojson"), driver="GeoJSON"
            )

            # gdf = merge_drop_unmutuals(gdf, ref)
        #  unique_ids = gdf.rid.unique()

    # years: list[str] = get_years(files)
    # for file in files:
    #    print
    # gdf = gpd.read_file(os.path.join(path, file))
    # print(gdf)
    # gdf.plot()
    # plt.show()


if __name__ == "__main__":
    region = "10"
    # run(region)
    postprocess(region)
