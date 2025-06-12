"""
This script did not really do what it was supposed to do.

It was intended to do what parquet2all_table.py does, but it was not implemented correctly.

This script will eventually be removed.
"""

import os

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from ep.cli.parquet2kommun import load_parquet, parquet_filenames
from ep.cli.sql2parquet import as_parquet
from ep.config import default_years, raps_categories, regions


def print_merged_files(merged_files: dict) -> None:
    """docstring"""
    for category, files in merged_files.items():
        print(f"\n{category}:")
        for file in files:
            print(file)


def merge_files_with_raps() -> dict[str : list[str]]:
    """
    Merge parquet files based on predefined categories and regions.
    This function groups parquet files into categories defined in `raps_categories`
    and associates them with the respective regions.

    Args:
        None

    Returns:
        dict: A dictionary where keys are categories (e.g., 'bostader', 'transport', etc.)
              and values are lists of file paths for each category.

    Example:
          {
              "bostader": ["region1/file1.parquet", "region2/file2.parquet", ...],
              "transport": ["region1/file3.parquet", ...],
              ...
          }
    """
    result = {}

    # Iterate over each category (bostader, transport, etc.) and all raps associated with it (bostader: ["Smahus", "Flerbostadshus", ...], etc.)
    for category, category_raps in raps_categories.items():
        # Check if the category already exists in the result dictionary
        result[category] = []

        # Iterate over each region and all parquet filenames associated with it
        for region in regions:
            filenames = parquet_filenames(region)

            # Iterate over each raps in the category and check if it is present in the filenames
            for raps in category_raps:
                # Replace spaces with underscores to match the filename format
                raps = raps.replace(" ", "_")

                # Check if the raps is present in the filenames
                for filename in filenames:
                    if raps in filename:
                        result[category].append(os.path.join(region, filename))
    return result


def drop_duplicates(
    df: pd.DataFrame | gpd.GeoDataFrame,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Drop duplicates in a DataFrame or GeoDataFrame based on the 'rid' column.

    Args:
        df (pd.DataFrame | gpd.GeoDataFrame): The DataFrame or GeoDataFrame to process.

    Returns:
        pd.DataFrame | gpd.GeoDataFrame: A DataFrame or GeoDataFrame with duplicates dropped, keeping the first occurrence.
    """
    return df.drop_duplicates(subset="rid", keep="first")  # Or 'last' or False


def drop_duplicates_keep_highest(
    df: pd.DataFrame | gpd.GeoDataFrame, id_col: str, value_col: str
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    Drop duplicates in a DataFrame or GeoDataFrame, keeping the row with the highest value in a specified column.

    Args:
        df (pd.DataFrame | gpd.GeoDataFrame): The DataFrame or GeoDataFrame to process.
        id_col (str): The column name to group by (e.g., 'rid').
        value_col (str): The column name to determine the highest value (e.g., 'eb').
    Returns:
        pd.DataFrame | gpd.GeoDataFrame: A DataFrame or GeoDataFrame with duplicates dropped, keeping the row with the highest value in the specified column.
    """
    df_max_eb = df.loc[df.groupby(id_col)[value_col].idxmax()].reset_index(drop=True)
    return df_max_eb


def collect_files(merged_files):
    for category, subpaths in merged_files.items():
        for year in default_years:
            dfs = []
            subpaths_filtered = [s for s in subpaths if str(year) in s]
            for subpath in subpaths_filtered:
                parts = subpath.split(os.sep)
                region = parts[0]
                filename = parts[1]
                df = load_parquet(filename, region)
                dfs.append(df)
            yield category, year, pd.concat(dfs, ignore_index=True)


def collect_files_per_file(merged_files):
    # Iterate over each category (bostader, transport, etc.) and all parquet subpaths associated with it
    for category, subpaths in merged_files.items():
        # Iterate over each year and filter the subpaths for that year
        for year in default_years:
            dfs = []
            subpaths_filtered = [s for s in subpaths if str(year) in s]

            # Iterate over each subpath associated with the category and year
            for subpath in subpaths_filtered:
                parts = subpath.split(os.sep)
                region = parts[0]
                filename = parts[1]
                df = load_parquet(filename, region)
                dfs.append(df)

            yield category, year, pd.concat(dfs, ignore_index=True)


def main():
    merged_files = merge_files_with_raps()
    # print_merged_files(merged_files)

    for category, year, df in collect_files(merged_files):
        tqdm.write(f"{category} {year}")
        df = drop_duplicates(df)
        as_parquet(df, region="alla", table=f"{year}_{category}")


if __name__ == "__main__":
    main()
