import os
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from ep.config import regions, raps_categories, default_years
from ep.cli.parquet2kommun import parquet_filenames, load_parquet
from ep.cli.sql2parquet import as_parquet


def print_merged_files(merged_files):
    for category, files in merged_files.items():
        print(f"\n{category}:")
        for file in files:
            print(file)


def merge_files_with_raps():
    result = {}
    for category, category_raps in raps_categories.items():
        result[category] = []

        for region in ["06", "07"]:
            filenames = parquet_filenames(region)

            for raps in category_raps:
                raps = raps.replace(" ", "_")
                for filename in filenames:
                    if raps in filename:
                        result[category].append(os.path.join(region, filename))
    return result


def drop_duplicates(df):
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


def main():
    merged_files = merge_files_with_raps()
    print_merged_files(merged_files)

    for category, year, df in collect_files(merged_files):
        tqdm.write(f"{category} {year}")
        df = drop_duplicates(df)
        as_parquet(df, region="alla", table=f"{year}_{category}")


if __name__ == "__main__":
    main()
