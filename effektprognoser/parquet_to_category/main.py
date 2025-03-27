import os
import numpy as np
import pandas as pd
import geopandas as gpd
from effektprognoser.paths import PARQUET_DIR_LOCAL, DATA_DIR
from effektprognoser.utils import get_metadata, get_category_mapping

try:
    from ..sql_manager import get_years_in_table_names
except Exception:
    from effektprognoser.sql_manager import get_years_in_table_names


def get_parquet_category_match(files):
    print("get_parquet_category_match")
    result = {}
    category_mapping = get_category_mapping()
    for category, raps_list in category_mapping.items():
        if category not in result:
            result[category] = []
        # print()
        # print(f"Category: {category}")
        # print(f"Merging RAPS: {raps_list}")

        # print("Matching Parquet files:")
        for raps in raps_list:
            for file in files:
                if raps in file:
                    # print(file)
                    result[category].append(file)
    return result


def get_year_match(matches, years):
    print("get_year_match")
    result = {}
    for key, files in matches.items():
        if key not in result:
            result[key] = {}
        for year in years:
            if year not in result[key]:
                result[key][year] = []
            for file in files:
                if year in file:
                    result[key][year].append(file)

    return result


def group_data(to_group: dict, input_path):
    print("group_data")
    result = {}
    for category, years in to_group.items():
        # print(f"\nCategory: {category}")
        if category not in result:
            result[category] = {}
        for year, files in years.items():
            # print(f"\nYear: {year}")
            if year not in result[category]:
                result[category][year] = {}
            # print("Files:")
            for file in files:
                # print(file)

                filepath = os.path.join(input_path, file)
                parquet = gpd.read_parquet(filepath)
                # print(parquet.head())
                for rid in parquet.rid.unique():
                    if rid not in result[category][year]:
                        result[category][year][rid] = []
                    parquet_rid = parquet.loc[parquet.rid == rid]
                    if parquet_rid.shape[0] != 1:
                        raise ValueError("parquet_rid as more than 1 row")
                    aggregated_lp = np.zeros(8784 if year == "2040" else 8760)
                    for lastprofil in parquet_rid.lp:
                        aggregated_lp += lastprofil
                    result[category][year][rid].append(
                        [aggregated_lp, parquet_rid.geometry.values]
                    )

    return result


def save_aggregated_data(data: dict, region: str) -> None:
    print("save_aggregated_data")
    for category, years in data.items():
        # print(f"Category: {category}")
        for year, rids in years.items():
            # print(f"Year: {year}")
            output_filename = f"{year}_{category}.parquet"
            rutid, lp, geometry = [], [], []
            for rid, data in rids.items():
                rutid.append(rid)
                data = data[0]
                lp.append(data[0])
                geometry.append(data[1][0])
                # print(f"RutID: {rid}")
                # print(f"Lastprofil: {lp}")
                # print(f"Geometry: {geometry[0]}")
            output_path = os.path.join(
                DATA_DIR, "parquet_to_category", "preprocess", region
            )
            os.makedirs(output_path, exist_ok=True)
            output_filepath = os.path.join(output_path, output_filename)
            gdf = gpd.GeoDataFrame(
                {"rid": rutid, "lastprofil": lp, "geometry": geometry}
            )
            gdf.to_parquet(output_filepath)


def main(region) -> None:
    # Get Parquet files associated with region
    input_path = os.path.join(PARQUET_DIR_LOCAL, region)
    files = os.listdir(input_path)

    # Get years covered by files
    years = get_years_in_table_names(files)

    matches = get_parquet_category_match(files)

    to_group = get_year_match(matches, years)

    aggregated_data = group_data(to_group, input_path)

    save_aggregated_data(aggregated_data, region)


if __name__ == "__main__":
    main("06")
