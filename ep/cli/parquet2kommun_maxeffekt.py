"""
placeholder
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from ep.config import default_raps, EXCEL_DIR
from ep.cli.parquet2kommun import parquet_filenames, load_parquet


def max_index(arr):
    return np.max(arr), np.argmax(arr)


def aggregated_loadprofile(gdf, year):
    aggregated_lp = np.zeros(8784 if year == "2040" else 8760)
    for lp in gdf["lp"]:
        aggregated_lp += lp
    return aggregated_lp


def year_from_filename(filename: str) -> str:
    return filename.split("_")[1]


def kommuner_loadprofile(filenames, kommuner, region) -> dict:
    """Get a summed loadprofile for the whole kommun"""
    result = {}
    for kommun in kommuner:
        if kommun not in result:
            result[kommun] = {
                "2022": np.zeros(8760),
                "2027": np.zeros(8760),
                "2030": np.zeros(8760),
                "2040": np.zeros(8784),
            }
        for filename in filenames:
            year = year_from_filename(filename)
            gdf = load_parquet(filename, region)
            gdf_kommun = gdf.loc[gdf["kn"] == kommun]
            lp_kommun = aggregated_loadprofile(gdf_kommun, year)

            result[kommun][year] += lp_kommun
    return result


def kommuner_max_time(filenames, kommuner, region, lp_kommuner) -> dict:
    result = {}
    for kommun in kommuner:
        if kommun not in result:
            result[kommun] = {}
        for filename in filenames:
            year = year_from_filename(filename)
            gdf = load_parquet(filename, region)
            gdf_kommun = gdf.loc[gdf["kn"] == kommun].reset_index(drop=True)
            lp_kommun = aggregated_loadprofile(gdf_kommun, year)
            lp_max, lp_max_time = max_index(lp_kommun)
            result[kommun][filename] = [lp_max_time, lp_max, lp_kommun]
    return result


def get_kommuner_in_region(filenames: list[str], region: str) -> list[str]:
    result = []
    for filename in filenames:
        gdf = load_parquet(filename, region, cols=["kn"])
        kommuner = get_kommuner(gdf)
        result = result + kommuner
    return list(set(result))


def get_kommuner(df: pd.DataFrame | gpd.GeoDataFrame) -> list[str]:
    return list(df["kn"].unique())


def get_time_data(time_dict, kommun, year):
    filtered_times = {
        " ".join(k.split("_")[2:-1]): [int(v[0]), float(v[1]), v[2]]
        for k, v in time_dict[kommun].items()
        if year in k
    }
    return filtered_times


def get_max_and_index(loadprofile):
    max_value = np.max(loadprofile)
    index = np.argmax(loadprofile)
    return max_value, index


def process_lp(lp, time, region):
    for kommun, _ in lp.items():
        rows = {}
        df_year = pd.DataFrame()
        c1 = []
        c2 = []
        c3 = []

        for year, loadprofile in _.items():
            if year not in rows:
                rows[year] = []
            xrange = range(len(loadprofile))
            filtered_times = get_time_data(time, kommun, year)
            max_value, index = get_max_and_index(loadprofile)
            # rows.append({"År": year, "MaxTidpunkt": index, "MaxTidpunktEB": max_value})
            d1 = []
            d2 = []
            c1.append(max_value)
            c2.append(index)
            c3.append(year)

            for k, v in filtered_times.items():
                d1.append(k)
                d2.append(v[2][index])

            rows[year] = d1
            df_tmp = pd.DataFrame(d2, index=d1, columns=[year])

            if df_year.empty:
                df_year = df_tmp
            else:
                df_year = pd.merge(
                    df_year, df_tmp, left_index=True, right_index=True, how="outer"
                )

        df_year_max = pd.DataFrame({"Tidpunkt": c2, "Max": c1}, index=c3).T

        df_year = df_year.reindex(default_raps)
        df_year.replace(0, np.nan, inplace=True)

        save_to_excel(df_year_max, df_year, kommun, region)


def save_to_excel(df_year_max, df_year, kommun, region):
    out_path = os.path.join(EXCEL_DIR, region)
    with pd.ExcelWriter(
        os.path.join(out_path, f"{kommun} - Maxeffekt.xlsx"), engine="xlsxwriter"
    ) as writer:
        df_year_max.to_excel(writer, sheet_name="Combined")

        # Leave 2 empty rows between the tables for clarity
        start_row = len(df_year_max) + 3

        df_year.to_excel(writer, sheet_name="Combined", startrow=start_row)


def main(region: str) -> None:
    tqdm.write(f"Processing region {region}")
    filenames = parquet_filenames(region)
    kommuner = get_kommuner_in_region(filenames, region)

    lp_kommuner: dict = kommuner_loadprofile(filenames, kommuner, region)
    lp_max_time: dict = kommuner_max_time(filenames, kommuner, region, lp_kommuner)

    tqdm.write("Processing load profiles.")
    process_lp(lp_kommuner, lp_max_time, region)


if __name__ == "__main__":
    from ep.config import regions

    for region in regions:
        main(region)
