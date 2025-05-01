import os
import sys
import typer
import numpy as np
import matplotlib.pyplot as plt
from eptools.processing.dataframe import (
    parquet_filenames,
    load_parquet,
    year_from_filename,
    max_index,
    as_pickle,
)
from eptools.processing.transform import (
    aggregated_loadprofile,
    kommuner_in_region,
    get_kommuner,
)
from eptools.utils.paths import PARQUET_DIR

"""
Write a docstring of what this script does.
"""


app = typer.Typer()


def kommun_loadprofile(filenames, kommuner, region) -> dict:
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


@app.command()
def run(region: str):
    print("kommun")
    print(f"Running pipeline for region {region}")

    filenames = parquet_filenames(region)
    kommuner = kommuner_in_region(filenames, region)
    lp_kommuner: dict = kommun_loadprofile(filenames, kommuner, region)
    lp_max_time: dict = kommuner_max_time(filenames, kommuner, region, lp_kommuner)

    as_pickle(lp_kommuner, f"{region}_lp_kommuner", region)
    as_pickle(lp_max_time, f"{region}_max_time", region)
