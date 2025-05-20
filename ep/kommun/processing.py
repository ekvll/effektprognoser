import os
import pandas as pd
import geopandas as gpd

from ep.config import PARQUET_DIR
from ep.files import load_parquet


def kommuner_in_region(filenames: list[str], region: str) -> list[str]:
    """Get the list of unique kommuner in the region from the parquet files."""
    kommuner = []
    for filename in filenames:
        gdf = load_parquet(filename, region, cols=["kn"])
        gdf_kommuner = list(gdf["kn"].unique())
        kommuner += kommuner + gdf_kommuner
    return list(set(kommuner))


def kommun_loadprofile(filenames, kommuner, region):
    pass


def kommuner_max_time(filenames, kommuner, region, lp_kommuner):
    pass
