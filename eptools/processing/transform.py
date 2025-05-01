import pandas as pd
import geopandas as gpd
import numpy as np
from .dataframe import load_parquet


def get_category(filename: str) -> str:
    """
    Extract the category from filename. For example, for filename 'EF_2022_RAPS_16_V1.parquet' the corresponding category is 'RAPS_16".

    Args:
        filename (str): Filename to extract category from.

    Returns:
        str: The category extracted from filename.
    """
    category = " ".join(filename.split("_")[2:-1])
    return category


def aggregated_loadprofile(gdf, year):
    aggregated_lp = np.zeros(8784 if year == "2040" else 8760)
    for lp in gdf["lp"]:
        aggregated_lp += lp
    return aggregated_lp


def kommuner_in_region(filenames: list[str], region: str) -> list[str]:
    kommuner = []
    for filename in filenames:
        gdf = load_parquet(filename, region, cols=["kn"])
        kommuner_file = get_kommuner(gdf)
        kommuner = kommuner + kommuner_file
    return list(set(kommuner))


def get_kommuner(df: pd.DataFrame | gpd.GeoDataFrame):
    return list(df["kn"].unique())
