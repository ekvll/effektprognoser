import geopandas as gpd
import numpy as np
import pandas as pd


def sort_dict(data):
    return dict(sorted(data.items(), key=lambda item: item[1].max(), reverse=True))


def make_np_array(obj: list | pd.Series | np.ndarray) -> np.ndarray | None:
    """
    Tries to convert an object into a numpy array.

    Args:
        obj (list | pd.Series | np.ndarray): Object to convert.

    Returns:
        np.ndarray | None: Numpy array if conversion is possible, None otherwise.
    """
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return None


def substring_search(main_string: str, string_list: list[str]) -> bool:
    """
    Sub-string search.

    Args:
        main_string (str): The string to look-up.
        string_list (list[str]): List of strings of strings to check for.

    Returns:
        bool: True if main_string contain any string in string_list, False otherwise.
    """
    return any(substring in main_string for substring in string_list)


def verify_gdf(gdf):
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:3006")

    if not gdf.crs == "EPSG:3006":
        gdf = gdf.set_crs("EPSG:3006")

    return gdf


def get_metadata():
    raps = [
        "Flerbostadshus",
        "Smahus",
        "LOM_Flerbostadshus",
        "LOM_Smahus",
        "RAPS_1_3",
        "RAPS_4",
        "RAPS_5",
        "RAPS_6",
        "RAPS_7",
        "RAPS_8",
        "RAPS_9",
        "RAPS_10",
        "RAPS_11",
        "RAPS_12",
        "RAPS_13",
        "RAPS_14",
        "RAPS_15",
        "RAPS_16",
        "RAPS_17",
        "RAPS_18",
        "RAPS_19",
        "RAPS_20",
        "RAPS_21",
        "RAPS_22",
        "RAPS_23",
        "RAPS_24",
        "RAPS_27",
        "RAPS_7777",
        "RAPS_8888",
        "PB",
        "LL",
        "TT_DEP",
        "TT_DEST",
        "TT_RESTSTOP",
    ]
    raps = [r + "_" for r in raps]
    years = ["2022", "2027", "2030", "2040"]
    return raps, years


def combine_metadata(raps_list, years):
    checks = []
    for raps in raps_list:
        for year in years:
            checks.append(f"{year}_{raps}")
    return checks
