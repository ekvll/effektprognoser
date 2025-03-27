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


def get_category_mapping():
    category_mapping = {
        "bostader": ["Smahus_V1", "Flerbostadshus_V1"],
        "transport": [
            "LL_V1",
            "PB_V1",
            "TT_V1",
            "TT_DEP_V1",
            "TT_RESTSTOP_V1",
            "TT_DEST_V1",
        ],
        "offentlig_och_privat_sektor": ["7777_V1", "8888_V1"],
        "industri_och_bygg": [
            "_4_V1",
            "_5_V1",
            "_6_V1",
            "_7_V1",
            "_8_V1",
            "_9_V1",
            "_10_V1",
            "_11_V1",
            "_13_V1",
            "_14_V1",
            "_15_V1",
            "_16_V1",
            "_17_V1",
            "_18_V1",
            "_19_V1",
            "_20_V1",
            "_21_V1",
            "_22_V1",
            "_23_V1",
            "_24_V1",
            "_27_V1",
        ],
        "jordbruk_skog_och_fiske": ["_1_3_V1"],
    }
    return category_mapping
