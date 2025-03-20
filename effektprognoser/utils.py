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
