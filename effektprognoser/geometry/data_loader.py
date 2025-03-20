import geopandas as gpd
import numpy as np
import os

from effektprognoser.paths import DATA_DIR


def _gen_file_path(file_name: str, sub_folder: str = None) -> str:
    if sub_folder:
        return os.path.join(DATA_DIR, sub_folder, file_name)
    return os.path.join(DATA_DIR, file_name)


def _preprocess_grid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        gdf_renamed = gdf.rename(columns={"rut_id": "rid"})
    except Exception as e:
        raise NameError(f"Error in renaming 'rut_id' to 'rid': {e}")

    try:
        if gdf_renamed.rid.dtype != np.uint64:
            gdf_renamed.rid = gdf_renamed.rid.astype(np.uint64)
    except Exception as e:
        raise ValueError(f"Error in setting 'rid' dtype to np.uint64: {e}")

    try:
        gdf_renamed.set_index("rid", inplace=True)
    except Exception as e:
        raise IndexError(f"Error in setting 'rid' as index: {e}")

    return gdf_renamed


def load_gpkg(file_path: str, crs: str, keep_col: list[str] = None) -> gpd.GeoDataFrame:
    """
    Loads GeoPackage data into a GeoPandas DataFrame

    Args:
        file_path (str): File path to where the file is located.
        crs (str): Coordinate Reference System. For SWEREF 99 TM crs would be "EPSG:3006".
        keep_col (list[str]): Optional. A list of column names of the columns to keep. Default is None and thus no filtering is applied.

    Example:
        file_path = "/path/to/gpkg/"
        crs = "EPSG:3006"
        keep_col = ["x", "y"]
        gdf = load_gpkg(file_path, crs, keep_col)

    Return:
        gpd.GeoDataFrame: A GeoPandas GeoDataFrame.
    """
    if not file_path.endswith(".gpkg"):
        file_path += ".gpkg"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    gdf: gpd.GeoDataFrame = gpd.read_file(file_path)
    gdf = gdf.explode(index_parts=True)

    if keep_col:
        return gdf[keep_col]

    return gdf


def load_grid() -> gpd.GeoDataFrame:
    sub_folder = "grid"
    file_name = "RSS_Skane_squares.gpkg"
    file_path = _gen_file_path(file_name, sub_folder)

    crs = "EPSG:3006"

    keep_col = ["rut_id", "geometry"]

    gdf: gpd.GeoDataFrame = load_gpkg(file_path, crs, keep_col)

    print(f"load_grid() - Grid shape {gdf.shape}")
    return _preprocess_grid(gdf)


def load_kommuner() -> gpd.GeoDataFrame:
    sub_folder = "gis"
    file_name = "RSS_Skane_kommuner.gpkg"
    file_path: str = _gen_file_path(file_name, sub_folder)

    crs = "EPSG:3006"

    keep_col = ["KOMMUNNAMN", "KOMMUNKOD", "geometry"]

    gdf = load_gpkg(file_path, crs, keep_col)

    # Preprocess dataframe
    gdf = gdf.rename(columns={"KOMMUNNAMN": "kn", "KOMMUNKOD": "kk"})
    gdf = gdf.dissolve(by="kn")
    gdf = gdf.reset_index()
    print(f"load_kommuner() - Grid shape {gdf.shape}")
    return gdf


def load_natomrade() -> gpd.GeoDataFrame:
    sub_folder = "gis"
    file_name = "natomraden.gpkg"
    file_path: str = _gen_file_path(file_name, sub_folder)

    crs = "EPSG:3006"

    keep_col = ["id", "company", "geometry"]

    gdf = load_gpkg(file_path, crs, keep_col)

    # Preprocess dataframe
    gdf = gdf.dissolve(by="company")
    gdf = gdf.reset_index()
    print(f"load_natomrade() - Grid shape {gdf.shape}")
    return gdf
