import pandas as pd
import geopandas as gpd
import os
import numpy as np

from effektprognoser.paths import DATA_DIR


def add_geometry(df, gdf_grid):
    square_mapping = gdf_grid["geometry"].to_dict()
    geometries = [square_mapping.get(row.rid, None) for _, row in df.iterrows()]
    df = df.assign(geometry=geometries)
    return df


def convert_df_to_gdf(df, geometry: str = "geometry", crs: str = "EPSG:3006"):
    if isinstance(df, gpd.GeoDataFrame):
        if not df.crs:
            df = df.set_crs(crs)

        if df.crs != crs:
            df = df.to_crs(crs)

        if geometry not in df.columns:
            raise NameError(f"Column '{geometry}' not found in DataFrame")

        return df

    return gpd.GeoDataFrame(df, geometry=df[geometry], crs=crs)


def intersection_by_polygon(gdf, gdf_intersect) -> gpd.GeoDataFrame:
    if not gdf.is_valid.all():
        print(
            "intersection_by_polygon: invalid geometries detected in gdf. Attempting to fix."
        )
        gdf = gdf[gdf.is_valid].reset_index(drop=True)

    if not gdf_intersect.is_valid.all():
        print(
            "intersection_by_polygon: invalid geometries in gdf_intersect. Attempting to fix."
        )
        gdf_intersect = gdf_intersect[gdf_intersect.is_valid].reset_index(drop=True)

    if gdf.empty:
        print("intersection_by_polygon: gdf is empty")
    if gdf_intersect.empty:
        print("intersection_by_polygon: gdf_intersect is empty")

    gdf_copy = gdf.copy(deep=True)
    gdf_copy = gpd.overlay(gdf_copy, gdf_intersect, how="intersection")
    return gdf_copy


def keep_largest_area(gdf: gpd.GeoDataFrame):
    def _get_largest_area_geometry(gdf):
        gdf_uid = gdf.assign(area=gdf["geometry"].area / 10**6).reset_index(drop=True)
        idx_max_area = gdf_uid.area.idxmax()
        gdf_dissolve = gdf_uid.dissolve()
        gdf_max = gdf_uid.iloc[[idx_max_area]]
        gdf_max = gdf_max.assign(geometry=gdf_dissolve.iloc[0]["geometry"])
        return gdf_max

    gdf_out = gpd.GeoDataFrame()

    for uid in gdf.rid.unique():
        gdf_uid = gpd.GeoDataFrame(
            data=gdf[gdf.rid == uid], geometry="geometry", crs="EPSG:3006"
        )

        if gdf_uid.shape[0] > 1:
            gdf_max = _get_largest_area_geometry(gdf_uid)
            gdf_out = pd.concat([gdf_out, gdf_max])

        else:
            gdf_out = pd.concat([gdf_out, gdf_uid])

    if "area" in gdf_out.columns:
        gdf_out.drop(columns=["area"], axis=0, inplace=True)
    return gdf_out


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


if __name__ == "__main__":
    pass
