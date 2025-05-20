import os
import geopandas as gpd
import numpy as np

from ep.config import BG_DIR


def set_crs(gdf: gpd.GeoDataFrame, crs: str = None) -> gpd.GeoDataFrame:
    """Set the coordinate reference system (CRS) for a GeoDataFrame."""
    if crs:
        if gdf.crs is None:
            gdf.set_crs(crs, allow_override=True, inplace=True)
        if gdf.crs != crs:
            gdf.to_crs(crs, inplace=True)

    return gdf


def load_gpkg(path: str, cols: list[str] = None, crs: str = None) -> gpd.GeoDataFrame:
    """Load the GPKG file."""
    gdf = gpd.read_file(path)
    gdf = set_crs(gdf, crs)

    if cols:
        return gdf[cols]
    return gdf


def load_grid() -> gpd.GeoDataFrame:
    """Load the grid data."""
    path = os.path.join(BG_DIR, "RSS_Skane_squares.gpkg")

    cols = ["rut_id", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(columns={"rut_id": "rid"}, inplace=True)
    gdf = gdf.astype({"rid": np.uint64})

    gdf = gdf.dissolve(by="rid")
    gdf = gdf.reset_index()

    return gdf


def load_kommun() -> gpd.GeoDataFrame:
    """Load the kommun data."""
    path = os.path.join(BG_DIR, "RSS_Skane_kommuner.gpkg")

    cols = ["KOMMUNNAMN", "KOMMUNKOD", "LANSKOD", "LANSNAMN", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(
        columns={
            "KOMMUNKOD": "kk",
            "KOMMUNNAMN": "kn",
            "LANSNAMN": "lan",
            "LANSKOD": "lan_id",
        },
        inplace=True,
    )
    gdf = gdf.dissolve(by="kn")
    gdf = gdf.reset_index()

    return gdf


def load_natomrade() -> gpd.GeoDataFrame:
    """Load the natomrade data."""
    path = os.path.join(BG_DIR, "natomraden.gpkg")

    cols = ["company", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(columns={"company": "natbolag"}, inplace=True)
    gdf = gdf.dissolve(by="natbolag")
    gdf = gdf.reset_index()

    return gdf


if __name__ == "__main__":
    load_grid()
    load_kommun()
    load_natomrade()
