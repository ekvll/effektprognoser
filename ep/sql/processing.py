import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd


def db_tables(cursor: sqlite3.Cursor) -> list[str]:
    """Get a list of all tables in the database."""
    sql_query = "SELECT name FROM sqlite_master;"
    cursor.execute(sql_query)

    # Put the table names in a list
    tables = [table[0] for table in cursor.fetchall()]
    return tables


def db_years(tables: list[str]) -> list[int]:
    """Get a list of all years from the file names."""
    years = []
    for table in tables:
        # Get the year from the file name
        year = table.split("_")[1]
        if year not in years:
            years.append(year)
    return sorted(years)


def drop_nan_row(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Drop rows with NaN values."""
    if col:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        return df.dropna(subset=[col])
    return df.dropna()


def get_column_dtypes() -> dict:
    """Define expected data types for each column."""
    return {
        "rid": np.uint64,
        "Elanvandning": np.float64,
        "Tidpunkt": np.uint64,
    }


def cast_dtypes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """Apply data types to DataFrame columns."""
    try:
        return df.astype(dtype_map)
    except Exception as e:
        raise ValueError(f"Error casting dtypes: {e}")


def set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Set predefined dtypes for a DataFrame."""
    dtype_map = get_column_dtypes()
    return cast_dtypes(df, dtype_map)


def sort_df(
    df: pd.DataFrame, col: list[str], ascending: bool | list[bool]
) -> pd.DataFrame:
    """Sort a DataFrame by one or more columns."""
    try:
        df_sorted = df.sort_values(by=col, ascending=ascending)
        return df_sorted.reset_index(drop=True)
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")
    except Exception as e:
        raise ValueError(f"Could not sort DataFrame: {e}")


def drop_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop a column from a DataFrame."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    return df.drop(columns=[col])


def group_elanvandning(df: pd.DataFrame) -> pd.DataFrame:
    """Group 'Elanvandning' by 'rid' into lists."""
    return df.groupby("rid")["Elanvandning"].apply(list).reset_index(name="lp")


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 'Elanvandning' (ea) and 'Effektbehov' (eb) statistics for each group."""
    df["ea"] = df["lp"].apply(lambda x: sum(x))
    df["eb"] = df["lp"].apply(lambda x: max(x))
    return df


def calculate_energy_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate energy data by rid."""
    transposed = group_elanvandning(df)
    return compute_summary_stats(transposed)


def add_geometry(
    df: pd.DataFrame, grid: gpd.GeoDataFrame
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Assign geometry to df rows based on matching 'rid' in the grid."""
    geometry_map = grid.set_index("rid")["geometry"]
    if not df["rid"].isin(geometry_map.index).all():
        missing = df.loc[~df["rid"].isin(geometry_map.index), "rid"].unique()
        raise ValueError(f"Missing geometry for rid(s): {missing}")

    df["geometry"] = df["rid"].map(geometry_map)
    return df


def to_gdf(df: pd.DataFrame, crs: str, geometry: str = "geometry") -> gpd.GeoDataFrame:
    """Convert DataFrame to GeoDataFrame."""
    if geometry not in df.columns:
        raise KeyError(f"Column '{geometry}' not found in DataFrame.")
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def polygon_intersection(
    gdf: gpd.GeoDataFrame, intersect: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Perform intersection of two GeoDataFrames."""
    gdf = _validate_geometries(gdf)
    intersect = _validate_geometries(intersect)

    return gpd.overlay(gdf, intersect, how="intersection")


def _validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop invalid geometries from a GeoDataFrame."""
    if not gdf.is_valid.all():
        gdf = gdf[gdf.is_valid].reset_index(drop=True)
        print("Dropped invalid geometries.")
    return gdf


def _get_largest_area_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.assign(area=gdf.geometry.area / 10**6).reset_index(drop=True)

    idx_max_area = gdf.area.idxmax()

    gdf_dissolve = gdf.dissolve()

    gdf_max = gdf.iloc[[idx_max_area]]
    gdf_max = gdf_max.assign(geometry=gdf_dissolve.iloc[0]["geometry"])

    return gdf_max


def largest_area(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Get the largest area geometry from a GeoDataFrame."""
    results = []

    for uid in gdf.rid.unique():
        gdf_uid = to_gdf(gdf[gdf.rid == uid], crs="EPSG:3006")

        if len(gdf_uid) > 1:
            largest = _get_largest_area_geometry(gdf_uid)
            results.append(largest)

        else:
            results.append(gdf_uid)

    gdf_out = pd.concat(results, ignore_index=True)
    if "area" in gdf_out.columns:
        gdf_out = drop_column(gdf_out, "area")

    return gdf_out
