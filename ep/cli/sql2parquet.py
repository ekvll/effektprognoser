import os
import pandas as pd
import geopandas as gpd
import numpy as np
import sqlite3

from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ep.config import SQL_DIR, PARQUET_DIR, BG_DIR

"""
This script processes SQLite database tables and converts them into Parquet files.

It performs the following steps:
1. Connects to the SQLite database.
2. Retrieves the list of tables and years from the database.
3. Filters tables by year.
4. Loads data from the tables in chunks.
5. Cleans and formats the data.
6. Performs quality checks on the data.
7. Calculates energy statistics.
8. Adds geometry to the data.
9. Performs polygon intersection with kommuner and natomrade.
10. Saves the processed data as Parquet files.

The script is designed to work with a specific database structure and assumes the presence of certain columns in the tables.
"""


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


def filter_tables(tables: list[str], year: str) -> list[str]:
    """Filter tables by year."""
    return [table for table in tables if year in table]


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
    gdf = validate_geometries(gdf)
    intersect = validate_geometries(intersect)

    return gpd.overlay(gdf, intersect, how="intersection")


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop invalid geometries from a GeoDataFrame."""
    if not gdf.is_valid.all():
        gdf = gdf[gdf.is_valid].reset_index(drop=True)
        tqdm.write("Dropped invalid geometries.")
    return gdf


def get_largest_area_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
            largest = get_largest_area_geometry(gdf_uid)
            results.append(largest)

        else:
            results.append(gdf_uid)

    gdf_out = pd.concat(results, ignore_index=True)
    if "area" in gdf_out.columns:
        gdf_out = drop_column(gdf_out, "area")

    return gdf_out


def aggregate_loadprofile(
    gdf: pd.DataFrame | gpd.GeoDataFrame, year: str
) -> np.ndarray:
    """Aggregate the load profile for a specific kommun for a given year."""
    expected_length = 8784 if year == "2040" else 8760

    aggregated_lp = np.zeros(expected_length)

    for i, lp in enumerate(gdf["lp"]):
        lp_array = np.asarray(lp)
        if lp_array.shape[0] != expected_length:
            raise ValueError(
                f"Load profile at index {i} has incorrect length: "
                f"{lp_array.shape[0]}, expected: {expected_length}"
            )
        aggregated_lp += lp_array

    return aggregated_lp


def initialize_total_loadprofile() -> dict[str, np.ndarray]:
    """Initialize the total load profile dictionary."""
    return {
        "2022": np.zeros(8760),
        "2027": np.zeros(8760),
        "2030": np.zeros(8760),
        "2040": np.zeros(8784),
    }


def process_file_for_kommun(
    filename: str, kommun: str, region: str
) -> tuple[str, np.ndarray] | None:
    """Process a single parquet file for one kommun.
    Retuns (year, aggregated_profile) or None."""
    gdf = load_parquet(filename, region)
    gdf_kommun = gdf.loc[gdf["kn"] == kommun]

    if gdf_kommun.empty:
        tqdm.write(f"DataFrame for kommun {kommun} is empty. Skipping.")
        return None

    year = filename.split("_")[1]
    profile = aggregate_loadprofile(gdf_kommun, year)

    return year, profile


def calc_total_loadprofile_per_kommun(
    filenames: list[str], kommuner, region: str
) -> dict:
    """Calculate the total load profile per kommun."""
    total_loadprofile = {}

    for kommun in kommuner:
        total_loadprofile[kommun] = initialize_total_loadprofile()

        for filename in filenames:
            result = process_file_for_kommun(filename, kommun, region)
            if result is None:
                continue

            year, profile = result
            total_loadprofile[kommun][year] += profile

    return total_loadprofile


def get_column_names(cursor: sqlite3.Cursor, table: str) -> list[str]:
    """Get the column names of a table in a SQLite database."""
    sql_query = f"PRAGMA table_info({table});"
    cursor.execute(sql_query)
    columns = [c[1] for c in cursor.fetchall()]
    return columns


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the DataFrame."""
    df = df.rename(columns={"rut_id": "rid"})
    df = df.sort_values(by=["rid", "Tidpunkt"], ascending=[True, True])
    return df.reset_index(drop=True)


def build_select_query(table: str, columns: list[str]) -> str:
    """Build a SQL SELECT query for a table."""
    columns_str = ", ".join(columns)
    return f"SELECT {columns_str} FROM {table};"


def load_table_chunks(
    conn: sqlite3.Connection,
    cursor: sqlite3.Cursor,
    table: str,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Load a table from a SQLite database in chunks."""
    columns = get_column_names(cursor, table)
    sql_query = build_select_query(table, columns)

    # Load the data in chunks
    chunks = pd.read_sql_query(sql_query, conn, chunksize=chunk_size)
    df = pd.concat(chunks, ignore_index=True)

    return format_df(df)


def get_kommuner_in_region(filenames: list[str], region: str) -> list[str]:
    """Get the list of unique kommuner in the region from the parquet files."""

    kommun_set = set()
    for filename in filenames:
        gdf = load_parquet(filename, region)

        for kn, kk in zip(gdf["kn"], gdf["kk"]):
            kommun_set.add((kn, kk))

    unique_kommuner: list[dict] = [
        {"kommunnamn": kn, "kommunkod": kk} for kn, kk in kommun_set
    ]

    result = {"kommunnamn": [], "kommunkod": []}

    for d in unique_kommuner:
        result["kommunnamn"].append(d["kommunnamn"])
        result["kommunkod"].append(d["kommunkod"])

    x = region  # or x = 10

    filtered_kommunnamn = []
    filtered_kommunkod = []

    for kn, kk in zip(result["kommunnamn"], result["kommunkod"]):
        # Convert kommunkod to string and check first two digits
        if str(kk).startswith(str(x)):
            filtered_kommunnamn.append(kn)
            filtered_kommunkod.append(kk)

    filtered_result = {
        "kommunnamn": filtered_kommunnamn,
        "kommunkod": filtered_kommunkod,
    }

    return sort_dictionary(filtered_result, "kommunnamn", "kommunkod")


def sort_dictionary(d: dict, c1: str, c2: str) -> dict:
    """Sort a dictionary by the first column and return a new dictionary."""
    # Zip the lists together into tuples
    combined = list(zip(d[c1], d[c2]))

    # Sort by c1 (which is the firt element in each tuple)
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Unzip back to two lists
    c1_sorted, c2_sorted = zip(*combined_sorted)

    # Convert back to the dict format
    sorted_dict = {c1: list(c1_sorted), c2: list(c2_sorted)}

    return sorted_dict


def connect_to_db(path: str) -> sqlite3.Connection:
    """Create a connection to the SQLite database."""
    try:
        return sqlite3.connect(path)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to database: {e}")


def get_cursor(conn: sqlite3.Connection) -> sqlite3.Cursor:
    """Get a cursor object from the database connection."""
    return conn.cursor()


def validate_connection(conn: sqlite3.Connection) -> None:
    """Validate the database connection by executing a simple query."""
    try:
        conn.execute("SELECT 1")
    except sqlite3.OperationalError as e:
        conn.close()
        raise sqlite3.OperationalError(f"Database connection failed: {e}")


def db_connect(path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create a connection to the SQLite database and return the connection and cursor."""
    conn = connect_to_db(path)
    cursor = get_cursor(conn)
    validate_connection(conn)
    return conn, cursor


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


def load_grid(filename: str = "RSS_Skane_squares.gpkg") -> gpd.GeoDataFrame:
    """
    Load the grid data from a GPKG file.

    Args:
        filename (str): The name of the GPKG file containing the grid data. Defaults to "RSS_Skane_squares.gpkg".

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the grid data with columns 'rid' and 'geometry'.

    Example:
        >>> grid = load_grid()
        >>> grid.columns
        Index(['rid', 'geometry'], dtype='object')
    """
    path = os.path.join(BG_DIR, filename)

    cols = ["rut_id", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(columns={"rut_id": "rid"}, inplace=True)
    gdf = gdf.astype({"rid": np.uint64})

    gdf = gdf.dissolve(by="rid")
    gdf = gdf.reset_index()

    tqdm.write(f"Loaded GeoDataFrame grid ({filename}).")
    return gdf


def load_kommun(filename: str = "RSS_Skane_kommuner.gpkg") -> gpd.GeoDataFrame:
    """Load the kommun data."""
    path = os.path.join(BG_DIR, filename)

    # cols = ["KOMMUNNAMN", "KOMMUNKOD", "LANSKOD", "LANSNAMN", "geometry"]
    cols = ["KOMMUNNAMN", "KOMMUNKOD", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(
        columns={
            "KOMMUNKOD": "kk",
            "KOMMUNNAMN": "kn",
            # "LANSNAMN": "lan",
            # "LANSKOD": "lan_id",
        },
        inplace=True,
    )
    gdf = gdf.dissolve(by="kn")
    gdf = gdf.reset_index()

    tqdm.write(f"Loaded GeoDataFrame kommuner ({filename}).")
    return gdf


def load_natomrade(filename: str = "natomraden.gpkg") -> gpd.GeoDataFrame:
    """Load the natomrade data."""
    path = os.path.join(BG_DIR, filename)

    cols = ["company", "geometry"]
    crs = "EPSG:3006"

    gdf = load_gpkg(path, cols, crs)

    # Format the columns
    gdf.rename(columns={"company": "natbolag"}, inplace=True)
    gdf = gdf.dissolve(by="natbolag")
    gdf = gdf.reset_index()

    tqdm.write(f"Loaded GeoDataFrame natomrade ({filename}).")
    return gdf


def as_parquet(
    gdf: gpd.GeoDataFrame | pd.DataFrame, region: str, table: str, subdir: str = None
) -> None:
    """
    Save a GeoDataFrame as a Parquet file.

    Args:
        gdf (gpd.GeoDataFrame | pd.DataFrame): The GeoDataFrame or DataFrame to save.
        region (str): The region identifier.
        table (str): The name of the table (used as the filename).
        subdir (str, optional): Subdirectory within the region directory. Defaults to None.

    Returns:
        None: The function saves the GeoDataFrame as a Parquet file.

    Example:
        >>> as_parquet(gdf, "13", "2022_table1")
        This will save the GeoDataFrame as '13/2022_table1.parquet' in the PARQUET_DIR.
    """
    dirpath = os.path.join(PARQUET_DIR, region)
    if subdir:
        dirpath = os.path.join(dirpath, subdir)
        os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, table + ".parquet")
    gdf.to_parquet(filepath)


def parquet_filenames(region: str) -> list[str]:
    """
    Get the list of all parquet filenames for a given region.

    Args:
        region (str): The region identifier.

    Returns:
        list[str]: A list of parquet filenames in the specified region directory.

    Example:
        >>> parquet_filenames("13")
        ['2022_table1.parquet', '2022_table2.parquet', ...]
    """
    region_path = Path(PARQUET_DIR) / region

    if not region_path.exists():
        raise FileNotFoundError(f"Region path {region_path} does not exist")

    filenames = [
        f.name for f in region_path.iterdir() if f.is_file() and f.suffix == ".parquet"
    ]

    if not filenames:
        raise FileNotFoundError(f"No parquet files found in {region_path}")

    return filenames


def load_parquet(
    filename: str, region: str, cols: Optional[list[str]] = None
) -> gpd.GeoDataFrame:
    """
    Load a parquet file into a GeoDataFrame.

    Args:
        filename (str): The name of the parquet file to load.
        region (str): The region identifier.
        cols (list[str], optional): List of columns to select from the GeoDataFrame. Defaults to None.

    Returns:
        gpd.GeoDataFrame: The loaded GeoDataFrame containing the specified columns.

    Example:
        >>> load_parquet("2022_table1.parquet", "13", cols=["rid", "geometry"])
        This will load the specified columns from '13/2022_table1.parquet'.
    """
    file_path = Path(PARQUET_DIR) / region / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist")

    df = gpd.read_parquet(file_path)
    return df[cols] if cols else df


def db_path(region: str) -> str:
    """Create the path to the SQLite database file for a given region."""
    dir_name = f"Effektmodell_{region}"
    db_name = dir_name + ".sqlite"
    db_path = os.path.join(SQL_DIR, dir_name, db_name)
    if os.path.isfile(db_path):
        return db_path
    raise FileNotFoundError(f"Database file {db_path} not found.")


def has_no_all_zero_columns(df: pd.DataFrame) -> bool:
    return not any((df[col] == 0.0).all() for col in df.columns)


def has_no_all_nan_columns(df: pd.DataFrame) -> bool:
    return not any(df[col].isna().all() for col in df.columns)


def has_complete_days(df: pd.DataFrame, year: int | str) -> bool:
    if isinstance(year, int):
        year = str(year)
    days_in_df = int(df.shape[0] / df.rid.nunique())
    expected_days = 8784 if year in ["2020", "2040"] else 8760
    return days_in_df == expected_days


def qc(df: pd.DataFrame, year: int | str) -> bool:
    """
    Perform quality checks on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        year (int | str): The year associated with the data.

    Returns:
        bool: True if all checks pass, False otherwise.

    This function performs the following checks:
        - No all-zero columns
        - No all-NaN columns
        - Complete days for the specified year
        - DataFrame is not empty

    Example:
        >>> df = pd.DataFrame({'rid': [1, 2], 'Elanvandning': [0.0, 1.0], 'Tidpunkt': [20220101, 20220102]})
        >>> qc(df, 2022)
        True
    """
    checks = {
        "no all-zero columns": has_no_all_zero_columns(df),
        "no all-NaN columns": has_no_all_nan_columns(df),
        "complete days for year": has_complete_days(df, year),
        "not empty": not df.empty,
    }

    failed = [name for name, result in checks.items() if not result]

    for name in failed:
        tqdm.write(f"Failed QC: {name}")

    return not failed


def verify_gdf(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    """Verify and set the CRS of a GeoDataFrame."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=crs)

    if not gdf.crs == crs:
        gdf = gdf.set_crs(crs)

    return gdf


def main(region):
    """Main function to process SQLite database tables."""

    tqdm.write(f"Processing region: {region}")

    grid = load_grid()
    kommuner = load_kommun()
    natomrade = load_natomrade()

    path = db_path(region)
    conn, cursor = db_connect(path)
    tables = db_tables(cursor)
    years = db_years(tables)

    for year in tqdm(years, desc="Years", position=0, leave=False):
        tables_filtered = filter_tables(tables, year)

        for table in tqdm(tables_filtered, desc="Tables", position=1, leave=False):
            df = load_table_chunks(conn, cursor, table)
            df = drop_nan_row(df, "rid")
            df = set_dtypes(df)
            df = sort_df(df, ["rid", "Tidpunkt"], [True, True])

            if not qc(df, year):
                tqdm.write(f"Quality check failed for table {table}. Skipping.")
                continue

            df = drop_column(df, "Tidpunkt")
            df = calculate_energy_statistics(df)
            df = add_geometry(df, grid)

            gdf = to_gdf(df, crs="EPSG:3006")

            gdf = polygon_intersection(gdf, kommuner)
            gdf = largest_area(gdf)
            gdf = polygon_intersection(gdf, natomrade)
            gdf = largest_area(gdf)

            as_parquet(gdf, region, table)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    tqdm.write("Starting sql2parquet script")
    from ep.config import regions

    # Run from region 12 and onwards
    # regions = ["12"]
    for region in regions[5:]:
        main(region)
