import pandas as pd
import geopandas as gpd

from tqdm import tqdm

from ep.cli.sql2parquet import (
    db_connect,
    db_path,
    db_tables,
    db_years,
    filter_tables,
    format_df,
    drop_nan_row,
    set_dtypes,
    sort_df,
    qc,
    drop_column,
    calculate_energy_statistics,
    add_geometry,
    to_gdf,
    polygon_intersection,
    largest_area,
    load_grid,
    load_kommun,
    load_natomrade,
    as_parquet,
)
from ep.cli.parquet2kommun import get_expected_length


def process_chunk(
    chunk: pd.DataFrame,
    year: int | str,
    grid: gpd.GeoDataFrame,
    kommuner: gpd.GeoDataFrame,
    natomrade: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Process a single chunk of data from the database.

    This function performs the following operations:
    1. Formats the DataFrame.
    2. Drops rows with NaN values in the 'rid' column.
    3. Sets the appropriate data types for the DataFrame.
    4. Sorts the DataFrame by 'rid' and 'Tidpunkt'.
    5. Performs quality control checks.
    6. Drops the 'Tidpunkt' column.
    7. Calculates energy statistics.
    8. Adds geometry to the DataFrame.
    9. Converts the DataFrame to a GeoDataFrame.
    10. Performs polygon intersection with kommuner and natomrade.
    11. Returns the largest area from the intersection.

    Args:
        chunk (pd.DataFrame): The chunk of data to process.
        year (int): The year for which the data is being processed.
        grid (gpd.GeoDataFrame): Grid data for geometry processing.
        kommuner (gpd.GeoDataFrame): Kommuner data for geometry processing.
        natomrade (gpd.GeoDataFrame): Natomrade data for geometry processing.

    Returns:
        pd.DataFrame: A GeoDataFrame containing the processed data after geometry operations.
    """
    df = format_df(chunk)
    df = drop_nan_row(df, "rid")
    df = set_dtypes(df)
    df = sort_df(df, ["rid", "Tidpunkt"], [True, True])

    if not qc(df, year):
        tqdm.write(f"QC fauled for year {year} in table {chunk.name}")
        return None

    df = drop_column(df, "Tidpunkt")
    df = calculate_energy_statistics(df)
    df = add_geometry(df, grid)

    gdf = to_gdf(df, crs="EPSG:3006")

    gdf = polygon_intersection(gdf, kommuner)
    gdf = largest_area(gdf)
    gdf = polygon_intersection(gdf, natomrade)
    gdf = largest_area(gdf)

    return gdf


def get_ordered_query(table: str, sort_column: str) -> str:
    return f"SELECT * FROM {table} ORDER BY {sort_column}"


def iterate_chunks(conn, table: str, chunk_size: int):
    query = get_ordered_query(table)
    return pd.read_sql_query(query, conn, chunksize=chunk_size)


def process_table(
    table, conn, year, grid, kommuner, natomrade, chunk_size: int
) -> pd.DataFrame:
    """
    Process a single table by iterating over its chunks and processing each chunk.

    Args:
        table (str): The name of the table to process.
        conn: Database connection object.
        year (int): The year for which the data is being processed.
        grid: Grid data for geometry processing.
        kommuner: Kommuner data for geometry processing.
        natomrade: Natomrade data for geometry processing.
        chunk_size (int): The size of each chunk to process.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data from the table.
    """
    chunks = []

    for chunk in iterate_chunks(conn, table, chunk_size):
        gdf = process_chunk(chunk, year, grid, kommuner, natomrade)

        if gdf is not None:
            chunks.append(gdf)

    return pd.concat(chunks, ignore_index=True)


def process_region(region: str):
    path = db_path(region)
    conn, cursor = db_connect(path)
    tables = db_tables(cursor)
    years = db_years(tables)

    grid = load_grid()
    kommuner = load_kommun()
    natomrade = load_natomrade()

    for year in tqdm(years, desc=f"Years ({region})", position=0, leave=True):
        expected_rows = get_expected_length(year)
        chunk_size = 10 * expected_rows
        filtered_tables = filter_tables(tables, year)

        for table in tqdm(filtered_tables, desc="Tables", position=1, leavel=False):
            gdf = process_table(
                table, conn, year, grid, kommuner, natomrade, chunk_size
            )
            as_parquet(gdf, region, table)

    cursor.close()
    conn.close()


def main():
    regions = ["12"]
    for region in regions:
        process_region(region)


if __name__ == "__main__":
    main()
    # regions = ["12"]

    # grid = load_grid()
    # kommuner = load_kommun()
    # natomrade = load_natomrade()

    # for region in regions:
    #     path = db_path(region)
    #     conn, cursor = db_connect(path)
    #     tables = db_tables(cursor)
    #     years = db_years(tables)

    #     for year in tqdm(years, desc="Years", position=0, leave=True):
    #         tables_filtered = filter_tables(tables, year)

    #         # How many rows in a table equal to one RutID?
    #         one_rid = get_expected_length(year)

    #         # How many multiples of one_rid to process in each chunk?
    #         n_rid = 10

    #         for table in tqdm(tables_filtered, desc="Tables", position=1, leave=False):
    #             chunks = []

    #             for chunk in load_chunk(table, conn, chunk_size=n_rid * one_rid):
    #                 processed_chunk = process_chunk(
    #                     chunk, year, grid, kommuner, natomrade
    #                 )

    #                 if processed_chunk is not None:
    #                     chunks.append(processed_chunk)
    #                 # print()
    #                 # print(processed_chunk.head())
    #                 # print(processed_chunk.shape)

    #             gdf = pd.concat(chunks, ignore_index=True)
    #             as_parquet(gdf, region, table)

    #     cursor.close()
    #     conn.close()
