from tqdm import tqdm

from ep.sql.connection import db_connect
from ep.sql.path import db_path
from ep.sql.processing import (
    db_tables,
    db_years,
    filter_tables,
    drop_nan_row,
    set_dtypes,
    sort_df,
    drop_column,
    calculate_energy_statistics,
    add_geometry,
    to_gdf,
    polygon_intersection,
    largest_area,
)
from ep.sql.load import load_table_chunks
from ep.sql.qc import qc
from ep.layers import load_grid, load_kommun, load_natomrade
from ep.save import as_parquet


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

    for year in years:
        tqdm.write(f"Processing year: {year}")
        tables_filtered = filter_tables(tables, year)

        for table in tqdm(tables_filtered, desc="Tables", position=0):
            df = load_table_chunks(conn, cursor, table)
            df = drop_nan_row(df, "rid")
            df = set_dtypes(df)
            df = sort_df(df, ["rid", "Tidpunkt"], [True, True])

            if not qc(df, year):
                tqdm.write(f"Skipping table {table}.")

            df = drop_column(df, "Tidpunkt")
            df = calculate_energy_statistics(df)
            df = add_geometry(df, grid)

            gdf = to_gdf(df, crs="EPSG:3006")

            gdf = polygon_intersection(gdf, kommuner)
            gdf = largest_area(gdf)
            gdf = polygon_intersection(gdf, natomrade)
            gdf = largest_area(gdf)

            as_parquet(gdf, region, table)


if __name__ == "__main__":
    from ep.config import regions

    for region in regions:
        main(region)
