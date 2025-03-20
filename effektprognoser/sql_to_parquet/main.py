import gc
from .database_utils import fetch_data_for_region, close_db_connection
from .data_processing import drop_row_if_nan_in_column, set_data_types, sort_df
from .feature_engineering import calc_lastprofil_effektbehov_elanvandning
from .qc import quality_check
from ..geometry import (
    load_grid,
    load_kommuner,
    load_natomrade,
    add_geometry,
    convert_df_to_gdf,
    keep_largest_area,
    intersection_by_polygon,
)
from ..sql_manager import filter_tables, load_table_in_chunks, get_table_column_names
from .save import as_parquet


def process_region(region, gdf_grid, gdf_kommuner, gdf_natomrade):
    conn, cursor, tables, years = fetch_data_for_region(region)

    for year in years:
        tables_filtered = filter_tables(tables, year)

        for table in tables_filtered:
            df = load_table_in_chunks(
                conn, table, get_table_column_names(cursor, table), chunk_size=20000
            )

            df = drop_row_if_nan_in_column(df, "rid")
            df = set_data_types(df)
            df = sort_df(df, ["rid", "Tidpunkt"])

            if quality_check(df, year):  # Skip table if QC fails
                continue

            df = calc_lastprofil_effektbehov_elanvandning(df)
            df = add_geometry(df, gdf_grid)

            gdf = convert_df_to_gdf(df)
            gdf = intersection_by_polygon(gdf, gdf_kommuner)
            gdf = keep_largest_area(gdf)
            gdf = intersection_by_polygon(gdf, gdf_natomrade)
            gdf = keep_largest_area(gdf)

            as_parquet(gdf, region, table)

            del df, gdf
            gc.collect()

    close_db_connection(conn, cursor)


def main(regions) -> None:
    gdf_grid = load_grid()
    gdf_kommuner = load_kommuner()
    gdf_natomrade = load_natomrade()

    for region_index, region in enumerate(regions):
        print(f"Processing region {region} ({region_index + 1}/{len(regions)})")
        process_region(region, gdf_grid, gdf_kommuner, gdf_natomrade)
