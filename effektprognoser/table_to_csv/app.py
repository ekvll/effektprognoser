import gc
from .qc import quality_check
from ..geo.load_geo_data import load_grid, load_kommuner, load_natomrade
from ..geo.utils import (
    add_geometry,
    convert_df_to_gdf,
    keep_largest_area,
    intersection_by_polygon,
)
from ..save import save_table_as_csv, save_table_as_geojson, save_table_as_parquet
from ..sql.utils import (
    gen_db_path,
    connect_to_db,
    close_connection,
    get_table_names_in_db,
    get_years_in_table_names,
    print_db_content,
    filter_tables,
    get_table_column_names,
)

from ..sql.load_table import load_table_in_chunks
from .utils import (
    drop_row_if_nan_in_column,
    set_data_types,
    sort_df,
    calc_lastprofil_effektbehov_elanvandning,
)


def main(regions) -> None:
    gdf_grid = load_grid()
    gdf_kommuner = load_kommuner()
    gdf_natomrade = load_natomrade()

    for region_index, region in enumerate(regions):
        print(f"Processing region {region} ({region_index + 1}/{len(regions)})")

        db_path = gen_db_path(region)
        conn, cursor = connect_to_db(db_path)
        tables = get_table_names_in_db(cursor)
        years = get_years_in_table_names(tables)
        # print_db_content(tables, years)

        for year in years:
            tables_filtered = filter_tables(tables, year)

            for table in tables_filtered:
                table_column_names = get_table_column_names(cursor, table)

                df = load_table_in_chunks(
                    conn, table, table_column_names, chunk_size=20000
                )

                # Process DataFrame
                df = drop_row_if_nan_in_column(df, "rid")
                df = set_data_types(df)
                df = sort_df(df, ["rid", "Tidpunkt"])

                # Perform quality check
                if quality_check(df, year):  # skip table if qc failes (True)
                    continue

                # Calculate lp, eb, ea
                df = calc_lastprofil_effektbehov_elanvandning(df)

                df = add_geometry(df, gdf_grid)

                gdf = convert_df_to_gdf(df)
                gdf = intersection_by_polygon(gdf, gdf_kommuner)
                gdf = keep_largest_area(gdf)
                gdf = intersection_by_polygon(gdf, gdf_natomrade)
                gdf = keep_largest_area(gdf)

                # print(gdf.head())
                # print(gdf.columns)

                # save_table_as_csv(gdf, region, table)
                # save_table_as_geojson(gdf, region, table)
                save_table_as_parquet(gdf, region, table)

                # Clean up
                del df, gdf
                gc.collect()

        close_connection(conn, cursor)
