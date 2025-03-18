import gc
import pandas as pd
import numpy as np
from ..geometry import (
    load_grid,
    load_kommuner,
    load_natomrade,
    add_geometry,
    convert_df_to_gdf,
    keep_largest_area,
    intersection_by_polygon,
)
from ..save import save_table_as_parquet
from ..sqlite import (
    load_table_in_chunks,
    gen_db_path,
    connect_to_db,
    close_connection,
    get_table_names_in_db,
    get_years_in_table_names,
    filter_tables,
    get_table_column_names,
)


def quality_check(df, year) -> bool:
    if isinstance(year, str):
        year = int(year)

    if all_zeros_in_col(df):
        print("DataFrame column contain only zeroes. Skipping.")
        return True

    if all_nan_in_col(df):
        print("DataFrame contains NaN values in all columns. Skipping.")
        return True

    if false_number_of_days(df, year):
        print("Number of days in dataframe is not correct. Skipping.")
        return True

    if df_empty(df):
        print("DataFrame is empty. Skipping.")
        return True

    return False


def all_zeros_in_col(df) -> bool:
    for col in df.columns:
        if (df[col] == 0.0).all():
            return True
    return False


def all_nan_in_col(df) -> bool:
    for col in df.columns:
        if df[col].isna().all():
            return True
    return False


def false_number_of_days(df, year: int | str) -> bool:
    days_in_df = int(df.shape[0] / df.rid.nunique())

    if isinstance(year, str):
        year = int(year)

    if year in [2020, 2040]:
        days_in_year = 8784  # leap year
        if not days_in_df == days_in_year:
            return True

    else:
        days_in_year = 8760  # normal year
        if not days_in_df == days_in_year:
            return True

    return False


def df_empty(df) -> bool:
    return df.empty


def drop_row_if_nan_in_column(df, column_name):
    initial_shape = df.shape[0]

    try:
        df_drop = df.dropna(subset=[column_name])
        final_shape = df_drop.shape[0]

        shape_diff = initial_shape - final_shape
        # Check if rows were dropped
        if not shape_diff == 0:
            print(f"Dropped {shape_diff} rows due to NaN in column '{column_name}'")

        return df_drop

    except Exception:
        raise NameError(
            "Could not check if rows should be dropped as if NaN is in column '{column_name}'"
        )


def set_data_types(df: pd.DataFrame) -> pd.DataFrame:
    dtypes_dict = {"rid": np.uint64, "Elanvandning": np.float64, "Tidpunkt": np.uint64}
    _check_if_missing_columns(df, dtypes_dict)

    try:
        df = df.astype(dtypes_dict)
        return df
    except ValueError as e:
        print(f"Error occurred while setting data types: {e}")


def _check_if_missing_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise NameError(
            f"The following columns are missing from DataFrame: {missing_columns}"
        )


def sort_df(df, sort_columns, ascending: bool = True) -> pd.DataFrame:
    _check_if_missing_columns(df, sort_columns)

    # Ensure ascending is a list
    if isinstance(ascending, bool):
        ascending = [ascending] * len(sort_columns)
    elif len(ascending) != len(sort_columns):
        raise ValueError(
            "The length of the 'ascending' list must match the number of sort columns"
        )
    try:
        df_sorted = df.sort_values(by=sort_columns, ascending=ascending).reset_index(
            drop=True
        )
        return df_sorted
    except Exception as e:
        raise ValueError(f"Error occurred while sorting the DataFrame: {e}")


def calc_lastprofil_effektbehov_elanvandning(df: pd.DataFrame):
    df.drop(columns=["Tidpunkt"], inplace=True)
    transposed = df.groupby("rid")["Elanvandning"].apply(list).reset_index(name="lp")

    transposed["eb"] = transposed["lp"].apply(lambda x: max(x))
    transposed["ea"] = transposed["lp"].apply(lambda x: sum(x))

    return transposed


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

                # save_table_as_csv(gdf, region, table)
                # save_table_as_geojson(gdf, region, table)
                save_table_as_parquet(gdf, region, table)

                # Clean up
                del df, gdf
                gc.collect()

        close_connection(conn, cursor)
