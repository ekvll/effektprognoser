import numpy as np
import pandas as pd


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
