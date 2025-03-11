import pandas as pd
import geopandas as gpd


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
