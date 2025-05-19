import pandas as pd


def has_no_all_zero_columns(df: pd.DataFrame) -> bool:
    return not any((df[col] == 0.0).all() for col in df.columns)


def has_no_all_nan_columns(df: pd.DataFrame) -> bool:
    return not any(df[col].isna().all() for col in df.columns)


def has_complete_days(df: pd.DataFrame, year: int) -> bool:
    days_in_df = int(df.shape[0] / df.tid.nunique())
    expected_days = 8784 if year in [2020, 2040] else 8760
    return days_in_df == expected_days


def qc(df: pd.DataFrame, year: int) -> bool:
    if isinstance(year, str):
        year = int(year)

    checks = {
        "no all-zero columns": has_no_all_zero_columns(df),
        "no all-NaN columns": has_no_all_nan_columns(df),
        "complete days for year": has_complete_days(df, year),
        "not empty": not df.empty,
    }

    failed = [name for name, result in checks.items() if not result]
    for name in failed:
        print(f"Failed QC: {name}")

    return not failed
