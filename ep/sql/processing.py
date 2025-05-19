import sqlite3
import numpy as np
import pandas as pd


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
        year = int(table.split("_")[1])
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
