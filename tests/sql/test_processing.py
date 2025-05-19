import sqlite3
import pytest
import pandas as pd
import numpy as np

from ep.sql.processing import db_tables, db_years, drop_nan_row, set_dtypes, sort_df


def test_db_tables_returns_table_names() -> None:
    # Use an in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create some tables
    cursor.execute("CREATE TABLE test1 (id INTEGER);")
    cursor.execute("CREATE TABLE test2 (name TEST);")
    conn.commit()

    # Call the function
    result = db_tables(cursor)

    # Check if the expected tables are in the result
    assert "test1" in result
    assert "test2" in result
    assert isinstance(result, list)
    assert all(isinstance(name, str) for name in result)

    conn.close()


def test_db_tables_with_no_tables() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    result = db_tables(cursor)
    assert result == []

    conn.close()


def test_db_years() -> None:
    tables = ["data_2021", "data_2020", "data_2022"]

    expected_years = ["2020", "2021", "2022"]

    # Call the function
    result = db_years(tables)

    # Check if the expected years are in the result
    assert result == expected_years


def test_drop_nan_row() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, None],
            "B": [4, None, 6],
        }
    )

    result = drop_nan_row(df)

    assert result.shape[0] == 1
    assert result.shape[1] == 2
    assert result.iloc[0]["A"] == 1
    assert result.iloc[0]["B"] == 4


def test_drop_nan_row_with_col() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, None],
            "B": [4, 5, 6],
        }
    )

    result = drop_nan_row(df)
    # Should drop the row with NaN in column A
    assert result.shape[0] == 2
    assert result["A"].isnull().sum() == 0


def test_drop_nan_row_with_invalid_col() -> None:
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError, match="Column 'missing' not found"):
        drop_nan_row(df, col="missing")


def test_drop_nan_row_all_cols_no_nan() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )

    result = drop_nan_row(df)

    # No rows should be dropped
    pd.testing.assert_frame_equal(result, df)


def test_set_dtypes_success():
    df = pd.DataFrame(
        {
            "rid": [1, 2],
            "Elanvandning": [0.1, 0.2],
            "Tidpunkt": [20220101, 20220102],
        }
    )
    result = set_dtypes(df)
    assert result.dtypes["rid"] == np.dtype("uint64")


def test_sort_df_multiple_columns():
    df = pd.DataFrame(
        {
            "a": [2, 1, 3],
            "b": [9, 8, 7],
        }
    )
    sorted_df = sort_df(df, col=["a", "b"], ascending=[True, False])
    assert sorted_df.iloc[0]["a"] == 1
