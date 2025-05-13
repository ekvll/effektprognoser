import sqlite3
import pandas as pd
import pytest

from ep.sql.load import (
    get_column_names,
    build_select_query,
    format_df,
    load_table_chunks,
)


def test_get_column_names_returns_correct_names() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER, name TEXT);")

    cols = get_column_names(cursor, "test")

    assert cols == ["id", "name"]

    conn.close()


def test_format_df_renames_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "rut_id": [2, 1, 1],
            "Tidpunkt": ["2025-05-13", "2025-05-14", "2025-05-15"],
            "value": [300, 100, 200],
        }
    )

    formatted = format_df(df)

    assert list(formatted.columns) == ["rid", "Tidpunkt", "value"]
    assert formatted.iloc[0]["rid"] == 1  # First row after sorting
    assert formatted.iloc[0]["Tidpunkt"] == "2025-05-14"


def test_load_table_chunks() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (rut_id INTEGER, Tidpunkt TEXT, value REAL);")
    cursor.executemany(
        "INSERT INTO test (rut_id, Tidpunkt, value) VALUES (?, ?, ?);",
        [(1, "2025-05-13", 300), (2, "2025-05-14", 100), (1, "2025-05-15", 200)],
    )
    conn.commit()

    df = load_table_chunks(conn, cursor, "test", chunk_size=2)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["rid", "Tidpunkt", "value"]
    assert df.iloc[0]["rid"] == 1  # First row after sorting

    conn.close()
