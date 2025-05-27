import pytest

from ep.cli.sql2parquet_chunk import get_ordered_query


def test_get_ordered_query():
    table = "test_table"
    sort_column = "test_column"
    expected_query = f"SELECT * FROM {table} ORDER BY {sort_column}"

    actual_query = get_ordered_query(table, sort_column)

    assert actual_query == expected_query, (
        f"Expected: {expected_query}, but got: {actual_query}"
    )
