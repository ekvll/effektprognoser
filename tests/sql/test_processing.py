import sqlite3

from ep.sql.processing import db_tables, db_years


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
