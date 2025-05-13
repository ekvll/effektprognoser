import sqlite3

from ep.sql.processing import db_tables


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
