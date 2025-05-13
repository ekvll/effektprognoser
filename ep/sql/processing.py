import sqlite3


def db_tables(cursor: sqlite3.Cursor) -> list[str]:
    """Get a list of all tables in the database."""
    sql_query = "SELECT name FROM sqlite_master;"
    cursor.execute(sql_query)

    # Put the table names in a list
    tables = [table[0] for table in cursor.fetchall()]
    return tables
