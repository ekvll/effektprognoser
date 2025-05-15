import sqlite3


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
