import sqlite3


def get_table_names_in_db(cursor: sqlite3.Cursor) -> list[str]:
    sql_query = "SELECT name FROM sqlite_master;"
    cursor.execute(sql_query)

    # Put the table names in a list
    tables = [table[0] for table in cursor.fetchall()]

    print(f"Found {len(tables)} tables in database")
    return tables


def get_years_in_table_names(tables: list[str]) -> list[str]:
    years = []
    for table in tables:
        year: str = table.split("_")[1]
        if year not in years:
            years.append(year)

    print(f"Found {len(years)} years in database ({', '.join(years)})")
    return sorted(years)


def print_db_content(tables: list[str], years: list[str]) -> None:
    len_tables = len(tables)

    print("Tables in database:")
    for tbl_idx, tbl in enumerate(tables):
        print(f"{tbl} ({tbl_idx + 1}/{len_tables})")

    print("Years in database:")
    for year in years:
        print(year)


def filter_tables(tables: list[str], year: str) -> list[str]:
    tables_filtered = [t for t in tables if year in t]
    return tables_filtered


def get_table_column_names(cursor: sqlite3.Cursor, table_name: str) -> list[str]:
    sql_query = f"PRAGMA table_info({table_name});"
    cursor.execute(sql_query)
    column_names = [column[1] for column in cursor.fetchall()]
    return column_names
