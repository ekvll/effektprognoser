import os
import sqlite3
from effektprognoser.paths import SQL_DIR


def gen_db_path(region: str) -> str:
    region_dir = f"Effektmodell_{region}"
    db_name = region_dir + ".sqlite"
    return os.path.join(SQL_DIR, region_dir, db_name)


def connect_to_db(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    try:
        connection = sqlite3.connect(db_path)
        if is_connection_valid(connection):
            print(f"Connected to {db_path}")
            return connection, connection.cursor()
        else:
            raise sqlite3.Error("Connection to database is not valid")
    except sqlite3.Error:
        raise sqlite3.Error(f"Could not connecto to {db_path}")


def is_connection_valid(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.Error:
        return False


def close_connection(connection, cursor) -> None:
    cursor.close()
    connection.close()


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
