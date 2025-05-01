import os
import sqlite3
from eptools.utils.paths import SQL_DIR


def db_path(region: str) -> str:
    """
    Generate the full path to a SQLite database.

    Args:
        region (str): Region whose SQLite data to connect to.

    Returns:
        str: Full path to SQLite database.
    """
    db_dir = f"Effektmodell_{region}"
    db_name = db_dir + ".sqlite"
    path = os.path.join(db_dir, db_name)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"File '{path}' is not found")


def db_connect(path: str):
    """
    Establish connection to a SQLite data base given the path to the datbase.

    Args:
        path (str): Path to the database.

    Returns:
        sqlite3.Connection: SQLite3 connection object.
        sqlite3.Cursor: SQLite3 cursor object
    """
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()

        try:
            conn.execute("SELECT 1;")
            return conn, cursor
        except sqlite3.Error as e:
            raise e(f"Connection to {path} is not valid")
    except sqlite3.Error as e:
        raise e(f"Could not connect to {path}")


def db_tables(cursor) -> list[str]:
    """
    Retreive name of all tables in SQLite3 database.

    Args:
        cursor (sqlite3.Cursor): SQLite3 cursor object.

    Returns:
        list[str]: List of strings of all tablenames in SQLite3 database.
    """
    sql_query = "SELECT name FROM sqlite_master;"
    cursor.execute(sql_query)

    # Put the table names in a list
    tables = [table[0] for table in cursor.fetchall()]
    #   tqdm.write(f"Found {len(tables)} tables in database")
    return tables
