import sqlite3
import os


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
