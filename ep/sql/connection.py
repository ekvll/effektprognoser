import sqlite3


def db_connect(path: str) -> (sqlite3.Connection, sqlite3.Cursor):
    """Create a connection to the SQLite database and return the connection and cursor."""
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        try:
            conn.execute("SELECT 1")
            return conn, cursor
        except sqlite3.OperationalError as e:
            conn.close()
            raise sqlite3.OperationalError(f"Database connection failed: {e}")
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to database: {e}")
