import sqlite3


def connect_to_db(path: str) -> sqlite3.Connection:
    try:
        return sqlite3.connect(path)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Failed to connect to database: {e}")


def get_cursor(conn: sqlite3.Connection) -> sqlite3.Cursor:
    return conn.cursor()


def validate_connection(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("SELECT 1")
    except sqlite3.OperationalError as e:
        conn.close()
        raise sqlite3.OperationalError(f"Database connection failed: {e}")


def db_connect(path: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create a connection to the SQLite database and return the connection and cursor."""
    conn = connect_to_db(path)
    cursor = get_cursor(conn)
    validate_connection(conn)
    return conn, cursor
