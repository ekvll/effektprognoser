import sqlite3
import tempfile
import os
import pytest

from ep.sql.connection import connect_to_db, get_cursor, validate_connection, db_connect


def test_connect_to_db_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = connect_to_db(tmp.name)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()


def test_get_cursor() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = sqlite3.connect(tmp.name)
        cursor = get_cursor(conn)
        assert isinstance(cursor, sqlite3.Cursor)
        conn.close()


def test_validate_connection_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = sqlite3.connect(tmp.name)
        validate_connection(conn)
        conn.close()


def test_db_connect_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn, cursor = db_connect(tmp.name)
        assert isinstance(conn, sqlite3.Connection)
        assert isinstance(cursor, sqlite3.Cursor)
        conn.close()
