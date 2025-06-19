import os
import sqlite3

import pytest

from ep.cli.sql2parquet import pipeline
from ep.config import TEST_DIR


def _is_connection_open(conn):
    try:
        conn.execute("SELECT 1")
        return True
    except sqlite3.ProgrammingError:
        return False


def _test_conn():

    region = "test"
    dir_name = f"Effektmodell_{region}"
    db_name = dir_name + ".sqlite"
    db_path = os.path.join(TEST_DIR, "sqlite", dir_name, db_name)

    conn = sqlite3.connect(db_path)
    return conn, conn.cursor()


def test_connect_db_test_success() -> None:
    conn, _ = _test_conn()

    # check that conn is not None
    assert conn is not None

    # check that connection is open
    assert _is_connection_open(conn)

    # close the connection
    conn.close()

    # check that connection is closed
    assert not _is_connection_open(conn)

    # using the closed connection should raise ProgrammingError
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")


def test_end2end() -> None:
    conn, cursor = _test_conn()
    for gdf, table in pipeline(conn, cursor):
        assert not gdf.empty
