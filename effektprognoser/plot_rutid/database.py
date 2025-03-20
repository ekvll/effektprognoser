import os
import sqlite3
from effektprognoser.paths import DATA_DIR


def get_db_path(region):
    db_filename = f"Effektmodell_{region}.sqlite"
    return os.path.join(DATA_DIR, "rut_id", region, db_filename)


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor


def row_exists(cursor, table_name, row_id):
    cursor.execute(
        f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE rut_id = ? LIMIT 1);",
        (row_id,),
    )
    return cursor.fetchone()[0] == 1


def filter_tables_by_rutid(cursor, tables, rutid):
    return [table for table in tables if row_exists(cursor, table, rutid)]
