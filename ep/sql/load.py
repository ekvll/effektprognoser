import sqlite3

import pandas as pd


def get_column_names(cursor: sqlite3.Cursor, table: str) -> list[str]:
    """Get the column names of a table in a SQLite database."""
    sql_query = f"PRAGMA table_info({table});"
    cursor.execute(sql_query)
    columns = [c[1] for c in cursor.fetchall()]
    return columns


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the DataFrame."""
    df = df.rename(columns={"rut_id": "rid"})
    df = df.sort_values(by=["rid", "Tidpunkt"], ascending=[True, True])
    return df.reset_index(drop=True)


def build_select_query(table: str, columns: list[str]) -> str:
    """Build a SQL SELECT query for a table."""
    columns_str = ", ".join(columns)
    return f"SELECT {columns_str} FROM {table};"


def load_table_chunks(
    conn: sqlite3.Connection, cursor: sqlite3.Cursor, table: str, chunk_size: int = 1000
) -> pd.DataFrame:
    """Load a table from a SQLite database in chunks."""
    columns = get_column_names(cursor, table)
    sql_query = build_select_query(table, columns)

    # Load the data in chunks
    chunks = pd.read_sql_query(sql_query, conn, chunksize=chunk_size)
    df = pd.concat(chunks, ignore_index=True)

    return format_df(df)
