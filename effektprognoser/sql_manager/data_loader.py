import sqlite3
import pandas as pd


def load_table_in_chunks(
    conn: sqlite3.Connection, table: str, column_names: list[str], chunk_size: int
) -> pd.DataFrame:
    # Join the column names, stored in a list, as a single string
    columns_str = ", ".join(column_names)

    sql_query = f"SELECT {columns_str} FROM {table};"

    # Load the table in chunks using the SQL query
    chunks = pd.read_sql_query(sql_query, conn, chunksize=chunk_size)

    # Concatenate the chunks into a single DataFrame
    df = pd.concat(chunks, ignore_index=True)

    df = df.rename(columns={"rut_id": "rid"})
    df = df.sort_values(by=["rid", "Tidpunkt"], ascending=[True, True]).reset_index(
        drop=True
    )
    print(f"Table {table} shape: {df.shape}")
    return df
