import os
import sqlite3

from tqdm import tqdm

from ep.cli.sql2parquet import (
    db_connect,
    db_path,
    db_tables,
    db_years,
    filter_tables,
    load_table_chunks,
    sort_df,
)
from ep.config import TEST_DIR


def db_test_path(region: str = "test") -> str:
    """Create the path to the SQLite database file for the test mode."""
    dir_name = f"Effektmodell_{region}"
    dir_path = os.path.join(TEST_DIR, "sqlite", dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    db_name = dir_name + ".sqlite"
    db_path = os.path.join(dir_path, db_name)
    return db_path


def main():
    region = "10"
    tqdm.write(f"Generating test data based on region {region}")

    path = db_path(region)
    conn, cursor = db_connect(path)
    tables = db_tables(cursor)
    years = db_years(tables)

    # Open a connection to the new database (create if doesn't exist)
    test_db_path = db_test_path()
    test_conn = sqlite3.connect(test_db_path)

    for year in tqdm(years, desc="Years", position=0, leave=False):
        tables_filtered = filter_tables(tables, year)

        rows = 8784 if str(year) == "2040" else 8760
        n_rows = 10
        crop_rows = int(n_rows * rows)

        for table in tqdm(tables_filtered, desc="Tables", position=1, leave=False):
            df = load_table_chunks(conn, cursor, table)
            df = sort_df(df, ["rid", "Tidpunkt"], [True, True])

            df_crop = df.head(crop_rows)

            df_crop.to_sql(table, test_conn, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
