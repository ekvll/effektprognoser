import typer
from eptools.processing.db import db_connect, db_path, db_tables

app = typer.Typer()

"""
Reads a SQLite3 database. One database per region. For example, is region is "10", then the "Effektmodell_10.sqlite" database is read.

All tables inside a database are iterated over. The script performs preprocessing of each table. Then saves each preprocessed table as a parquet.
"""


@app.command()
def run(region: str):
    print("sql_to_parquet")
    print(f"Running pipeline for region {region}")
    path: str = db_path(region)
    conn, cursor = db_connect(path)
    tables = db_tables(cursor)
