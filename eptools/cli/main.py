# eptools/cli/main.py

import typer
from eptools.cli import sql_to_parquet, kommun

"""
This script simply orchestrates all CLI subcommands.
"""

app = typer.Typer()
app.add_typer(sql_to_parquet.app, name="sql-to-parquet")
app.add_typer(kommun.app, name="kommun")

if __name__ == "__main__":
    app()
