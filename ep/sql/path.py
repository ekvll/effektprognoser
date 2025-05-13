import os

from ep.config import SQL_DIR


def db_path(region: str) -> str:
    """Create the path to the SQLite database file for a given region."""
    dir_name = f"Effektmodell_{region}"
    db_name = dir_name + ".sqlite"
    db_path = os.path.join(SQL_DIR, dir_name, db_name)
    if os.path.isfile(db_path):
        return db_path
    raise FileNotFoundError(f"Database file {db_path} not found.")
