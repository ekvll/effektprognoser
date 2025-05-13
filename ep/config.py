import os
from pathlib import Path


SQL_DIR = Path("/mnt/d/effektprognoser/sqlite")


def strict_path_validation(lst: list[str]) -> None:
    """
    Validate that all paths in the list exist.
    """
    for path in lst:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist.")


strict_path_validation([SQL_DIR])


if __name__ == "__main__":
    print(SQL_DIR)
    strict_path_validation([SQL_DIR])
