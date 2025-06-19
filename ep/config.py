import os
import sys
from pathlib import Path

from tqdm import tqdm

"""
Manually mount the Windows drive in WSL2
sudo mount -t drvfs D: /mnt/d/
"""

# Absolute path to SQL directory (adjust as needed)
# SQL_DIR = Path("/mnt/d/effektprognoser/sqlite")
SQL_DIR = Path("/mnt/d/effektprognoser/sqlite")


def sql_path_exists():
    global SQL_DIR

    if not SQL_DIR or str(SQL_DIR) == ".":  # an empty Path("") equates to str(".")
        sql_path = input("Please enter the path to your SQL directory: ").strip()
        if not sql_path:
            print("No path specified. Exiting.")
            sys.exit(1)
        # Update the config file itself
        with open(__file__, "r") as f:
            lines = f.readlines()
        with open(__file__, "w") as f:
            for line in lines:
                if line.strip().startswith("SQL_DIR"):
                    f.write(f'SQL_DIR = Path(r"{sql_path}")\n')
                else:
                    f.write(line)
        print(f"Config file updated with SQL_DIR = {sql_path}.")
        sys.exit(0)


# Ensure SQL_DIR exists
# if not SQL_DIR.exists():
#     existence = input(
#         f"SQL directory does not exist: {SQL_DIR}. Continue anyway? (y/n): "
#     )
#     if existence.lower() != "y":
#         pass
#     elif existence.lower() == "n":
#         raise FileNotFoundError(f"SQL directory does not exist: {SQL_DIR}")

# Automatically resolve project root (2 levels above current file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
BG_DIR = PROJECT_ROOT / "data" / "background"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
GEOJSON_DIR = PROJECT_ROOT / "data" / "geojson"
EXCEL_DIR = PROJECT_ROOT / "data" / "excel"

# Temporary directory for GeoJSON files
GEOJSON_TMP_DIR = GEOJSON_DIR / "_tmp"

# Test data
TEST_DIR = PROJECT_ROOT / "data" / "_test"


def validate_paths(paths: list[Path]) -> None:
    """Raise FileNotFoundError if any path does not exist."""
    for path in paths:
        if not path.exists():
            path_create = input(
                f"Path does not exist: {path}. Do you want to create it? (y/n): "
            )
            if path_create.lower() == "y":
                path.mkdir(parents=True, exist_ok=True)
                tqdm.write(f"Created path: {path}")
            else:
                raise FileNotFoundError(f"Path does not exist: {path}")


paths = [
    DATA_DIR,
    BG_DIR,
    PARQUET_DIR,
    GEOJSON_DIR,
    GEOJSON_TMP_DIR,
    EXCEL_DIR,
    TEST_DIR,
]
validate_paths(paths)


def create_region_directories(paths: list[str], regions: list[str]) -> None:
    """Create region directories if they do not exist."""
    for path in paths:
        for region in regions:
            region_path = os.path.join(path, region)
            if not os.path.exists(region_path):
                os.makedirs(region_path)


regions = ["06", "07", "08", "10", "12", "13"]
regions_alla = ["06", "07", "08", "10", "12", "13", "alla"]

paths_regions = [PARQUET_DIR, GEOJSON_DIR, GEOJSON_TMP_DIR, EXCEL_DIR]
create_region_directories(paths_regions, regions)


default_raps = [
    "Flerbostadshus",
    "Smahus",
    "LOM Flerbostadshus",
    "LOM Smahus",
    "RAPS 1 3",
    "RAPS 4",
    "RAPS 5",
    "RAPS 6",
    "RAPS 7",
    "RAPS 8",
    "RAPS 9",
    "RAPS 10",
    "RAPS 11",
    "RAPS 12",
    "RAPS 13",
    "RAPS 14",
    "RAPS 15",
    "RAPS 16",
    "RAPS 17",
    "RAPS 18",
    "RAPS 19",
    "RAPS 20",
    "RAPS 21",
    "RAPS 22",
    "RAPS 23",
    "RAPS 24",
    "RAPS 27",
    "RAPS 7777",
    "RAPS 8888",
    "PB",
    "LL",
    "TT DEP",
    "TT DEST",
    "TT RESTSTOP",
]

default_years = ["2022", "2027", "2030", "2040"]

raps_categories = {
    "total": default_raps,
    "bostader": ["Flerbostadshus", "Smahus", "LOM Flerbostadshus", "LOM Smahus"],
    "industri_och_bygg": [
        "RAPS 4",
        "RAPS 5",
        "RAPS 6",
        "RAPS 7",
        "RAPS 8",
        "RAPS 9",
        "RAPS 10",
        "RAPS 11",
        "RAPS 13",
        "RAPS 14",
        "RAPS 15",
        "RAPS 16",
        "RAPS 17",
        "RAPS 18",
        "RAPS 19",
        "RAPS 20",
        "RAPS 21",
        "RAPS 22",
        "RAPS 23",
        "RAPS 24",
        "RAPS 27",
    ],
    "offentlig_och_privat_sektor": ["RAPS 7777", "RAPS 8888"],
    "transport": ["PB", "LL", "TT DEP", "TT DEST", "TT RESTSTOP"],
    "jordbruk_skogsbruk": ["RAPS 1 3"],
}

if __name__ == "__main__":
    sql_path_exists()
    for path in paths:
        tqdm.write(str(path))
    validate_paths(paths)
    create_region_directories(paths_regions, regions)
