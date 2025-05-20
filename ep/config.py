import os
from pathlib import Path
from tqdm import tqdm

# Absolute path to SQL directory (adjust as needed)
SQL_DIR = Path("/mnt/d/effektprognoser/sqlite")

# Automatically resolve project root (2 levels above current file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
BG_DIR = PROJECT_ROOT / "data" / "background"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"


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


paths = [SQL_DIR, DATA_DIR, BG_DIR, PARQUET_DIR]
validate_paths(paths)


def create_region_directories(paths: list[str], regions: list[str]) -> None:
    """Create region directories if they do not exist."""
    for path in paths:
        for region in regions:
            region_path = os.path.join(path, region)
            if not os.path.exists(region_path):
                os.makedirs(region_path)


regions = ["06", "07", "08", "10", "12", "13"]
paths_regions = [PARQUET_DIR]
create_region_directories(paths_regions, regions)


if __name__ == "__main__":
    for path in paths:
        tqdm.write(path)
    validate_paths(paths)
    create_region_directories(paths_regions, regions)
