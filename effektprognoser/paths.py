from pathlib import Path
import os

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# External SDD path
EXTERNAL_ROOT = "/mnt/d/effektprognoser/"
EXTERNAL_ROOT_WIN = "D:\effektprognoser"

# Define important directories
# Project root
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PARQUET_DIR_LOCAL = os.path.join(PROJECT_ROOT, "data", "parquet")

# External root
SQL_DIR = os.path.join(EXTERNAL_ROOT, "sqlite")
SQL_DIR_WIN = os.path.join(EXTERNAL_ROOT_WIN, "sqlite")
CSV_DIR = os.path.join(EXTERNAL_ROOT, "csv")
GEOJSON_DIR = os.path.join(EXTERNAL_ROOT, "geojson")
PARQUET_DIR = os.path.join(EXTERNAL_ROOT, "parquet")
EXCEL_DIR = os.path.join(EXTERNAL_ROOT, "excel")

if __name__ == "__main__":
    print(f"Data Directory: {DATA_DIR}")
