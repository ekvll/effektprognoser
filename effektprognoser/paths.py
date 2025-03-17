from pathlib import Path
import os

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# External SDD path
EXTERNAL_ROOT = "/mnt/d/effektprognoser/"

# Define important directories
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SQL_DIR = os.path.join(EXTERNAL_ROOT, "sqlite")
CSV_DIR = os.path.join(EXTERNAL_ROOT, "csv")
GEOJSON_DIR = os.path.join(EXTERNAL_ROOT, "geojson")
PARQUET_DIR = os.path.join(EXTERNAL_ROOT, "parquet")
PARQUET_DIR_LOCAL = os.path.join(PROJECT_ROOT, "data", "parquet")
EXCEL_DIR = os.path.join(EXTERNAL_ROOT, "excel")
if __name__ == "__main__":
    print(f"Data Directory: {DATA_DIR}")
