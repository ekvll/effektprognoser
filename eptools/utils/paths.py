import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EXTERNAL_ROOT = "D:\\effektprognoser"


# Paths to project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
EXCEL_DIR = os.path.join(DATA_DIR, "excel")
PKL_DIR = os.path.join(DATA_DIR, "pkl")
GEOJSON_DIR = os.path.join(DATA_DIR, "geojson")
GEOJSON_TMP_DIR = os.path.join(GEOJSON_DIR, "_tmp")

# Paths to external SDD
SQL_DIR = os.path.join(EXTERNAL_ROOT, "sqlite")

for dir in [
    DATA_DIR,
    PARQUET_DIR,
    EXCEL_DIR,
    SQL_DIR,
    PKL_DIR,
    GEOJSON_DIR,
    GEOJSON_TMP_DIR,
]:
    if not os.path.isdir(dir):
        print(f"Path does not exist {dir}")
        print("Creating path...")
        os.makedirs(dir, exist_ok=True)
