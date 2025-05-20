import os
import geopandas as gpd
from pathlib import Path
from typing import Optional

from ep.config import PARQUET_DIR


def parquet_filenames(region: str) -> list[str]:
    """Get the list of parquet filenames for a given region."""
    region_path = Path(PARQUET_DIR) / region

    if not region_path.exists():
        raise FileNotFoundError(f"Region path {region_path} does not exist")

    filenames = [
        f.name for f in region_path.iterdir() if f.is_file() and f.suffix == ".parquet"
    ]

    if not filenames:
        raise FileNotFoundError(f"No parquet files found in {region_path}")

    return filenames


def load_parquet(
    filename: str, region: str, cols: Optional[list[str]] = None
) -> gpd.GeoDataFrame:
    """Load a parquet file into a GeoDataFrame."""
    file_path = Path(PARQUET_DIR) / region / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist")

    df = gpd.read_parquet(file_path)
    return df[cols] if cols else df
