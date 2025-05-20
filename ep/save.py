import os

from ep.config import PARQUET_DIR


def as_parquet(gdf, region, table):
    """Save a GeoDataFrame as a Parquet file."""
    dirpath = os.path.join(PARQUET_DIR, region)
    filepath = os.path.join(dirpath, table + ".parquet")
    gdf.to_parquet(filepath)
