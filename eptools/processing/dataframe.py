import os
import pickle
import geopandas as gpd
import pandas as pd
import numpy as np
from eptools.utils.paths import PARQUET_DIR, PKL_DIR


def max_index(arr):
    return np.max(arr), np.argmax(arr)


def parquet_filenames(region):
    path = os.path.join(PARQUET_DIR, region)

    if os.path.exists(path):
        files = os.listdir(path)

        if len(files) > 0:
            return files
        raise ValueError("Path exists but contain no files")
    else:
        raise NameError(f"Path {path} does not exist")


def load_parquet(file, region, cols: list[str] = None):
    path = os.path.join(PARQUET_DIR, region, file)

    if os.path.isfile(path):
        if cols:
            return gpd.read_parquet(path)[cols]
        return gpd.read_parquet(path)
    raise FileNotFoundError(f"File {path} does not exist")


def year_from_filename(filename: str) -> str:
    return filename.split("_")[1]


def as_pickle(data: dict, filename, region) -> None:
    if filename.endswith(".parquet"):
        filename = filename.replace(".parquet", ".pkl")
    dir = os.path.join(PKL_DIR, region)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, filename)
    with open(path, "wb") as f:
        pickle.dump(data, f)
