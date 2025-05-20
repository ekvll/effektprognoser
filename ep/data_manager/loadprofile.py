import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm

from ep.paths import load_parquet


def aggregate_loadprofile(
    gdf: pd.DataFrame | gpd.GeoDataFrame, year: str
) -> np.ndarray:
    """Aggregate the load profile for a specific kommun for a given year."""
    expected_length = 8784 if year == "2040" else 8760

    aggregated_lp = np.zeros(expected_length)

    for i, lp in enumerate(gdf["lp"]):
        lp_array = np.asarray(lp)
        if lp_array.shape[0] != expected_length:
            raise ValueError(
                f"Load profile at index {i} has incorrect length: "
                f"{lp_array.shape[0]}, expected: {expected_length}"
            )
        aggregated_lp += lp_array

    return aggregated_lp


def initialize_total_loadprofile() -> dict[str, np.ndarray]:
    """Initialize the total load profile dictionary."""
    return {
        "2022": np.zeros(8760),
        "2027": np.zeros(8760),
        "2030": np.zeros(8760),
        "2040": np.zeros(8784),
    }


def process_file_for_kommun(
    filename: str, kommun: str, region: str
) -> tuple[str, np.ndarray] | None:
    """Process a single parquet file for one kommun.
    Retuns (year, aggregated_profile) or None."""
    gdf = load_parquet(filename, region)
    gdf_kommun = gdf.loc[gdf["kn"] == kommun]

    if gdf_kommun.empty:
        tqdm.write(f"DataFrame for kommun {kommun} is empty. Skipping.")
        return None

    year = filename.split("_")[1]
    profile = aggregate_loadprofile(gdf_kommun, year)

    return year, profile


def calc_total_loadprofile_per_kommun(
    filenames: list[str], kommuner, region: str
) -> dict:
    """Calculate the total load profile per kommun."""
    total_loadprofile = {}

    for kommun in kommuner:
        total_loadprofile[kommun] = initialize_total_loadprofile()

        for filename in filenames:
            result = process_file_for_kommun(filename, kommun, region)
            if result is None:
                continue

            year, profile = result
            total_loadprofile[kommun][year] += profile

    return total_loadprofile
