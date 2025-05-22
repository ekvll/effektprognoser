import numpy as np
import pandas as pd
from tqdm import tqdm

from ep.paths import parquet_filenames, as_parquet
from ep.data_manager import (
    get_kommuner_in_region,
    load_parquet,
    calc_total_loadprofile_per_kommun,
)
from ep.config import default_raps, default_years


def main(region):
    """Main function to process SQLite database tables."""

    tqdm.write(f"Processing region: {region}")

    filenames = parquet_filenames(region)
    kommuner = get_kommuner_in_region(filenames, region)

    tqdm.write(f"Found {len(kommuner)} kommuner in region {region}:")

    for kommun in kommuner:
        df_out = pd.DataFrame(columns=default_years, index=default_raps)
        for filename in filenames:
            gdf = load_parquet(filename, region)
            gdf = gdf[gdf["kk"].astype(str).str[:2] == str(region)]
            gdf_kommun = gdf.loc[gdf["kn"] == kommun]
            if gdf_kommun.empty:
                tqdm.write(f"DataFrame for kommun {kommun} is empty. Skipping.")
                continue
            print(gdf_kommun.head())
            exit(0)
            kommun_kod = gdf_kommun["kk"].unique()

            year = filename.split("_")[1]
            raps = " ".join(filename.split("_")[2:-1])

            df_out[year] = np.nan

            expected_length = 8784 if year == "2040" else 8760
            aggregated_lp = np.zeros(expected_length)

            for i, lp in enumerate(gdf_kommun["lp"]):
                lp_array = np.asarray(lp)
                if lp_array.shape[0] != expected_length:
                    raise ValueError(
                        f"Load profile at index {i} has incorrect length: "
                        f"{lp_array.shape[0]}, expected: {expected_length}"
                    )
                aggregated_lp += lp_array

            df_out.loc[raps, year] = max(aggregated_lp)
        as_parquet(gdf, region, f"{kommun_kod}_{kommun}", "_parquet2kommun_effektbehov")


if __name__ == "__main__":
    # from ep.config import regions
    regions = ["10"]
    for region in regions:
        main(region)
