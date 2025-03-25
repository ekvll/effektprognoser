import os
import numpy as np
import geopandas as gpd
from ..sql_manager import filter_tables


def process_region(region, input_path, files, years):
    result = {}

    for year in years:
        files_filtered = filter_tables(files, year)

        for file in files_filtered:
            output_filename = "_".join(file.split("_")[2:]).split("_V1")[0]
            input_filepath = os.path.join(input_path, file)

            gdf = gpd.read_parquet(input_filepath)
            unique_kommuner = gdf["kn"].unique()

            for kommun in unique_kommuner:
                if kommun not in result:
                    result[kommun] = []

                df_kommun = gdf[gdf["kn"] == kommun].reset_index(drop=True)

                aggregated_lp = np.zeros(8784 if year == "2040" else 8760)
                for lp_str in df_kommun.lp:
                    aggregated_lp += lp_str

                result[kommun].append(
                    {
                        "year": year,
                        "region": region,
                        "out_fn": output_filename,
                        "effektbehov": max(aggregated_lp),
                        "elanvandning": sum(aggregated_lp),
                        "kommunkod": df_kommun["kk"].iloc[0],
                    }
                )

    return result
