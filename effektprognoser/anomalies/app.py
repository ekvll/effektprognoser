import os
import geopandas as gpd
import numpy as np
import pandas as pd
from effektprognoser.paths import PARQUET_DIR_LOCAL, LOG_DIR
from effektprognoser.sql.utils import (
    get_table_names_in_db,
    get_years_in_table_names,
    filter_tables,
    gen_db_path_local,
)
from effektprognoser.qc import QCPipelines


def main(regions):
    for region_index, region in enumerate(regions):
        input_path = os.path.join(PARQUET_DIR_LOCAL, region)
        files = os.listdir(input_path)
        years = get_years_in_table_names(files)

        for year in years:
            print(f"Year: {year}")
            files_filtered = filter_tables(files, year)

            for file in files_filtered:
                input_filepath = os.path.join(input_path, file)
                df = gpd.read_parquet(input_filepath)

                qc = QCPipelines(df, file)
                qc.qc_lp()
