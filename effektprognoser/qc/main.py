import os
import geopandas as gpd
from effektprognoser.paths import PARQUET_DIR_LOCAL
from ..sqlite import get_years_in_table_names, filter_tables
from .qc_pipelines import QCPipelines


def main(regions):
    for region_index, region in enumerate(regions):
        print(f"Quality check region {region}")

        # Path to look for Parquet files
        input_path = os.path.join(PARQUET_DIR_LOCAL, region)

        # List files in path
        files = os.listdir(input_path)

        # Extract years from filenames
        years = get_years_in_table_names(files)

        for year in years:
            print(f"Year: {year}")
            files_filtered = filter_tables(files, year)

            for file in files_filtered:
                input_filepath = os.path.join(input_path, file)
                df = gpd.read_parquet(input_filepath)

                qc = QCPipelines(df, file, region)
                qc.qc_lp()
