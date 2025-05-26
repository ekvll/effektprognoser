import numpy as np

from tqdm import tqdm

from ep.cli.sql2parquet import parquet_filenames, load_parquet
from ep.cli.parquet2kommun import (
    gen_df_from_defaults,
    get_year_and_raps,
    get_expected_length,
    gen_array_of_zeros,
    verify_array_length,
    make_excel_with_chart,
    gen_excel_filename,
    gen_excel_filepath,
)
from ep.config import default_raps, default_years

"""
This script processes parquet files for a specific region. It generates Excel files with charts for effektbehov and elanvändning based on the data in the parquet files.

It performs the following steps:
1. Load parquet filenames for the specified region.
3. Create output DataFrames for effektbehov and elanvändning.
4. For each parquet file, load the data.
5. Aggregate the load profiles and calculate the maximum and sum for effektbehov and elanvändning.
6. Generate Excel filenames and file paths based on the year.
7. Save the DataFrames as Excel files with charts.
"""


def main(region: str) -> None:
    tqdm.write(f"Processing region {region}")

    filenames = parquet_filenames(region)

    df_effektbehov = gen_df_from_defaults(default_years, default_raps)
    df_elanvandning = gen_df_from_defaults(default_years, default_raps)

    for filename in filenames:
        year, raps = get_year_and_raps(filename)

        gdf = load_parquet(filename, region)

        expected_length = get_expected_length(year)
        aggregated_lp = gen_array_of_zeros(expected_length)

        for _, lp in enumerate(gdf["lp"]):
            lp_array = np.asarray(lp)
            verify_array_length(lp_array, expected_length)
            aggregated_lp += lp_array

        df_effektbehov.loc[raps, year] = max(aggregated_lp)
        df_elanvandning.loc[raps, year] = sum(aggregated_lp)

    filename_effektbehov = gen_excel_filename(
        s1=region, s2="alla_kommuner", effektbehov=True
    )
    filename_elanvandning = gen_excel_filename(
        s1=region, s2="alla_kommuner", effektbehov=False
    )

    filepath_effektbehov = gen_excel_filepath(region, filename_effektbehov)
    filepath_elanvandning = gen_excel_filepath(region, filename_elanvandning)

    make_excel_with_chart(df_effektbehov, filepath_effektbehov, effektbehov=True)
    make_excel_with_chart(df_elanvandning, filepath_elanvandning, effektbehov=False)


if __name__ == "__main__":
    tqdm.write("ep.cli.parquet2region")

    from ep.config import regions
    # regions = ["10"]

    for region in regions:
        main(region)
