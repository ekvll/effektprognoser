import os
import pandas as pd
from tqdm import tqdm
from ep.config import regions, raps_categories, default_years
from ep.cli.parquet2kommun import parquet_filenames, load_parquet
from ep.cli.sql2parquet import as_parquet


def print_merged_files(merged_files):
    for category, files in merged_files.items():
        print(f"\n{category}:")
        for file in files:
            print(file)


def merge_files_with_raps():
    result = {}
    for category, category_raps in raps_categories.items():
        result[category] = []

        for region in ["06", "07"]:
            filenames = parquet_filenames(region)

            for raps in category_raps:
                raps = raps.replace(" ", "_")
                for filename in filenames:
                    if raps in filename:
                        result[category].append(os.path.join(region, filename))
    return result


def collect_files(merged_files):
    for category, subpaths in merged_files.items():
        for year in default_years:
            dfs = []
            subpaths_filtered = [s for s in subpaths if str(year) in s]
            for subpath in subpaths_filtered:
                parts = subpath.split(os.sep)
                region = parts[0]
                filename = parts[1]
                df = load_parquet(filename, region)
                dfs.append(df)
            yield category, year, pd.concat(dfs, ignore_index=True)


def main():
    merged_files = merge_files_with_raps()
    print_merged_files(merged_files)

    for category, year, df in collect_files(merged_files):
        tqdm.write(f"{category} {year}")
        # print(f"Year: {year}")
        # print(df.shape)
        # print(df.columns)
        as_parquet(df, region="alla", table=f"{year}_{category}")


if __name__ == "__main__":
    main()
