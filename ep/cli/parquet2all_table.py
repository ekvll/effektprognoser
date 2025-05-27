import pandas as pd
from tqdm import tqdm
from ep.config import regions
from ep.cli.parquet2kommun import parquet_filenames, load_parquet
from ep.cli.sql2parquet import as_parquet
from ep.cli.parquet2all import drop_duplicates_keep_highest


def get_unique_filenames() -> list[str]:
    all_filenames = []
    for region in regions:
        filenames = parquet_filenames(region)
        all_filenames.extend(filenames)

    return sorted(list(set(all_filenames)))


def merge_data(filenames: list[str]):
    for filename in filenames:
        tqdm.write(f"Processing file: {filename}")
        dfs = []
        for region in regions:
            region_filenames = parquet_filenames(region)
            if filename in region_filenames:
                df = load_parquet(filename, region)
                if not df.empty:
                    dfs.append(df)

        yield pd.concat(dfs, ignore_index=True), filename


def main() -> None:
    all_filenames = get_unique_filenames()
    for df, filename in merge_data(all_filenames):
        df = drop_duplicates_keep_highest(df, id_col="rid", value_col="eb")
        as_parquet(df, "alla", filename.replace(".parquet", ""))


if __name__ == "__main__":
    main()
