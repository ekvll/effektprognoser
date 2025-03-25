import pandas as pd
import numpy as np
import os
from effektprognoser.utils import get_metadata
from effektprognoser.paths import PARQUET_DIR_LOCAL, DATA_DIR


def filter_files(files: list[str], raps: str) -> list[str]:
    files_filtered = [f for f in files if raps in f]

    if "Flerbostadshus" in raps and "LOM" not in raps:
        return [f for f in files_filtered if "LOM" not in f]

    if "Smahus" in raps and "LOM" not in raps:
        return [f for f in files_filtered if "LOM" not in f]

    return files_filtered


def process_files(raps_list: list[str], years: list[str], files: list[str]) -> dict:
    """ """
    result = {}

    # Iterate over all RAPS
    for raps in raps_list:
        # Get only files assoicated with RAPS
        files_filtered = filter_files(files, raps)

        # Quality check
        if len(files_filtered) > 4:
            raise ValueError("Length of 'files_filtered' should not exceed 4")

        # Stored each RAPS and all files that contain that RAPS in a dict
        result[raps] = files_filtered

    return result


def load_data(files_filtered: dict, input_path: str):
    data = {}

    for raps, files in files_filtered.items():
        if raps not in data:
            data[raps] = {}

        for file in files:
            year = file.split("_")[1]

            input_filepath = os.path.join(input_path, file)
            df = pd.read_parquet(input_filepath)["lp"]
            aggregated_lp = np.zeros(8784 if "2040" in file else 8760)
            for lastprofil in df:
                aggregated_lp += lastprofil
            effektbehov = float(max(aggregated_lp))
            elanvandning = float(sum(aggregated_lp))

            data[raps][year] = {
                "effektbehov": effektbehov,
                "elanvandning": elanvandning,
            }
    return data


def dict_to_dataframe(data: dict) -> pd.DataFrame:
    records = []
    for raps, years in data.items():
        for year, values in years.items():
            records.append(
                {
                    "raps": raps,
                    "year": int(year),
                    "effektbehov": values["effektbehov"],
                    "elanvandning": values["elanvandning"],
                }
            )
    return pd.DataFrame(records)


def save_dict_to_parquet(data: dict, output_path: str) -> None:
    df = dict_to_dataframe(data)
    df.to_parquet(output_path, index=False)


def main(region):
    raps_list, years = get_metadata()
    input_path = os.path.join(PARQUET_DIR_LOCAL, region)
    files = os.listdir(input_path)

    files_filtered = process_files(raps_list, years, files)
    data = load_data(files_filtered, input_path)
    output_path = os.path.join(DATA_DIR, "tmp", f"{region}.parquet")
    save_dict_to_parquet(data, output_path)


if __name__ == "__main__":
    region = "10"
    main(region)
