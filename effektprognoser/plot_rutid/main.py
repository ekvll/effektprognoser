import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import effektprognoser as ep
from effektprognoser import sql_manager

try:
    from ..paths import PARQUET_DIR_LOCAL, DATA_DIR
except Exception:
    from effektprognoser.paths import PARQUET_DIR_LOCAL, DATA_DIR


def make_output_dir(region: str):
    output_dir = os.path.join(DATA_DIR, "plot_rutid", region)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sort_dict(data):
    return dict(sorted(data.items(), key=lambda item: np.max(item[1]), reverse=True))


def get_nrows(indata):
    ncols = []
    for _, data in indata.items():
        ncols.append(len(data))
    if len(set(ncols)) == 1:
        ncols = list(set(ncols))[0]
        return ncols
    else:
        return max(ncols)
        # raise ValueError("Not equal length")


def plot_dict(indata, region, rutid):
    # Get figure number of rows and columns
    nrows = get_nrows(indata)
    ncols = len(indata)

    # Initiate figure
    fig, ax = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(18, 3 * nrows), sharex=True, sharey=True
    )

    # Iterate over columns in figure
    for c, (year, data) in enumerate(indata.items()):
        data_sorted = sort_dict(data)

        # Iterate over rows in figure
        for r, (raps, lp) in enumerate(data_sorted.items()):
            # Set ax title
            if r == 0:
                ax[r, c].set_title(f"{year}\n{raps}")
            else:
                ax[r, c].set_title(raps)

            # Set y label (only for ax in column 0 (first column))
            if c == 0:
                ax[r, c].set_ylabel("Effektbehov [MW]")

            # Set x label (only on ax object in the last row)
            if r == nrows - 1:
                ax[r, c].set_xlabel("Timma på året")

            ax[r, c].plot(np.arange(len(lp)), lp, linewidth=1)
            ax[r, c].spines["top"].set_visible(False)
            ax[r, c].spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir = make_output_dir(region)
    output_name = f"{region}_{rutid}.png"

    plt.savefig(os.path.join(output_dir, output_name))
    plt.close(fig)


def main(region, rutid):
    # Verify object types
    if not isinstance(rutid, np.uint64):
        rutid = np.uint64(rutid)
    if not isinstance(region, str):
        region = str(region)

    # Get list of Parquet files associated with Region
    input_path = os.path.join(PARQUET_DIR_LOCAL, region)
    files = os.listdir(input_path)

    data = {}
    for file in files:
        input_filepath = os.path.join(input_path, file)
        gdf = gpd.read_parquet(input_filepath)

        if rutid in gdf.rid.values:
            gdf_unique = gdf.loc[gdf.rid == rutid]
            # print(gdf_unique)

            year = file.split("_")[1]
            raps = "_".join(file.split("_")[2:]).split("_V1")[0]
            # raps = "_".join(raps)
            if year not in data:
                data[year] = {}

            if raps not in data[year]:
                data[year][raps] = gdf_unique.lp.to_numpy()[0]
    plot_dict(data, region, rutid)


if __name__ == "__main__":
    main("10", "5170006228000")
