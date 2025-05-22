import numpy as np
import pandas as pd
import geopandas as gpd

from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from tqdm import tqdm

from ep.cli.sql2parquet import (
    get_kommuner_in_region,
    parquet_filenames,
    load_parquet,
)
from ep.config import default_raps, default_years, EXCEL_DIR


def save_df_as_excel(df, filename):
    df.to_excel(filename, sheet_name="Data")


def make_excel_with_chart(region, df, kommunnamn, kommunkod, effektbehov: bool) -> None:
    df_transposed = df.transpose()

    filename = f"{kommunkod}_{kommunnamn}"
    filename += "_effektbehov.xlsx" if effektbehov else "_elanvandning.xlsx"

    filepath = EXCEL_DIR / region / filename
    save_df_as_excel(df_transposed, filepath)

    wb = load_workbook(filepath)
    ws = wb["Data"]

    chart = LineChart()
    chart.title = "Effektbehov per år" if effektbehov else "Elanvändning per år"
    chart.style = 1
    chart.y_axis.title = "Effektbehov (MW)" if effektbehov else "Elanvändning (MWh)"
    chart.x_axis.title = "År"

    # Reference data (each column = one line)
    # A1: header row; A2:A5 = years; B1: series name
    min_row = 1  # start of years
    max_row = ws.max_row
    min_col = 1  # first data column (original row names)
    max_col = ws.max_column

    data = Reference(
        ws, min_col=min_col, max_col=max_col, min_row=min_row, max_row=max_row
    )
    categories = Reference(ws, min_col=1, max_col=1, min_row=min_row, max_row=max_row)

    chart.add_data(data, titles_from_data=True)
    chart.set_categories(categories)

    colors = [
        "FF0000",
        "00B050",
        "0070C0",
        "7030A0",
        "FFC000",
        "00FFFF",
        "FF00FF",
        "000000",
        "808080",
        "800000",
        "008080",
        "800080",
        "4682B4",
        "DAA520",
        "DC143C",
        "2E8B57",
        "A0522D",
        "5F9EA0",
        "D2691E",
        "9ACD32",
        "00008B",
        "B22222",
        "FF1493",
    ]

    for i, ser in enumerate(chart.series):
        ser.graphicalProperties.line.solidFill = colors[i % len(colors)]
    ws.add_chart(chart, "B10")
    wb.save(filepath)


def gen_array_of_zeros(length: int) -> np.ndarray:
    """Generate an array of zeros with the correct length based on the year."""
    return np.zeros(length)


def get_expected_length(year: int | str) -> int:
    """Get the expected length of the load profile based on the year."""
    if isinstance(year, int):
        year = str(year)
    return 8784 if year == "2040" else 8760


def verify_array_length(array: np.ndarray, expected_length: int) -> None:
    if array.shape[0] != expected_length:
        raise ValueError(
            f"Load profile has incorrect length: {array.shape[0]}, expected: {expected_length}"
        )


def get_year_and_raps(filename: str) -> tuple[str, str]:
    """Extract year and RAPS from the filename."""
    parts = filename.split("_")
    year = parts[1]
    raps = " ".join(parts[2:-1])
    return year, raps


def filter_df_by_kommun(
    gdf: pd.DataFrame | gpd.GeoDataFrame, kommun: str
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Filter the GeoDataFrame by the specified kommun."""
    if "kn" not in gdf.columns:
        raise ValueError("The GeoDataFrame does not contain a 'kn' column.")

    gdf_kommun = gdf[gdf["kn"] == kommun]

    if gdf_kommun.empty:
        # tqdm.write(f"No data found for kommun: {kommun}")
        return None

    return gdf_kommun


def main(region):
    tqdm.write(f"Processing region: {region}")

    filenames = parquet_filenames(region)
    kommuner = get_kommuner_in_region(filenames, region)

    tqdm.write(f"Found {len(kommuner['kommunnamn'])} kommuner in region {region}:")

    for kommun, kommunkod in zip(kommuner["kommunnamn"], kommuner["kommunkod"]):
        tqdm.write(f"Processing kommun: {kommun}")
        df_effektbehov = pd.DataFrame(columns=default_years, index=default_raps)
        df_elanvandning = pd.DataFrame(columns=default_years, index=default_raps)

        for filename in filenames:
            year, raps = get_year_and_raps(filename)

            gdf = load_parquet(filename, region)
            gdf_kommun = filter_df_by_kommun(gdf, kommun)

            if gdf_kommun is None:
                continue  # Skip if no data for the kommun

            expected_length = get_expected_length(year)
            aggregated_lp = gen_array_of_zeros(expected_length)

            for _, lp in enumerate(gdf_kommun["lp"]):
                lp_array = np.asarray(lp)
                verify_array_length(lp_array, expected_length)
                aggregated_lp += lp_array

            df_effektbehov.loc[raps, year] = max(aggregated_lp)
            df_elanvandning.loc[raps, year] = sum(aggregated_lp)

        make_excel_with_chart(
            region, df_effektbehov, kommun, kommunkod, effektbehov=True
        )

        make_excel_with_chart(
            region, df_elanvandning, kommun, kommunkod, effektbehov=False
        )


if __name__ == "__main__":
    # from ep.config import regions
    regions = ["10"]
    for region in regions:
        main(region)
