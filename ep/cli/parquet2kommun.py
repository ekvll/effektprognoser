import numpy as np
import pandas as pd

from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.utils import get_column_letter
from openpyxl.chart.series import SeriesLabel
from openpyxl.drawing.colors import ColorChoice
from openpyxl.chart.series import Series
from tqdm import tqdm

from ep.cli.sql2parquet import parquet_filenames, as_parquet
from ep.cli.sql2parquet import (
    get_kommuner_in_region,
    load_parquet,
    calc_total_loadprofile_per_kommun,
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
            year = filename.split("_")[1]
            raps = " ".join(filename.split("_")[2:-1])

            gdf = load_parquet(filename, region)
            gdf_kommun = gdf.loc[gdf["kn"] == kommun]

            if gdf_kommun.empty:
                # tqdm.write(f"{kommun}, in {year}, does not contain {raps}.")
                continue

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

            df_effektbehov.loc[raps, year] = max(aggregated_lp)
            df_elanvandning.loc[raps, year] = sum(aggregated_lp)
        make_excel_with_chart(
            region, df_effektbehov, kommun, kommunkod, effektbehov=True
        )
        make_excel_with_chart(
            region, df_elanvandning, kommun, kommunkod, effektbehov=False
        )
    # as_parquet(gdf, region, f"{kommun_kod}_{kommun}", "_parquet2kommun_effektbehov")


if __name__ == "__main__":
    # from ep.config import regions
    regions = ["10"]
    for region in regions:
        main(region)
