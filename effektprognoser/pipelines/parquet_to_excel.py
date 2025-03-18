import os

import pandas as pd
import numpy as np
import geopandas as gpd
from effektprognoser.geometry import load_kommuner
from effektprognoser.paths import PARQUET_DIR, EXCEL_DIR
from effektprognoser.sqlite import get_years_in_table_names, filter_tables


def main(regions):
    kommuner = load_kommuner()

    for region_index, region in enumerate(regions):
        input_path = os.path.join(PARQUET_DIR, region)
        files = os.listdir(input_path)
        years = get_years_in_table_names(files)

        result = {}
        print("Processing year:")
        for year in years:
            print(year)
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

        for category in ["effektbehov", "elanvandning"]:
            for kommun, df in result.items():
                kommunkod = str(df[0]["kommunkod"])
                if kommunkod[0] == "6":
                    kommunkod = "06"
                elif kommunkod[0] == "7":
                    kommunkod = "07"
                elif kommunkod[0] == "8":
                    kommunkod = "08"

                if kommunkod[:2] != str(region):
                    print(f"Kommunkod '{kommunkod}' not equal to region '{region}'")
                    continue

                df_kommun = pd.DataFrame()
                data_year = []
                raps = []
                value = []
                for row in df:
                    data_year.append(row["year"])
                    raps.append(row["out_fn"])
                    value.append(row[category])
                df_kommun["year"] = data_year
                df_kommun["raps"] = raps
                df_kommun[category] = value

                df_out = pd.DataFrame(
                    index=df_kommun.raps.unique(), columns=df_kommun.year.unique()
                )

                for unique_year in df_kommun.year.unique():
                    df_year = df_kommun[df_kommun.year == unique_year]
                    df_year = df_year.drop(columns=["year"], axis=1).reset_index(
                        drop=True
                    )
                    for idx, row_year in df_year.iterrows():
                        df_out.loc[row_year["raps"], unique_year] = row_year[category]

                df_out = df_out.sort_index()

                make_excel_table(df_out, kommun, category, region, kommuner)


def make_excel_table(df_in, kommun, category, region, kommuner):
    # kommuner = load_kommuner()

    df_kommun = kommuner[["kn", "kk"]]
    kommunkod = get_kommunkod_by_kommunnamn(df_kommun, kommun)

    df = pd.DataFrame(
        index=[
            "Flerbostadshus",
            "Smahus",
            "LOM_Flerbostadshus",
            "LOM_Smahus",
            "RAPS_1_3",
            "RAPS_4",
            "RAPS_5",
            "RAPS_6",
            "RAPS_7",
            "RAPS_8",
            "RAPS_9",
            "RAPS_10",
            "RAPS_11",
            "RAPS_12",
            "RAPS_13",
            "RAPS_14",
            "RAPS_15",
            "RAPS_16",
            "RAPS_17",
            "RAPS_18",
            "RAPS_19",
            "RAPS_20",
            "RAPS_21",
            "RAPS_22",
            "RAPS_23",
            "RAPS_24",
            "RAPS_27",
            "RAPS_7777",
            "RAPS_8888",
            "PB",
            "LL",
            "TT_DEP",
            "TT_DEST",
            "TT_RESTSTOP",
        ],
        columns=["2022", "2027", "2030", "2040"],
    )

    df = merge_dataframes(df_left=df, df_right=df_in)
    df = drop_column(df)

    # save_path = f"data/RSS/output/{region}/xlsx/"
    save_path = os.path.join(EXCEL_DIR, region)
    # save_path = rf"D:\python\effektprognoser\data\RSS\output\{region}\xlsx/"
    os.makedirs(save_path, exist_ok=True)

    if category == "effektbehov":
        category_name = "Effektbehov (MW)"

    elif category == "elanvandning":
        category_name = "Elanvändning (MWh)"

    writer = pd.ExcelWriter(
        os.path.join(save_path, f"{kommunkod} - {kommun} - {category_name}.xlsx"),
        engine="xlsxwriter",
    )
    df.to_excel(writer, sheet_name="Sheet1")
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]

    chart = workbook.add_chart({"type": "line"})

    for i in range(len(df)):
        chart.add_series(
            {
                "name": [f"Sheet1", i + 1, 0],  # Row label from column A
                "categories": [
                    f"Sheet1",
                    0,
                    1,
                    0,
                    len(df.columns),
                ],  # Categories (Years)
                "values": [
                    f"Sheet1",
                    i + 1,
                    1,
                    i + 1,
                    len(df.columns),
                ],  # Values from the row
                "marker": {
                    "type": "circle",
                    "size": 6,
                },  # Adds circular markers to the line
            }
        )

    # Add x-axis and y-axis labels
    chart.set_x_axis(
        {
            "name": "År",  # X-axis label
            "name_font": {"size": 12, "bold": True},  # Font size and style
            "num_font": {"italic": True},  # Format for x-axis tick labels (optional)
        }
    )
    chart.set_y_axis(
        {
            "name": category_name,  # Y-axis label
            "name_font": {"size": 12, "bold": True},  # Font size and style
            "num_format": "0.00",  # Format for y-axis values (optional)
        }
    )

    # Optional: Set a chart title
    chart.set_title({"name": f"{kommun} kommun - {category_name}"})

    # Adjust the chart size
    chart.set_size({"width": 600, "height": 550})  # Example size

    worksheet.insert_chart("G2", chart)
    writer._save()


def drop_column(df: pd.DataFrame, drop_col: str = "_left") -> pd.DataFrame:
    """
    Drops columns from a DataFrame that contain the substring '_base'.

    Parameters:
    df (pd.DataFrame): The DataFrame to remove columns from.

    Returns:
    pd.DataFrame: The input DataFrame with columns containing '_base' removed.
    """
    for col in df.columns:
        if drop_col in col:
            df.drop(columns=col, inplace=True)
    return df


def merge_dataframes(
    df_left: pd.DataFrame, df_right: pd.DataFrame, suffixes: tuple = ("_left", "")
) -> pd.DataFrame:
    """
    Merges two DataFrames using a left join on their indices, ensuring all rows from the left DataFrame
    are preserved. Columns with the same name will be suffixed with '_left' for the left DataFrame
    and left unchanged for the right DataFrame. Also returns the rows from df_right that were dropped
    for not having a matching index in df_left.

    Parameters:
    df_left (pd.DataFrame): The left DataFrame, whose rows will be fully preserved in the output.
    df_right (pd.DataFrame): The right DataFrame to merge into the left, using the index as the key.

    Returns:
    tuple:
        - pd.DataFrame: A merged DataFrame where all rows from df_left are retained. Columns from df_left
          that overlap with df_right will be suffixed with '_left' to prevent name collisions.
        - pd.Index: An index of rows that were present in df_right but were dropped due to no match in df_left.
    """
    # Perform a left merge on indices, preserving all rows from df_left
    # If there are common column names, add '_left' suffix to columns from df_left
    merged_df = pd.merge(
        df_left,
        df_right,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=suffixes,
    )

    # Identify indices in df_right that were not matched in df_left
    dropped_rows = df_right.index.difference(df_left.index)

    if len(dropped_rows) > 0:
        print(f"Dropped rows: {dropped_rows}")

    return merged_df


def get_kommunkod_by_kommunnamn(df, kommunnamn):
    result = df.loc[df["kn"] == kommunnamn, "kk"]
    if not result.empty:
        return result.iloc[0]
    return None


if __name__ == "__main__":
    regions = "06"
    main(regions)
