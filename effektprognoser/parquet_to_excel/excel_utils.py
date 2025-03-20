import os
import pandas as pd
from effektprognoser.paths import EXCEL_DIR
from .utils import merge_dataframes, drop_column, get_kommunkod_by_kommunnamn


def make_excel_table(df_in, kommun, category, region, kommuner):
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

    save_path = os.path.join(EXCEL_DIR, region)
    os.makedirs(save_path, exist_ok=True)

    category_name = (
        "Effektbehov (MW)" if category == "effektbehov" else "Elanvändning (MWh)"
    )

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
                "name": [f"Sheet1", i + 1, 0],
                "categories": ["Sheet1", 0, 1, 0, len(df.columns)],
                "values": ["Sheet1", i + 1, 1, i + 1, len(df.columns)],
                "marker": {"type": "circle", "size": 6},
            }
        )

    chart.set_x_axis({"name": "År", "name_font": {"size": 12, "bold": True}})
    chart.set_y_axis(
        {
            "name": category_name,
            "name_font": {"size": 12, "bold": True},
            "num_format": "0.00",
        }
    )
    chart.set_title({"name": f"{kommun} kommun - {category_name}"})
    chart.set_size({"width": 600, "height": 550})

    worksheet.insert_chart("G2", chart)
    writer._save()
