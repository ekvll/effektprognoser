import os
import pickle
import sys
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib import cm

"""
Organize this code. But the code into functions, etc.
Also add to CLI.
"""

"""
Write a docstring of what this script does.
"""


region = "13"
path = f"../data/pkl/{region}/"
if os.path.exists(path):
    files = os.listdir(path)
    print(files)
else:
    print(f"Path {path} does not exist.")
    exit()


with open(path + f"{region}_max_time", "rb") as f:
    time = pickle.load(f)

with open(path + f"{region}_lp_kommuner", "rb") as f:
    lp = pickle.load(f)


def get_time_data(time_dict, kommun, year):
    filtered_times = {
        " ".join(k.split("_")[2:-1]): [int(v[0]), float(v[1]), v[2]]
        for k, v in time_dict[kommun].items()
        if year in k
    }
    return filtered_times


def get_max_and_index(loadprofile):
    max_value = np.max(loadprofile)
    index = np.argmax(loadprofile)
    return max_value, index


# Your custom key order
custom_order = [
    "Flerbostadshus",
    "Smahus",
    "LOM Flerbostadshus",
    "LOM Smahus",
    "RAPS 1 3",
    "RAPS 4",
    "RAPS 5",
    "RAPS 6",
    "RAPS 7",
    "RAPS 8",
    "RAPS 9",
    "RAPS 10",
    "RAPS 11",
    "RAPS 12",
    "RAPS 13",
    "RAPS 14",
    "RAPS 15",
    "RAPS 16",
    "RAPS 17",
    "RAPS 18",
    "RAPS 19",
    "RAPS 20",
    "RAPS 21",
    "RAPS 22",
    "RAPS 23",
    "RAPS 24",
    "RAPS 27",
    "RAPS 7777",
    "RAPS 8888",
    "PB",
    "LL",
    "TT DEP",
    "TT DEST",
    "TT RESTSTOP",
]

for kommun, _ in lp.items():
    rows = {}
    df_year = pd.DataFrame()
    c1 = []
    c2 = []
    c3 = []
    for year, loadprofile in _.items():
        if year not in rows:
            rows[year] = []
        xrange = range(len(loadprofile))
        filtered_times = get_time_data(time, kommun, year)
        max_value, index = get_max_and_index(loadprofile)
        # rows.append({"År": year, "MaxTidpunkt": index, "MaxTidpunktEB": max_value})
        d1 = []
        d2 = []
        c1.append(max_value)
        c2.append(index)
        c3.append(year)
        for k, v in filtered_times.items():
            d1.append(k)
            d2.append(v[2][index])
        rows[year] = d1
        df_tmp = pd.DataFrame(d2, index=d1, columns=[year])

        if df_year.empty:
            df_year = df_tmp
        else:
            df_year = pd.merge(
                df_year, df_tmp, left_index=True, right_index=True, how="outer"
            )

    df_year_max = pd.DataFrame({"Tidpunkt": c2, "Max": c1}, index=c3).T

    df_year = df_year.reindex(custom_order)
    df_year.replace(0, np.nan, inplace=True)

    # Save to the same sheet
    out_path = f"../data/excel/{region}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with pd.ExcelWriter(
        os.path.join(out_path, f"{kommun} - Maxeffekt.xlsx"), engine="xlsxwriter"
    ) as writer:
        df_year_max.to_excel(writer, sheet_name="Combined")

        # Leave 2 empty rows between the tables for clarity
        start_row = len(df_year_max) + 3

        df_year.to_excel(writer, sheet_name="Combined", startrow=start_row)
