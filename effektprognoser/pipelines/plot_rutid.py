import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from effektprognoser.paths import DATA_DIR
from effektprognoser.sqlite import (
    get_table_names_in_db,
    get_years_in_table_names,
    filter_tables,
)
from effektprognoser.utils import sort_dict


def get_db_path(region):
    db_filename = f"Effektmodell_{region}.sqlite"
    return os.path.join(DATA_DIR, "rut_id", region, db_filename)


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return conn, cursor


def row_exists(cursor, table_name, row_id):
    cursor.execute(
        f"SELECT EXISTS(SELECT 1 FROM {table_name} WHERE rut_id = ? LIMIT 1);",
        (row_id,),
    )
    if cursor.fetchone()[0] == 1:
        return table_name
    else:
        # print(f"{row_id} not in {table_name}")
        return None


def filter_tables_by_rutid(cursor, tables, rutid):
    tables_rutid = []
    for table in tables:
        exists = row_exists(cursor, table, rutid)
        if exists:
            tables_rutid.append(exists)
    return tables_rutid


def make_plot(data, region, rutid, year):
    fz = 15
    fig, ax = plt.subplots(
        nrows=len(data) + 1, figsize=(18, 5 * len(data)), sharex=True, sharey=True
    )
    ax[0].set_title(
        f"Region: {region} - ID: {rutid} - År: {year}\nAlla data", fontsize=fz
    )

    for i, (table, ser) in enumerate(data.items()):
        label_ = table.split(f"EF_{year}_")[-1].split("_V1")[0].replace("_", " ")
        ax[0].plot(ser, "-", linewidth=1, label=label_)
        ax[i + 1].set_title(label_, fontsize=fz)
        ax[i + 1].plot(ser, "-", color="black", linewidth=1)

    for axe in ax:
        axe.set_ylabel("Effektbehov [MW]", fontsize=fz)
        axe.tick_params(axis="both", labelsize=fz)
    ax[-1].set_xlabel("Timma på året [h]", fontsize=fz)
    ax[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=fz, title="År")

    fig.tight_layout()
    output_name = f"{region}_{rutid}_{year}.png"
    plt.savefig(os.path.join(DATA_DIR, "rut_id", region, output_name))
    plt.close(fig)


def make_plot_year(region, rutid, data_dict):
    color = ["red", "blue", "green", "magenta"]
    for table, data in data_dict.items():
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(12, 7))
        ax[0].set_title(f"Region: {region} - ID:{rutid}\n{table}")
        ax[1].set_title("Delta-kurvor")
        for i, (year, ser) in enumerate(data.items()):
            ax[0].plot(ser, linewidth=1, color=color[i], label=year, zorder=4 - i)
            if i == 0:
                ref_ser = ser
                ref_year = year
            else:
                ax[1].plot(
                    ser - ref_ser,
                    linewidth=1,
                    color=color[i],
                    label=f"{year} - {ref_year}",
                    zorder=4 - i,
                )
        ax[1].set_xlabel("Timma på året [h]")
        for axe in ax:
            axe.set_ylabel("Effektbehov [MW]")
            axe.legend(loc="upper left", bbox_to_anchor=(1, 1), title="År")
        output_name = f"{region}_{rutid}_{table}.png"
        fig.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "rut_id", region, output_name))


def main(region, rutid):
    print(f"Processing region {region}")
    print(f"Processing RutID {rutid}")
    db_path = get_db_path(region)
    conn, cursor = connect_to_db(db_path)
    tables = get_table_names_in_db(cursor)
    tables_rutid = filter_tables_by_rutid(cursor, tables, rutid)
    years = get_years_in_table_names(tables_rutid)

    data_year = {}
    for year in years:
        data = {}
        tables_filtered = filter_tables(tables_rutid, year)

        for table in tables_filtered:
            elanvandning: pd.Series = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE rut_id = {rutid};", conn
            )["Elanvandning"].reset_index(drop=True)

            if len(elanvandning) > 0:
                data[table] = elanvandning
                table_mod = (
                    table.split(f"EF_{year}_")[-1].split("_V1")[0].replace("_", " ")
                )
                if table_mod not in data_year:
                    data_year[table_mod] = {}
                data_year[table_mod][year] = elanvandning

        sorted_data = sort_dict(data)
        make_plot(sorted_data, region, rutid, year)
    make_plot_year(region, rutid, data_year)
