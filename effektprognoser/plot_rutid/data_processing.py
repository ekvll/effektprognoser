import pandas as pd
from effektprognoser.sqlite import (
    get_table_names_in_db,
    get_years_in_table_names,
    filter_tables,
)
from effektprognoser.utils import sort_dict


def extract_data(cursor, conn, tables_rutid, rutid):
    years = get_years_in_table_names(tables_rutid)
    data_year = {}

    for year in years:
        data = {}
        tables_filtered = filter_tables(tables_rutid, year)

        for table in tables_filtered:
            elanvandning: pd.Series = pd.read_sql_query(
                f"SELECT * FROM {table} WHERE rut_id = {rutid};", conn
            )["Elanvandning"].reset_index(drop=True)

            if not elanvandning.empty:
                data[table] = elanvandning
                table_mod = (
                    table.split(f"EF_{year}_")[-1].split("_V1")[0].replace("_", " ")
                )
                data_year.setdefault(table_mod, {})[year] = elanvandning

        sorted_data = sort_dict(data)
        yield year, sorted_data

    yield None, data_year
