from .database import get_db_path, connect_to_db, filter_tables_by_rutid
from .data_processing import extract_data
from .visualization import make_plot, make_plot_year
from effektprognoser.sqlite import get_table_names_in_db


def main(region, rutid):
    print(f"Processing region {region}")
    print(f"Processing RutID {rutid}")
    db_path = get_db_path(region)
    conn, cursor = connect_to_db(db_path)

    tables = get_table_names_in_db(cursor)
    tables_rutid = filter_tables_by_rutid(cursor, tables, rutid)

    for year, sorted_data in extract_data(cursor, conn, tables_rutid, rutid):
        if year:
            make_plot(sorted_data, region, rutid, year)
        else:
            make_plot_year(region, rutid, sorted_data)

    conn.close()
