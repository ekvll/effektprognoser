from ..sql_manager import (
    gen_db_path,
    connect_to_db,
    close_connection,
    get_table_names_in_db,
    get_years_in_table_names,
    filter_tables,
    get_table_column_names,
    load_table_in_chunks,
)


def fetch_data_for_region(region):
    db_path = gen_db_path(region)
    conn, cursor = connect_to_db(db_path)

    tables = get_table_names_in_db(cursor)
    years = get_years_in_table_names(tables)

    return conn, cursor, tables, years


def close_db_connection(conn, cursor):
    close_connection(conn, cursor)
