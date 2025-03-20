import os
from effektprognoser.paths import SQL_DIR, DATA_DIR


def gen_db_path_local(region: str) -> str:
    db_filename = f"Effektmodell_{region}.sqlite"
    return os.path.join(DATA_DIR, "rut_id", region, db_filename)


def gen_db_path(region: str) -> str:
    region_dir = f"Effektmodell_{region}"
    db_name = region_dir + ".sqlite"
    return os.path.join(SQL_DIR, region_dir, db_name)
