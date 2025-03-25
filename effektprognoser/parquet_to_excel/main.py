import os
from effektprognoser.paths import PARQUET_DIR_LOCAL
from ..geometry import load_kommuner
from ..sql_manager import get_years_in_table_names
from .data_processing import process_region
from .excel_utils import make_excel_table


def main(regions):
    kommuner = load_kommuner()

    for region in regions:
        input_path = os.path.join(PARQUET_DIR_LOCAL, region)
        files = os.listdir(input_path)
        years = get_years_in_table_names(files)

        result = process_region(region, input_path, files, years)

        for category in ["effektbehov", "elanvandning"]:
            for kommun, df in result.items():
                make_excel_table(df, kommun, category, region, kommuner)


if __name__ == "__main__":
    regions = ["06"]
    main(regions)
