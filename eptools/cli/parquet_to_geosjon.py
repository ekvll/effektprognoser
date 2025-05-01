import os
from tqdm import tqdm
from eptools.utils.paths import PARQUET_DIR
from eptools.utils.groups import category_groups
from eptools.processing.transform import get_category

"""
This script extracts the category from each parquet file. Then groups corresponding categories into groups: bostader, transport, total, and so on.

Thereafter this script performs processing on each group and saves the output as GeoJSON, ready for the webmap.
"""


def run(region):
    tqdm.write(f"Processing region {region}")
    path = os.path.join(PARQUET_DIR, region)
    files = os.listdir(path)

    for file in files:
        print(file)
        category = get_category(file)
        print(category)


if __name__ == "__main__":
    region = "10"
    run(region)
