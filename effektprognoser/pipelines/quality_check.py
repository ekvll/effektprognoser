import os
import geopandas as gpd
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from effektprognoser.paths import PARQUET_DIR_LOCAL, LOG_DIR, DATA_DIR
from ..sqlite import (
    get_years_in_table_names,
    filter_tables,
)
from ..utils import make_np_array

TRANSPORT_CATEGORIES = {"LL", "PB", "TT"}


logging.basicConfig(
    filename=os.path.join(LOG_DIR, "log_quality_check.csv"),
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s, %(message)s",
)


def log_issue(log_msg: list[str]) -> None:
    """
    Append log file with log messages (issues).
    Log filepath is os.path.join(LOG_DIR, "qc_log.csv").

    Args:
        log_msg (list[str]): List of strings. The length of the list object need to be 4. Each entry in the list object correspond to a specific column in the log file.

    Returns:
        None
    """
    # Verify that the log message list is of length 4
    if not len(log_msg) == 4:
        raise ValueError(
            f"log_msg need to contain 'table', 'column', 'rid' and 'comment' ({log_msg})"
        )

    # Make the list of strings into a single string object joined by ','
    str_obj = ", ".join(log_msg)

    # Add the log message to the log file
    logging.info(str_obj)


class QCPipelines:
    def __init__(self, df: pd.DataFrame, table: str, region: str) -> None:
        self.df = df
        self.table = table.removesuffix(".parquet")
        self.tools = QCTools(self.table)
        self.region = region

    def qc_lp(self) -> None:
        """
        Perform quality check on 'lastprofil' (lp).

        Args:
            None

        Returns:
            None
        """
        # Set keep track of plots already created
        png_set = set()

        # Iterate over rows in dataframe
        for _, row in self.df.iterrows():
            lp = make_np_array(row["lp"])
            if lp is None:
                continue

            rid = str(row.rid)
            log_msg = [self.table, "lp", rid]
            log_comment = []

            # Check if any value in 'lastprofil' is zero
            if self.tools.count_zero(lp):
                log_comment.append("Noll-värde i lastprofil")

            # Check if any value in 'lastprofil' is outside given threshold boundaries
            boundary = self.tools.boundary_values(lp)
            if boundary:
                log_comment.append(f"Ett värde i lastprofil {boundary} (gränsvärde)")

            # Log any issues found
            if log_comment:
                log_issue(log_msg + log_comment)

                # Plot if it hasn't been plotted already
                if (rid, self.table) not in png_set:
                    plot_row(row, self.table, self.region)
                    png_set.add((rid, self.table))


class QCTools:
    def __init__(self, table) -> None:
        self.table = table

    def count_zero(self, arr: np.ndarray) -> bool:
        """
        Check if any value in an array is zero, unless the table is a transport category.

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            bool: True is zeros in array, False otherwise.
        """
        if self.is_transport_category():
            return False
        return np.any(np.isclose(arr, 0.0))

    def boundary_values(self, arr: np.ndarray) -> str | bool:
        """
        Check if any value in array is outside given threshold boundaries.

        Args:
            arr (np.ndarray): Array to check.

        Returns:
            str | bool: String of exceeded boundary, or None if within boundaries.

        """
        min_val, max_val = self.get_boundary_values()
        if min_val is not None and np.any(arr < min_val):
            return f"< {min_val}"
        if max_val is not None and np.any(arr > max_val):
            return f"> {max_val}"
        return None

    def is_transport_category(self) -> bool:
        """
        Check if the table belongs to category.

        Args:
            None

        Returns:
            bool: True is table belongs to transport categories, False otherwise.
        """
        return any(category in self.table for category in TRANSPORT_CATEGORIES)

    def get_boundary_values(self) -> tuple[int | None, int | None]:
        """
        Define boundary values based on table

        Args:
            None

        Returns:
            tuple: (int, int) if table belongs to a category, (None, None) otherwise.

        """
        if self.is_transport_category():
            return 0, 10**10
        return None, None  # Default boundaries


def plot_row(row, table: str, region: str) -> None:
    """
    Plot data contained in a Pandas dataframe row:

    Args:
        row (): The data row.
        table (str): Name of the table being quality checked.
        region (str): Region number of the region being quality checked.

    Returns:
        None
    """
    rid = str(row.rid)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Region: {region}\nTabell: {table}\nID: {rid}")
    ax.plot(row["lp"], linewidth=1, color="black")
    ax.set_xlabel("Timma på året [h]")
    ax.set_ylabel("Effektbehov [MW]")
    fig.tight_layout()

    # Prepare output file path
    output_dir = os.path.join(DATA_DIR, "quality_check", region)
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{region}_{rid}_{table}.png"
    output_filepath = os.path.join(output_dir, output_name)

    plt.savefig(output_filepath)
    plt.close(fig)


def main(regions):
    for region_index, region in enumerate(regions):
        print(f"Quality check region {region}")

        # Path to look for Parquet files
        input_path = os.path.join(PARQUET_DIR_LOCAL, region)

        # List files in path
        files = os.listdir(input_path)

        # Extract years from filenames
        years = get_years_in_table_names(files)

        for year in years:
            print(f"Year: {year}")
            files_filtered = filter_tables(files, year)

            for file in files_filtered:
                input_filepath = os.path.join(input_path, file)
                df = gpd.read_parquet(input_filepath)

                qc = QCPipelines(df, file, region)
                qc.qc_lp()
