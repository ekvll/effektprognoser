import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from .logger import log_issue
from effektprognoser.paths import LOG_DIR

TRANSPORT_CATEGORIES = {"LL", "PB", "TT"}


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
                log_issue(log_msg + [log_comment])

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

    fz = 15  # Fontsize
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Region: {region}\nTabell: {table}\nID: {rid}")
    ax.plot(row["lp"], linewidth=1, color="black")
    ax.set_xlabel("Timma på året [h]")
    ax.set_ylabel("Effektbehov [MW]")
    fig.tight_layout()

    # Prepare output file path
    output_dir = os.path.join(LOG_DIR, "img", region)
    os.makedirs(output_dir, exist_ok=True)

    output_name = f"{region}_{rid}_{table}.png"
    output_filepath = os.path.join(output_dir, output_name)

    plt.savefig(output_filepath)
    plt.close(fig)


def substring_search(main_string: str, string_list: list[str]) -> bool:
    """
    Sub-string search.

    Args:
        main_string (str): The string to look-up.
        string_list (list[str]): List of strings of strings to check for.

    Returns:
        bool: True if main_string contain any string in string_list, False otherwise.
    """
    return any(substring in main_string for substring in string_list)


def make_np_array(obj: list | pd.Series | np.ndarray) -> np.ndarray | None:
    """
    Tries to convert an object into a numpy array.

    Args:
        obj (list | pd.Series | np.ndarray): Object to convert.

    Returns:
        np.ndarray | None: Numpy array if conversion is possible, None otherwise.
    """
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return None
