import pandas as pd
import geopandas as gpd
import numpy as np
from effektprognoser.logger import log_issue


class QCPipelines:
    def __init__(self, df, table):
        self.df = df
        if table.endswith(".parquet"):
            table = table.split(".parquet")[0]
        self.table = table
        self.tools = QCTools(self.table)

    def qc_lp(self):
        """
        Perform quality check on 'lastprofil' (lp).
        """

        # Iterate over rows in dataframe
        for _, row in self.df.iterrows():
            # Make sure 'lastprofil' is a numpy array
            lp = make_np_array(row["lp"])

            if lp is not None:
                # Define the "base" of the log message
                log_msg = [self.table, "lp", str(row.rid)]

                # Perform quality checks
                # Check if any value in 'lastprofil' is zero
                if self.tools.count_zero(lp):
                    # If quality check is True, thus if any value is zero
                    # Update the log message
                    log_comment = "Noll-värde i lastprofil"  # Update the log message
                    log_issue(log_msg + [log_comment])

                # Check if any value in 'lastprofil' is outside given threshold boundaries
                boundaries = self.tools.boundary_values(lp)
                if boundaries:
                    # If quality check is True, thus if any value is outside threshold boundaries
                    # Update the log message
                    log_comment = f"Värde i lastprofil utanför gränsvärde {boundaries}"
                    log_issue(log_msg + [log_comment])


class QCTools:
    def __init__(self, table):
        self.table = table

    def count_zero(self, arr: np.ndarray) -> bool:
        """
        Check if any value in an array is zero.

        Args:
            arr (np.ndarray): Numpy array to check for zeroes.

        Returns:
            bool: True if zero in array. False otherwise.

        """
        if np.any(np.isclose(arr, 0.0)):
            return True
        return False

    def boundary_values(self, arr: np.ndarray) -> str | False:
        """
        Check if any value in arr is outside given threshold boundaries.
        """
        # Bostäder

        # Industri och bygg

        # Offentlig och privat tjänstesektor

        # Transport
        if substring_search(self.table, ["LL", "PB", "TT"]):
            min_val = 10.0
            max_val = 10**10
            if np.any(arr < min_val) | np.any(arr > max_val):
                return f"{min_val} {max_val}"

        return False  # Return False if nothing else


def substring_search(main_string: str, string_list: list[str]):
    return any(substring in main_string for substring in string_list)


def make_np_array(obj):
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return None
