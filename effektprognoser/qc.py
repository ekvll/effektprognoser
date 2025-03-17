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
        for _, row in self.df.iterrows():
            lp = make_np_array(row["lp"])

            if lp is not None:
                log_msg = [self.table, "lp", str(row.rid)]

                if self.tools.count_zero(lp):
                    log_issue(log_msg.append("Noll-värde i lastprofil"))

                boundaries = self.tools.boundary_values(lp)
                if boundaries:
                    log_issue(
                        log_msg.append(
                            f"Värde i lastprofil utanför gränsvärden {boundaries}"
                        )
                    )


class QCTools:
    def __init__(self, table):
        self.table = table

    def count_zero(self, arr: np.ndarray) -> bool:
        if np.any(np.isclose(arr, 0.0)):
            return True
        return False

    def boundary_values(self, arr: np.ndarray) -> bool:
        # Transport
        if substring_search(self.table, ["LL", "PB", "TT"]):
            min_val = 10.0
            max_val = 10e9
            if np.any(arr < min_val) or np.any(arr > max_val):
                return f"{min_val}, {max_val}"
        return False


def substring_search(main_string: str, string_list: list[str]):
    return any(substring in main_string for substring in string_list)


def make_np_array(obj):
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        return obj.to_numpy()
    if isinstance(obj, np.ndarray):
        return obj
    return None
