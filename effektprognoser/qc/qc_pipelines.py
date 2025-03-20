import pandas as pd
from .qc_tools import QCTools
from .plot_utils import plot_row
from .logging_utils import log_issue
from ..utils import make_np_array


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
