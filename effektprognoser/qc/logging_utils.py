import os
import logging
from effektprognoser.paths import LOG_DIR


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

    # Add the log message to the log file
    logging.info(", ".join(log_msg))
