import logging
import os
from effektprognoser.paths import LOG_DIR

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "qc_log.csv"),
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s, %(message)s",
)


def log_issue(log_msg):
    print(log_msg)

    if not len(log_msg) == 4:
        raise ValueError(
            "log_msg need to contain 'table', 'column', 'rid' and 'comment'"
        )
    str_obj = ", ".join(log_msg)
    logging.info(str_obj)
