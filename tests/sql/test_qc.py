import pytest
import pandas as pd

from ep.sql.qc import has_complete_days


def test_has_complete_days():
    df = pd.DataFrame(
        {
            "rid": [1] * 8760,
            "Elanvandning": [1] * 8760,
        }
    )

    assert has_complete_days(df, 2020)
