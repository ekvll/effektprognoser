import pytest
import pandas as pd

from ep.sql.qc import has_complete_days


def test_has_complete_days():
    df_2020 = pd.DataFrame(
        {
            "rid": [1] * 8784,
            "Elanvandning": [1] * 8784,
        }
    )

    assert has_complete_days(df_2020, 2020)
    assert has_complete_days(df_2020, "2020")

    df_2027 = pd.DataFrame(
        {
            "rid": [1] * 8760,
            "Elanvandning": [1] * 8760,
        }
    )

    assert has_complete_days(df_2027, 2027)
    assert has_complete_days(df_2027, "2027")


def test_has_no_all_nan_columns():
    pass


def test_has_no_all_zero_columns():
    pass
