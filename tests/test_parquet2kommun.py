import pytest

from ep.cli.parquet2kommun import get_expected_length


def test_get_expected_length():
    assert get_expected_length("2022") == 8760
    assert get_expected_length("2027") == 8760
    assert get_expected_length("2030") == 8760
    assert get_expected_length("2040") == 8784
