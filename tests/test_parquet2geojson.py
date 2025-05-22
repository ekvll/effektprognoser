import pytest

from ep.cli.parquet2geojson import extract_raps_from_filename


def test_extract_raps_from_filename() -> None:
    """
    Test the extract_raps_from_filename function.
    """
    # Test with a valid filename
    filename = "EF_2022_RAPS_16_V1.parquet"
    expected_category = "RAPS 16"
    category = extract_raps_from_filename(filename)

    assert category == expected_category, (
        f"Expected {expected_category}, but got {category}"
    )
    assert category != "RAPS_16", ("Category should not contain underscores",)
