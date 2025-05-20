import pandas as pd

from unittest.mock import patch

from ep.data_manager import get_kommuner_in_region


def test_get_kommuner_in_region():
    """What this tests?
    * That kommuner from multiple files are merged.
    * That duplicates are removed.
    * That the result is sorted (for consistency in testing)."""
    filenames = ["file_2022.parquet", "file_2030.parquet"]
    region = "test_region"

    # Mock return values for different files
    mock_returns = {
        "file_2022.parquet": pd.DataFrame({"kn": ["A", "B", "C"]}),
        "file_2030.parquet": pd.DataFrame({"kn": ["B", "C", "D"]}),
    }

    def mock_load_parquet(filename, region, cols=None):
        return mock_returns[filename]

    with patch("ep.data_manager.kommuner.load_parquet", side_effect=mock_load_parquet):
        kommuner = get_kommuner_in_region(filenames, region)
        assert kommuner == ["A", "B", "C", "D"]
