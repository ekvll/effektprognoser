import os
import pytest
from unittest import mock

from ep.sql.path import db_path


def test_db_path_exists():
    """Test that the database path exists for specific regions. Regions specified in 'regions' shall exist."""
    regions = ["06", "07", "08", "10", "12", "13"]

    for region in regions:
        path = db_path(region)
        assert os.path.isfile(path), f"Database file {path} not found."


def test_db_path_missing():
    """Test that the database path raises FileNotFoundError for missing regions."""
    regions = ["01", "02", "03", "04", "05"]

    for region in regions:
        # Mock os.path.isfile to return False (simulating file not existing)
        with mock.patch("os.path.isfile", return_value=False):
            # Assert that FileNotFoundError is raised
            with pytest.raises(FileNotFoundError):
                db_path(region)
