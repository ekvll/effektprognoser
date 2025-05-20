import pytest
import numpy as np
import pandas as pd
from ep.data_manager import aggregate_loadprofile


def test_aggregate_loadprofile_valid():
    year = "2022"
    loadprofiles = [
        np.ones(8760),
        np.full(8760, 2),
        np.arange(8760),
    ]

    gdf = pd.DataFrame({"lp": loadprofiles})

    result = aggregate_loadprofile(gdf, year)

    expected = np.ones(8760) + np.full(8760, 2) + np.arange(8760)

    np.testing.assert_array_equal(result, expected)
