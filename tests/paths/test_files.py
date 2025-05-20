import pytest
import geopandas as gpd
from shapely import Point

from ep.paths import parquet_filenames, load_parquet


# Patch PARQUET_DIR during the test
@pytest.fixture
def patch_parquet_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("ep.paths.files.PARQUET_DIR", str(tmp_path))
    return tmp_path


def test_parquet_filenames(patch_parquet_dir):
    region = "test_region"
    region_path = patch_parquet_dir / region
    region_path.mkdir()

    # Create dummy parquet files
    (region_path / "file1.parquet").touch()
    (region_path / "file2.parquet").touch()

    result = parquet_filenames(region)
    assert sorted(result) == ["file1.parquet", "file2.parquet"]


@pytest.fixture
def temp_parquet_file(tmp_path, monkeypatch):
    region = "test_region"
    region_path = tmp_path / region
    region_path.mkdir()

    df = gpd.GeoDataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    file_path = region_path / "sample.parquet"
    df.to_parquet(file_path)

    # Patch the global PARQUET_DIR to point to tmp_path
    monkeypatch.setattr("ep.paths.files.PARQUET_DIR", str(tmp_path))

    return "sample.parquet", region, df


def test_load_parquet_all_columns(temp_parquet_file):
    filename, region, expected_df = temp_parquet_file
    result = load_parquet(filename, region)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.equals(expected_df)
