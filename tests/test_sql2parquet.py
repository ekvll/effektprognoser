import os
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
import sqlite3
import tempfile

from unittest import mock
from shapely import Point, Polygon
from unittest.mock import patch

from ep.config import SQL_DIR, PARQUET_DIR
from ep.cli.sql2parquet import has_complete_days
from ep.cli.sql2parquet import parquet_filenames, load_parquet, db_path
from ep.cli.sql2parquet import (
    db_tables,
    db_years,
    drop_nan_row,
    set_dtypes,
    sort_df,
    drop_column,
    add_geometry,
    largest_area,
    polygon_intersection,
    to_gdf,
    group_elanvandning,
    compute_summary_stats,
    calculate_energy_statistics,
    get_largest_area_geometry,
    validate_geometries,
)
from ep.cli.sql2parquet import aggregate_loadprofile

from ep.cli.sql2parquet import (
    get_column_names,
    build_select_query,
    format_df,
    load_table_chunks,
)


from ep.cli.sql2parquet import get_kommuner_in_region


from ep.cli.sql2parquet import (
    connect_to_db,
    get_cursor,
    validate_connection,
    db_connect,
)


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


# Patch PARQUET_DIR during the test
@pytest.fixture
def patch_parquet_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("ep.cli.sql2parquet.PARQUET_DIR", str(tmp_path))
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
    monkeypatch.setattr("ep.cli.sql2parquet.PARQUET_DIR", str(tmp_path))

    return "sample.parquet", region, df


def test_load_parquet_all_columns(temp_parquet_file):
    filename, region, expected_df = temp_parquet_file
    result = load_parquet(filename, region)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.equals(expected_df)


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


poly_small = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
poly_large = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])


def test_db_tables_returns_table_names() -> None:
    # Use an in-memory database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create some tables
    cursor.execute("CREATE TABLE test1 (id INTEGER);")
    cursor.execute("CREATE TABLE test2 (name TEST);")
    conn.commit()

    # Call the function
    result = db_tables(cursor)

    # Check if the expected tables are in the result
    assert "test1" in result
    assert "test2" in result
    assert isinstance(result, list)
    assert all(isinstance(name, str) for name in result)

    conn.close()


def test_db_tables_with_no_tables() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    result = db_tables(cursor)
    assert result == []

    conn.close()


def test_db_years() -> None:
    tables = ["data_2021", "data_2020", "data_2022"]

    expected_years = ["2020", "2021", "2022"]

    # Call the function
    result = db_years(tables)

    # Check if the expected years are in the result
    assert result == expected_years


def test_drop_nan_row() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, None],
            "B": [4, None, 6],
        }
    )

    result = drop_nan_row(df)

    assert result.shape[0] == 1
    assert result.shape[1] == 2
    assert result.iloc[0]["A"] == 1
    assert result.iloc[0]["B"] == 4


def test_drop_nan_row_with_col() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, None],
            "B": [4, 5, 6],
        }
    )

    result = drop_nan_row(df)
    # Should drop the row with NaN in column A
    assert result.shape[0] == 2
    assert result["A"].isnull().sum() == 0


def test_drop_nan_row_with_invalid_col() -> None:
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError, match="Column 'missing' not found"):
        drop_nan_row(df, col="missing")


def test_drop_nan_row_all_cols_no_nan() -> None:
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )

    result = drop_nan_row(df)

    # No rows should be dropped
    pd.testing.assert_frame_equal(result, df)


def test_set_dtypes_success():
    df = pd.DataFrame(
        {
            "rid": [1, 2],
            "Elanvandning": [0.1, 0.2],
            "Tidpunkt": [20220101, 20220102],
        }
    )
    result = set_dtypes(df)
    assert result.dtypes["rid"] == np.dtype("uint64")


def test_sort_df_multiple_columns():
    df = pd.DataFrame(
        {
            "a": [2, 1, 3],
            "b": [9, 8, 7],
        }
    )
    sorted_df = sort_df(df, col=["a", "b"], ascending=[True, False])
    assert sorted_df.iloc[0]["a"] == 1


def test_drop_column():
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
        }
    )

    result = drop_column(df, "A")
    assert "A" not in result.columns


def test_add_geometry_success():
    df = pd.DataFrame({"rid": [1, 2]})
    grid = pd.DataFrame({"rid": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]})

    result = add_geometry(df, grid)
    assert "geometry" in result.columns
    assert result.loc[0, "geometry"] == Point(0, 0)
    assert result.loc[1, "geometry"] == Point(1, 1)


def test_add_geometry_missing_rid():
    df = pd.DataFrame({"rid": [1, 3]})
    grid = pd.DataFrame({"rid": [1, 2], "geometry": [Point(0, 0), Point(1, 1)]})

    with pytest.raises(ValueError) as excinfo:
        add_geometry(df, grid)

    assert "Missing geometry for rid(s): [3]" in str(excinfo.value)


def test_calculate_energy_statistics():
    df = pd.DataFrame(
        {
            "rid": [1, 1, 2, 2],
            "Elanvandning": [10.0, 12.5, 8.0, 9.0],
        }
    )

    result = calculate_energy_statistics(df)

    assert len(result) == 2
    assert result.iloc[0]["rid"] == 1
    assert result.iloc[0]["ea"] == 22.5
    assert result.iloc[0]["eb"] == 12.5
    assert result.iloc[0]["lp"] == [10.0, 12.5]

    assert result.iloc[1]["rid"] == 2
    assert result.iloc[1]["ea"] == 17.0
    assert result.iloc[1]["eb"] == 9.0
    assert result.iloc[1]["lp"] == [8.0, 9.0]


def test_compute_summary_stats():
    df = pd.DataFrame(
        {
            "rid": [1, 2, 3],
            "lp": [[10.0, 12.5], [8.0, 9.0, 7.5], [14.0]],
        }
    )

    result = compute_summary_stats(df)

    assert len(result) == 3  # One for each unique rid
    assert result.iloc[0]["rid"] == 1
    assert result.iloc[0]["ea"] == 22.5
    assert result.iloc[0]["eb"] == 12.5


def test_group_elanvandning():
    df = pd.DataFrame(
        {
            "rid": [1, 1, 2, 2, 2, 3],
            "Elanvandning": [10.0, 12.5, 8.0, 9.0, 7.5, 14.0],
        }
    )

    result = group_elanvandning(df)
    assert len(result) == 3  # One for each unique rid
    assert result["rid"].tolist() == [1, 2, 3]
    assert result.iloc[0]["lp"] == [10.0, 12.5]
    assert result.iloc[1]["lp"] == [8.0, 9.0, 7.5]
    assert result.iloc[2]["lp"] == [14.0]


def test_to_gdf():
    df = pd.DataFrame(
        {
            "rid": [1, 2],
            "geometry": [Point(0, 0), Point(1, 1)],
        }
    )

    gdf = to_gdf(df, crs="EPSG:3006")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:3006"


def test_polygon_intersection():
    # Define two polygons that intersect
    poly1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    poly2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])

    gdf1 = gpd.GeoDataFrame({"rid": [1]}, geometry=[poly1], crs="EPSG:3006")
    gdf2 = gpd.GeoDataFrame({"rid": [1]}, geometry=[poly2], crs="EPSG:3006")

    result = polygon_intersection(gdf1, gdf2)

    # One intersecting polygon should exist
    assert len(result) == 1

    # Check that the intersection geometry is correct
    expected_geom = poly1.intersection(poly2)
    assert result.geometry.iloc[0].equals_exact(expected_geom, tolerance=1e-6)


def test__validate_geometries(capsys):
    # Valid square polygon
    valid_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Invalid polygon (self-intersecting bowtie)
    invalid_polygon = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])

    gdf = gpd.GeoDataFrame(
        {
            "rid": [1, 2],
            "geometry": [valid_polygon, invalid_polygon],
        },
        crs="EPSG:3006",
    )

    result = validate_geometries(gdf)

    # Assert that only 1 row remains
    assert len(result) == 1
    assert result.iloc[0]["rid"] == 1

    # Capture and assert stdout
    captured = capsys.readouterr()
    assert "Dropped invalid geometries" in captured.out


def test__get_largest_area_geometry():
    global poly_small, poly_large

    df = gpd.GeoDataFrame(
        {
            "rid": [1, 1],
            "geometry": [poly_small, poly_large],
        },
        crs="EPSG:3006",
    )

    result = get_largest_area_geometry(df)

    assert len(result) == 1
    assert result.iloc[0].geometry.equals(poly_small.union(poly_large))
    assert result.iloc[0]["rid"] == 1


def test_largest_area():
    global poly_small, poly_large

    df = gpd.GeoDataFrame(
        {"rid": [1, 1, 2], "geometry": [poly_small, poly_large, poly_small]},
        crs="EPSG:3006",
    )

    result = largest_area(df)

    assert len(result) == 2  # One for each unique rid
    assert set(result["rid"]) == {1, 2}
    assert "area" not in result.columns


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


def test_get_column_names_returns_correct_names() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER, name TEXT);")

    cols = get_column_names(cursor, "test")

    assert cols == ["id", "name"]

    conn.close()


def test_format_df_renames_and_sorts() -> None:
    df = pd.DataFrame(
        {
            "rut_id": [2, 1, 1],
            "Tidpunkt": ["2025-05-13", "2025-05-14", "2025-05-15"],
            "value": [300, 100, 200],
        }
    )

    formatted = format_df(df)

    assert list(formatted.columns) == ["rid", "Tidpunkt", "value"]
    assert formatted.iloc[0]["rid"] == 1  # First row after sorting
    assert formatted.iloc[0]["Tidpunkt"] == "2025-05-14"


def test_load_table_chunks() -> None:
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (rut_id INTEGER, Tidpunkt TEXT, value REAL);")
    cursor.executemany(
        "INSERT INTO test (rut_id, Tidpunkt, value) VALUES (?, ?, ?);",
        [(1, "2025-05-13", 300), (2, "2025-05-14", 100), (1, "2025-05-15", 200)],
    )
    conn.commit()

    df = load_table_chunks(conn, cursor, "test", chunk_size=2)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert list(df.columns) == ["rid", "Tidpunkt", "value"]
    assert df.iloc[0]["rid"] == 1  # First row after sorting

    conn.close()


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

    with patch("ep.cli.sql2parquet.load_parquet", side_effect=mock_load_parquet):
        kommuner = get_kommuner_in_region(filenames, region)
        assert kommuner == ["A", "B", "C", "D"]


def test_connect_to_db_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = connect_to_db(tmp.name)
        assert isinstance(conn, sqlite3.Connection)
        conn.close()


def test_get_cursor() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = sqlite3.connect(tmp.name)
        cursor = get_cursor(conn)
        assert isinstance(cursor, sqlite3.Cursor)
        conn.close()


def test_validate_connection_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn = sqlite3.connect(tmp.name)
        validate_connection(conn)
        conn.close()


def test_db_connect_success() -> None:
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        conn, cursor = db_connect(tmp.name)
        assert isinstance(conn, sqlite3.Connection)
        assert isinstance(cursor, sqlite3.Cursor)
        conn.close()
