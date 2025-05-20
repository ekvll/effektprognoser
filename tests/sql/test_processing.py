import sqlite3
import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon

from ep.sql.processing import (
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
    _get_largest_area_geometry,
    _validate_geometries,
)


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
    pass


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
    print(result)
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

    result = _validate_geometries(gdf)

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

    result = _get_largest_area_geometry(df)

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


if __name__ == "__main__":
    test_group_elanvandning()
