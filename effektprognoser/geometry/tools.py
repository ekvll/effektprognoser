import pandas as pd
import geopandas as gpd


def add_geometry(df, gdf_grid):
    square_mapping = gdf_grid["geometry"].to_dict()
    geometries = [square_mapping.get(row.rid, None) for _, row in df.iterrows()]
    df = df.assign(geometry=geometries)
    return df


def convert_df_to_gdf(df, geometry: str = "geometry", crs: str = "EPSG:3006"):
    if isinstance(df, gpd.GeoDataFrame):
        if not df.crs:
            df = df.set_crs(crs)

        if df.crs != crs:
            df = df.to_crs(crs)

        if geometry not in df.columns:
            raise NameError(f"Column '{geometry}' not found in DataFrame")

        return df

    return gpd.GeoDataFrame(df, geometry=df[geometry], crs=crs)


def intersection_by_polygon(gdf, gdf_intersect) -> gpd.GeoDataFrame:
    if not gdf.is_valid.all():
        print(
            "intersection_by_polygon: invalid geometries detected in gdf. Attempting to fix."
        )
        gdf = gdf[gdf.is_valid].reset_index(drop=True)

    if not gdf_intersect.is_valid.all():
        print(
            "intersection_by_polygon: invalid geometries in gdf_intersect. Attempting to fix."
        )
        gdf_intersect = gdf_intersect[gdf_intersect.is_valid].reset_index(drop=True)

    if gdf.empty:
        print("intersection_by_polygon: gdf is empty")
    if gdf_intersect.empty:
        print("intersection_by_polygon: gdf_intersect is empty")

    gdf_copy = gdf.copy(deep=True)
    gdf_copy = gpd.overlay(gdf_copy, gdf_intersect, how="intersection")
    return gdf_copy


def keep_largest_area(gdf: gpd.GeoDataFrame):
    def _get_largest_area_geometry(gdf):
        gdf_uid = gdf.assign(area=gdf["geometry"].area / 10**6).reset_index(drop=True)
        idx_max_area = gdf_uid.area.idxmax()
        gdf_dissolve = gdf_uid.dissolve()
        gdf_max = gdf_uid.iloc[[idx_max_area]]
        gdf_max = gdf_max.assign(geometry=gdf_dissolve.iloc[0]["geometry"])
        return gdf_max

    gdf_out = gpd.GeoDataFrame()

    for uid in gdf.rid.unique():
        gdf_uid = gpd.GeoDataFrame(
            data=gdf[gdf.rid == uid], geometry="geometry", crs="EPSG:3006"
        )

        if gdf_uid.shape[0] > 1:
            gdf_max = _get_largest_area_geometry(gdf_uid)
            gdf_out = pd.concat([gdf_out, gdf_max])

        else:
            gdf_out = pd.concat([gdf_out, gdf_uid])

    if "area" in gdf_out.columns:
        gdf_out.drop(columns=["area"], axis=0, inplace=True)
    return gdf_out
