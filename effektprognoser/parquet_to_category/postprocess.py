import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from effektprognoser.paths import DATA_DIR


def compare_ref_model(ref_, model_):
    model_copy = model_.copy(deep=True)
    cols = ["eb", "ebd", "ebp", "ea", "ead", "eap"]
    model_copy[cols] = np.nan
    model_copy = model_copy.set_index("rid")

    for idx, row in model_.iterrows():
        model_eb = max(row.lastprofil)
        model_ea = sum(row.lastprofil)

        model_copy.loc[row.rid, "eb"] = model_eb
        model_copy.loc[row.rid, "ea"] = model_ea

        ref_rid = ref_.loc[ref_.rid == row.rid]
        if ref_rid.empty:
            model_copy.loc[row.rid, "ebd"] = model_eb - 0
            model_copy.loc[row.rid, "ebp"] = np.nan

            model_copy.loc[row.rid, "ead"] = model_ea - 0
            model_copy.loc[row.rid, "eap"] = np.nan

        else:
            ref_eb = max(ref_rid.lastprofil)
            ref_ea = sum(ref_rid.lastprofil)

            model_copy.loc[row.rid, "ebd"] = model_eb - ref_eb
            model_copy.loc[row.rid, "ebp"] = ((model_eb - ref_eb) / ref_eb) * 100

            model_copy.loc[row.rid, "ead"] = model_ea - ref_ea
            model_copy.loc[row.rid, "eap"] = ((model_ea - ref_ea) / ref_ea) * 100

    return model_copy


def compare_ref_model_2(ref_, model_):
    model_copy = model_.copy(deep=True)

    # Initialize empty columns
    cols = ["eb", "ebd", "ebp", "ea", "ead", "eap"]
    model_copy[cols] = np.nan

    # Set index for faster lookup
    model_copy = model_copy.set_index("rid")
    ref_indexed = ref_.set_index("rid")

    # Compute eb and ea
    model_copy["eb"] = model_["lastprofil"].apply(max)
    model_copy["ea"] = model_["lastprofil"].apply(sum)

    # Find matching ref values
    ref_eb = ref_indexed["lastprofil"].apply(max)
    ref_ea = ref_indexed["lastprofil"].apply(sum)

    # Compute Differences
    model_copy["ebd"] = model_copy["eb"] - ref_eb.reindex(
        model_copy.index, fill_value=0
    )
    model_copy["ead"] = model_copy["ea"] - ref_ea.reindex(
        model_copy.index, fill_value=0
    )

    # Compute Percentage Differences (Avoiding Division by Zero)
    model_copy["ebp"] = np.where(
        ref_eb.reindex(model_copy.index, fill_value=0) != 0,
        (model_copy["ebd"] / ref_eb.reindex(model_copy.index, fill_value=0)) * 100,
        np.nan,
    )

    model_copy["eap"] = np.where(
        ref_ea.reindex(model_copy.index, fill_value=0) != 0,
        (model_copy["ead"] / ref_ea.reindex(model_copy.index, fill_value=0)) * 100,
        np.nan,
    )

    return model_copy


def as_geojson(gdf, output_filepath) -> None:
    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(gdf)

    if not gdf.crs == "EPSG:3006":
        gdf = gdf.set_crs("EPSG:3006")

    if not gdf.crs == "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf.drop(columns=["lastprofil"])
    gdf.to_file(output_filepath, driver="GeoJSON")


def process_files(ref_, model_, input_path, output_path):
    fig, ax = plt.subplots()
    for m in model_:
        print(m)
        category = m.split("_")[1].split(".")[0]
        for r in ref_:
            if category in r:
                ref = gpd.read_parquet(os.path.join(input_path, r))
                print("Found reference data")
                break
        model = gpd.read_parquet(os.path.join(input_path, m))
        print("Model data")

        model_updated = compare_ref_model_2(ref, model)
        output_filepath = os.path.join(output_path, m.split(".")[0] + ".geojson")
        as_geojson(model_updated, output_filepath)

        # print(gdf.head())


def form_datasets(files):
    ref, model = [], []
    for file in files:
        if "2022" in file:
            ref.append(file)
        else:
            model.append(file)
    return ref, model


def main(region):
    input_path = os.path.join(DATA_DIR, "parquet_to_category", "preprocess", region)
    files = os.listdir(input_path)
    ref, model = form_datasets(files)

    output_path = os.path.join(DATA_DIR, "parquet_to_category", "postprocess", region)
    os.makedirs(output_path, exist_ok=True)

    process_files(ref, model, input_path, output_path)


if __name__ == "__main__":
    region = "06"
    main(region)
