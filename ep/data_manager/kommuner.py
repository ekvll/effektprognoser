from ep.paths import load_parquet


def get_kommuner_in_region(filenames: list[str], region: str) -> list[str]:
    """Get the list of unique kommuner in the region from the parquet files."""
    kommuner_set = set()

    for filename in filenames:
        gdf = load_parquet(filename, region, cols=["kn"])
        kommuner_set.update(gdf["kn"].unique())

    return sorted(list(kommuner_set))
