def get_regions(regions):
    if regions == "all":
        return ["06", "07", "08", "10", "12", "13"]
    if not isinstance(regions, list):
        return [regions]
    return regions
