!#/bin/bash

# Run all processing steps that start from Parquet files

# Activate the Python virtual environment
source .venv/bin/activate

# Combine all regions into one
python -m ep.cli.parquet2all_table

# Convert parquet files to geojson format
# This output is used in the webmap
python -m ep.cli.parquet2geojson
python -m ep.cli.geojson2statistics

# Run a Python module using the -m flag
python -m ep.cli.parquet2region
python -m ep.cli.parquet2kommun
python -m ep.cli.parquet2kommun_maxeffekt
