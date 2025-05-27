!#/bin/bash

# Activate the Python virtual environment
source .venv/bin/activate

# Convert SQL database to parquet files
python -m ep.cli.sql2parquet
python -m ep.cli.sql2parquet_chunk # For region 12

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
