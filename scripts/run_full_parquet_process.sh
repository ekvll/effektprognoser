!#/bin/bash

# Activate the Python virtual environment
source .venv/bin/activate

# Run a Python module using the -m flag
python -m ep.cli.parquet2region
python -m ep.cli.parquet2kommun

# geojson2statistics is dependent on parquet2geojson
python -m ep.cli.parquet2geojson
python -m ep.cli.geojson2statistics
