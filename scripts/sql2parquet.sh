!#/bin/bash

# Run the SQL to Parquet processes

# Activate the Python virtual environment
source .venv/bin/activate

python -m ep.cli.sql2parquet
python -m ep.cli.sql2parquet_chunk
