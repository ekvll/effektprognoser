#!/bin/bash

# Python CLI
PYTHON_CLI="eptools"

# List subcommands to execute"
SUBCOMMANDS=("kommun")

# Loop through subcommands
for cmd in "${SUBCOMMANDS[@]}"; do
    echo "Running: python $PYTHON_CLI $cmd"
    "$PYTHON_CLI" "$cmd" "--help"
done
