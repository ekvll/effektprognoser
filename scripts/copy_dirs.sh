#!/bin/bash

# Source and destination directories (edit these as needed)
SRC_DIR=$HOME"/python/effektprognoser/data/geojson/"
DEST_DIR=$HOME"/python/effektprognoser-webmap-2/assets/data/"

# Ensure destination directory exists
mkdir -p "$DEST_DIR"

# Use rsync to copy everything except _tmp directory
rsync -av --exclude='_tmp' "$SRC_DIR" "$DEST_DIR"

echo "Copy completed, excluding _tmp directory."
