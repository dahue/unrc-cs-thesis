#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Exit on any error
set -e

echo "Cleaning project at: $ROOT_PATH"
echo ""

echo "Removing temporary files..."
rm -rf "$TMP_DIR/spider_data"
rm -rf "$TMP_DIR/spider_data.zip"
rm -rf "$TMP_DIR/NatSQL"

echo "Removing databases..."
rm -rf "$ROOT_PATH/database/bronze/bronze.sqlite"
rm -rf "$ROOT_PATH/database/silver/silver.sqlite"
rm -rf "$ROOT_PATH/database/gold/gold.sqlite"
rm -rf "$ROOT_PATH/database/spider"

echo "Removing data artifacts..."
rm -rf "$ROOT_PATH/data/training"
rm -rf "$ROOT_PATH/data/predictions"
rm -rf "$ROOT_PATH/data/benchmark"

echo ""
echo "Done."

