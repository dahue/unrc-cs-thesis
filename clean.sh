#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Exit on any error
set -e

ALL=false
for arg in "$@"; do
    [ "$arg" = "--all" ] && ALL=true
done

echo "Cleaning project at: $ROOT_PATH"
echo ""

if [ "$ALL" = true ]; then
    echo "Removing Spider dataset..."
    rm -rf "$TMP_DIR/spider_data"
    rm -rf "$TMP_DIR/spider_data.zip"
    rm -rf "$TMP_DIR/NatSQL"
    echo ""
fi

echo "Removing databases..."
rm -rf "$ROOT_PATH/database/OpenText2SQL.db"
rm -rf "$ROOT_PATH/database/spider"

echo "Removing data artifacts..."
rm -rf "$ROOT_PATH/data/training"
rm -rf "$ROOT_PATH/data/predictions"
rm -rf "$ROOT_PATH/data/benchmark"

echo ""
echo "Done."
