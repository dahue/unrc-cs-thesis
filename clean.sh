#!/bin/bash

# Exit on any error
set -e

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Fail loudly instead of rm -rf'ing a root-relative path if it's unset
: "${ROOT_PATH:?ROOT_PATH not set in .env}"
TMP_DIR="${TMP_DIR:-/tmp}"

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
    echo ""
fi

echo "Removing databases..."
rm -rf "$ROOT_PATH/database/OpenText2SQL.db"
rm -rf "$ROOT_PATH/database/spider"

echo ""
echo "Done."
