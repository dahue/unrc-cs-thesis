#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Now use the variables
echo "Project root is: $ROOT_PATH"

# Exit on any error
set -e

rm -rf "$TMP_DIR/spider_data"
rm -rf "$TMP_DIR/spider_data.zip"
rm -rf "$TMP_DIR/NatSQL"

rm -rf "$ROOT_PATH/database/bronze/bronze.sqlite"
rm -rf "$ROOT_PATH/database/bronze/spider_databases"
rm -rf "$ROOT_PATH/database/silver/silver.sqlite"
rm -rf "$ROOT_PATH/database/gold/gold.sqlite"

rm -rf "$ROOT_PATH/data/training/nl2SQL/"
rm -rf "$ROOT_PATH/data/training/nl2NatSQL/"
