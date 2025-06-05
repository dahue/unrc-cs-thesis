#!/bin/bash

# Load .env file into the current shell session
set -o allexport
source .env
set +o allexport

# Now use the variables
echo "Project root is: $ROOT_PATH"

# Exit on any error
set -e

# Set TMP_DIR to /tmp if it is not defined
if [ -z "$TMP_DIR" ]; then
    echo "TMP_DIR is not defined. Setting it to /tmp"
    TMP_DIR="/tmp"
    echo "TMP_DIR=$TMP_DIR" >> .env
fi
mkdir -p "$TMP_DIR"
echo "Temporary directory is: $TMP_DIR"
echo ""

# Download Spider dataset
SPIDER_DIR="$TMP_DIR/spider_data"
ZIP_FILE="$TMP_DIR/spider_data.zip"
NATSQL_DIR="$TMP_DIR/NatSQL"

if [ -f "$ZIP_FILE" ]; then
  echo "📦 Found existing spider_data.zip. Extracting..."
  unzip "$ZIP_FILE" -d $TMP_DIR/
  rm -rf "$TMP_DIR/__MACOSX"
  echo "✅ spider_data is ready."
else
  echo "⬇️ Downloading spider_data.zip..."
  wget -O "$ZIP_FILE" "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&authuser=0&confirm=t&uuid=c519429f-e190-4024-9db5-5500dd9f73de&at=ALoNOgmVI-vAWDoXBUn2D2Ezy8Fy:1747082984773"
  echo "📦 Extracting spider_data.zip..."
  unzip "$ZIP_FILE" -d $TMP_DIR/
  rm -rf "$TMP_DIR/__MACOSX"
  echo "✅ spider_data is ready."
fi
echo ""

DB_TRAIN="$SPIDER_DIR/database"
DB_TEST="$SPIDER_DIR/test_database"

DB_TRAIN_DIR="$ROOT_PATH/database/spider"

if [ ! -d "$DB_TEST" ]; then
    echo "⚠️  Folder '$DB_TEST' does not exist. Moving."
else
    # Check if it's empty
    if [ -z "$(ls -A "$DB_TEST")" ]; then
        echo "⚠️  Folder '$DB_TEST' is empty. Nothing to move."
    else
        # Move contents with overwrite
        for db in "$DB_TEST"/*; do
            db_name=$(basename "$db")
            target="$DB_TRAIN/$db_name"

            if [ -e "$target" ]; then
                rm -rf "$target"
            fi

            mv "$db" "$target"
            echo "✅ Moved $db_name"
        done
        rm -rf $DB_TEST

        mkdir -p "$DB_TRAIN_DIR"
        for db in "$DB_TRAIN"/*; do
            db_name=$(basename "$db")
            target="$DB_TRAIN_DIR/$db_name"

            if [ -e "$target" ]; then
                rm -rf "$target"
            fi

            cp -r "$db" "$target"
            echo "✅ Copied $db_name"
        done
    fi
fi
echo "📁 All test databases moved into: $DB_TRAIN"
echo ""

# Download NatSQL repo
if [ -d "$NATSQL_DIR" ]; then
  echo "✅ NatSQL repo already exists at $NATSQL_DIR. Skipping clone."
else
  echo "⬇️ Cloning NatSQL repo..."
  git clone https://github.com/dahue/NatSQL "$NATSQL_DIR"
  echo "✅ NatSQL repo cloned."
fi
echo ""

# Install Python dependencies
echo "⬇️ Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Done."
echo ""

# Populate Bronze tables
echo "🐍 Populate Bronze tables..."
python scripts/pipeline/ingest_bronze.py
echo "✅ Done."
echo ""

# Transform Bronze to Silver tables
echo "🐍 Transform Bronze to Silver tables..."
python scripts/pipeline/bronze_to_silver.py
echo "✅ Done."
echo ""

# Transform Silver to Gold tables
echo "🐍 Transform Silver to Gold tables..."
python scripts/pipeline/silver_to_gold.py
echo "✅ Done."
echo ""

# Export Files for Training
echo "🐍 Generate Natural Language to SQL Training Data..."
python scripts/ML/create_dataset.py
echo "✅ Done."
echo ""