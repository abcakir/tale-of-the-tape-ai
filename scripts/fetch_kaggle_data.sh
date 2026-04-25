#!/usr/bin/env bash
set -euo pipefail

DATASET="aminealibi/ufc-fights-fighters-and-events-dataset"
TARGET_DIR="data"

mkdir -p "$TARGET_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "[fetch] Kaggle CLI nicht gefunden."
  echo "[fetch] Installiere lokal: pip install kaggle"
  echo "[fetch] Dann API-Key hinterlegen: ~/.kaggle/kaggle.json"
  exit 1
fi

echo "[fetch] Lade Dataset $DATASET nach $TARGET_DIR ..."
kaggle datasets download -d "$DATASET" -p "$TARGET_DIR" --unzip

echo "[fetch] Fertig. Dateien in $TARGET_DIR:"
find "$TARGET_DIR" -maxdepth 2 -type f -name "*.csv" -print
