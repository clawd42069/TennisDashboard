#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RAW_DIR="${TA_RAW_PATH:-$ROOT_DIR/data/ta_raw}"

mkdir -p "$RAW_DIR"

cd "$RAW_DIR"

if [ ! -d "tennis_atp" ]; then
  git clone --depth 1 https://github.com/JeffSackmann/tennis_atp.git
else
  echo "tennis_atp already exists; skipping clone"
fi

if [ ! -d "tennis_wta" ]; then
  git clone --depth 1 https://github.com/JeffSackmann/tennis_wta.git
else
  echo "tennis_wta already exists; skipping clone"
fi

echo "Done. Raw repos in: $RAW_DIR"