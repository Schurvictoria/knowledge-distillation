#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c "from distil.data import download_gender_data, download_rosbank_data, download_age_data; download_gender_data(); download_rosbank_data(); download_age_data()"
echo "Datasets downloaded into ./data/"
