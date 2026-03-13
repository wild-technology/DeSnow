#!/bin/bash
# Download the Marine Snow Removal Benchmark (MSRB) dataset
# Reference: https://github.com/ychtanaka/marine-snow
#
# Dataset structure after download:
#   data/msrb/train/snow/    - Images with marine snow (2300 images)
#   data/msrb/train/clean/   - Corresponding clean images
#   data/msrb/test/snow/     - Test images with marine snow (400 images)
#   data/msrb/test/clean/    - Corresponding clean test images

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/msrb"

echo "=== Marine Snow Removal Benchmark (MSRB) Dataset ==="
echo ""
echo "This dataset must be downloaded manually from:"
echo "  https://github.com/ychtanaka/marine-snow"
echo ""
echo "Instructions:"
echo "  1. Visit the GitHub repository above"
echo "  2. Follow the download links for the MSRB dataset"
echo "  3. Extract the archive into: ${DATA_DIR}"
echo ""
echo "Expected directory structure:"
echo "  ${DATA_DIR}/train/snow/   - Snowy training images"
echo "  ${DATA_DIR}/train/clean/  - Clean training images"
echo "  ${DATA_DIR}/test/snow/    - Snowy test images"
echo "  ${DATA_DIR}/test/clean/   - Clean test images"
echo ""

# Create directories
mkdir -p "${DATA_DIR}/train/snow"
mkdir -p "${DATA_DIR}/train/clean"
mkdir -p "${DATA_DIR}/test/snow"
mkdir -p "${DATA_DIR}/test/clean"

echo "Created directory structure at: ${DATA_DIR}"
echo ""

# Check if data exists
TRAIN_COUNT=$(find "${DATA_DIR}/train/snow" -type f -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
TEST_COUNT=$(find "${DATA_DIR}/test/snow" -type f -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)

if [ "$TRAIN_COUNT" -gt 0 ] || [ "$TEST_COUNT" -gt 0 ]; then
    echo "Found ${TRAIN_COUNT} training images and ${TEST_COUNT} test images."
else
    echo "No images found yet. Please download and extract the dataset."
fi
