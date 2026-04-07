#!/bin/bash
# ==========================================================================
# download_model.sh -- Download OmniVoice model from HuggingFace
#
# Downloads:
#   - k2-fsa/OmniVoice (~1.2 GB)
#
# Usage:
#   ./download_model.sh                        # download to ./model
#   ./download_model.sh --output-dir ./model   # download to specific dir
#   ./download_model.sh --check                # check if already downloaded
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/model"
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--output-dir /path/to/dir] [--check]"
            echo ""
            echo "Downloads k2-fsa/OmniVoice from HuggingFace."
            echo ""
            echo "Options:"
            echo "  --output-dir DIR   Download to specific directory (default: ./model)"
            echo "  --check            Only check if model is already available"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

MODEL_REPO="k2-fsa/OmniVoice"

# -------------------------------------------------------------------------
# Check if already downloaded
# -------------------------------------------------------------------------

check_model() {
    for candidate in "$OUTPUT_DIR" "$HOME/OmniVoice" "${SCRIPT_DIR}/OmniVoice"; do
        if [ -d "$candidate" ] && [ -f "$candidate/config.json" ]; then
            echo "FOUND: $candidate"
            return 0
        fi
    done
    echo "NOT_FOUND"
    return 1
}

echo "=========================================="
echo "  OmniVoice Model Downloader"
echo "=========================================="
echo "  Model: ${MODEL_REPO}"
echo "  Output: ${OUTPUT_DIR}"
echo "=========================================="

# Check existing
EXISTING=$(check_model 2>/dev/null || true)
if [[ "$EXISTING" == FOUND:* ]]; then
    MODEL_PATH="${EXISTING#FOUND: }"
    echo ""
    echo "  Model already downloaded: ${MODEL_PATH}"

    if [ "$CHECK_ONLY" = true ]; then
        echo ""
        echo "  Key files:"
        for f in "$MODEL_PATH"/config.json "$MODEL_PATH"/*.safetensors "$MODEL_PATH"/*.bin; do
            if [ -e "$f" ]; then
                SIZE=$(du -sh "$f" 2>/dev/null | cut -f1)
                echo "    $(basename "$f") (${SIZE})"
            fi
        done
        TOTAL=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
        echo "  Total: ${TOTAL}"
        exit 0
    fi

    echo "  Skipping download (already exists)."
    echo ""
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "  Model NOT found. Run without --check to download."
    exit 1
fi

# -------------------------------------------------------------------------
# Download
# -------------------------------------------------------------------------

echo ""
echo "Downloading ${MODEL_REPO}..."
echo "  This may take several minutes (~1.2 GB)."
echo ""

# Try huggingface-cli first, fall back to Python
if command -v hf &>/dev/null; then
    hf download "${MODEL_REPO}" --local-dir "${OUTPUT_DIR}"
elif command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "${MODEL_REPO}" --local-dir "${OUTPUT_DIR}"
else
    python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    '${MODEL_REPO}',
    local_dir='${OUTPUT_DIR}',
)
print(f'Downloaded to: {path}')
"
fi

DOWNLOAD_EXIT=$?
if [ $DOWNLOAD_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: Download failed (exit code: $DOWNLOAD_EXIT)"
    echo ""
    echo "Possible fixes:"
    echo "  1. Check internet connectivity"
    echo "  2. Install huggingface_hub: pip install huggingface_hub"
    echo "  3. Login if gated: huggingface-cli login"
    exit 1
fi

# Verify
echo ""
echo "Verifying download..."
VERIFY=$(check_model 2>/dev/null || true)
if [[ "$VERIFY" == FOUND:* ]]; then
    MODEL_PATH="${VERIFY#FOUND: }"
    echo "  Verified: ${MODEL_PATH}"
    TOTAL=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo "  Total: ${TOTAL}"
else
    echo "  WARNING: Could not verify download location."
fi

echo ""
echo "=========================================="
echo "  Download Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  Native:  ./launch.sh --native --port 8000"
echo "  Docker:  ./build.sh && ./launch.sh --port 8000"
echo ""
