#!/bin/bash
# ==========================================================================
# build.sh -- Build the OmniVoice Docker image with models and Neuron traces
#
# Two-phase build:
#   Phase 1: docker build  -- installs deps, copies code + model weights
#   Phase 2: docker run    -- traces models on NeuronCores (TP=2), then commits
#
# MUST be run on a trn1 / inf2 instance (requires /dev/neuron0) for Phase 2.
#
# Usage:
#   ./build.sh                                  # uses auto-detected model dir
#   ./build.sh --download                       # download from HuggingFace first
#   ./build.sh --model-dir /path/to/model       # custom model weights path
#   ./build.sh --skip-trace                     # skip phase 2 (traces on first launch)
#   ./build.sh --buckets 256,512,768,1024  # custom bucket sizes
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${OMNIVOICE_IMAGE_NAME:-omnivoice-neuron-streaming-server}"
IMAGE_TAG="${OMNIVOICE_IMAGE_TAG:-latest}"
BASE_TAG="${IMAGE_TAG}-base"
TP_DEGREE="${TP_DEGREE:-2}"
BUCKETS="${OMNIVOICE_BUCKETS:-256,512,768,1024}"

MODEL_DIR="${OMNIVOICE_MODEL_DIR:-}"
SKIP_TRACE=false
DOWNLOAD=false
DOCKER_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --skip-trace)
            SKIP_TRACE=true
            shift
            ;;
        --download)
            DOWNLOAD=true
            shift
            ;;
        --buckets)
            BUCKETS="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        *)
            DOCKER_EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# -------------------------------------------------------------------------
# Auto-detect model directory
# -------------------------------------------------------------------------

resolve_model_dir() {
    if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
        echo "$MODEL_DIR"
        return
    fi
    for candidate in \
        "$SCRIPT_DIR/model" \
        "$HOME/OmniVoice"; do
        if [ -d "$candidate" ] && [ -f "$candidate/config.json" ]; then
            echo "$candidate"
            return
        fi
    done
    echo ""
}

if [ "$DOWNLOAD" = true ]; then
    echo ""
    echo "[Pre-build] Downloading OmniVoice from HuggingFace..."
    ./download_model.sh --output-dir "${SCRIPT_DIR}/model"
fi

MODEL_DIR=$(resolve_model_dir)

# -------------------------------------------------------------------------
# Validate prerequisites
# -------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "  OmniVoice Docker Build"
echo "=========================================="
echo "  Image:      ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Model dir:  ${MODEL_DIR:-NOT FOUND}"
echo "  TP degree:  ${TP_DEGREE}"
echo "  Buckets:    ${BUCKETS}"
echo "  Skip trace: ${SKIP_TRACE}"
echo "=========================================="

if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found."
    echo ""
    echo "Either:"
    echo "  1. Run: ./build.sh --download"
    echo "  2. Use: ./build.sh --model-dir /path/to/model"
    echo "  3. Download manually:"
    echo "     huggingface-cli download k2-fsa/OmniVoice --local-dir ./model"
    exit 1
fi

if [ "$SKIP_TRACE" = false ]; then
    if [ ! -e /dev/neuron0 ]; then
        echo "ERROR: /dev/neuron0 not found. Neuron tracing requires a trn1/inf2 instance."
        echo "  Use --skip-trace to build without tracing (traces generated on first launch)."
        exit 1
    fi
fi

# -------------------------------------------------------------------------
# Phase 0: Stage model weights and voices into build context
# -------------------------------------------------------------------------

echo ""
echo "[Phase 0] Staging model weights into build context..."

BUILD_MODEL_DIR="${SCRIPT_DIR}/.build_model"
rm -rf "$BUILD_MODEL_DIR"
mkdir -p "$BUILD_MODEL_DIR"

echo "  Copying model files from ${MODEL_DIR}..."
cp -rL "$MODEL_DIR"/* "$BUILD_MODEL_DIR/" 2>/dev/null || true

MODEL_SIZE=$(du -sh "$BUILD_MODEL_DIR" | cut -f1)
FILE_COUNT=$(find "$BUILD_MODEL_DIR" -type f | wc -l)
echo "  Staged ${MODEL_SIZE} (${FILE_COUNT} files)"

# Stage voice reference samples
BUILD_VOICES="${SCRIPT_DIR}/.build_voices"
rm -rf "$BUILD_VOICES"
mkdir -p "$BUILD_VOICES"

VOICES_SRC="${OMNIVOICE_VOICES_DIR:-${SCRIPT_DIR}/voices}"
if [ -d "$VOICES_SRC" ]; then
    echo "  Copying voice samples from ${VOICES_SRC}..."
    cp -rL "$VOICES_SRC"/* "$BUILD_VOICES/" 2>/dev/null || true
fi

# -------------------------------------------------------------------------
# Phase 1: Docker build (code + deps + models)
# -------------------------------------------------------------------------

echo ""
echo "[Phase 1] Building base Docker image..."

docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}:${BASE_TAG}" \
    "${DOCKER_EXTRA_ARGS[@]+"${DOCKER_EXTRA_ARGS[@]}"}" \
    .

echo "  Base image built: ${IMAGE_NAME}:${BASE_TAG}"

rm -rf "$BUILD_MODEL_DIR"
rm -rf "$BUILD_VOICES"

# -------------------------------------------------------------------------
# Phase 2: Trace models on NeuronCores
# -------------------------------------------------------------------------

if [ "$SKIP_TRACE" = true ]; then
    echo ""
    echo "[Phase 2] SKIPPED (--skip-trace)"
    docker tag "${IMAGE_NAME}:${BASE_TAG}" "${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo ""
    echo "[Phase 2] Tracing models on NeuronCores..."
    echo "  This compiles OmniVoice backbone (TP=${TP_DEGREE}, buckets: ${BUCKETS})."
    echo "  May take 10-20 minutes depending on instance type and bucket count."

    TRACE_CONTAINER="omnivoice-neuron-trace-$$"

    docker run \
        --name "$TRACE_CONTAINER" \
        --device=/dev/neuron0 \
        -e NEURON_RT_NUM_CORES=2 \
        -e OMP_NUM_THREADS=1 \
        -e MKL_NUM_THREADS=1 \
        -e TP_DEGREE="${TP_DEGREE}" \
        -e OMNIVOICE_PYTHON_CMD="python3 -u" \
        --entrypoint python3 \
        "${IMAGE_NAME}:${BASE_TAG}" \
        -c "
import sys, os
sys.path.insert(0, '/app')
os.environ.setdefault('OMNIVOICE_MODEL_DIR', '/app/model')
os.environ.setdefault('OMNIVOICE_TRACE_DIR', '/app/neuron_traces')

from omnivoice_neuron_pipeline import OmniVoiceNeuronPipeline

model_dir = os.environ['OMNIVOICE_MODEL_DIR']
trace_dir = os.environ['OMNIVOICE_TRACE_DIR']
tp = int(os.environ.get('TP_DEGREE', '2'))
buckets = [int(b) for b in '${BUCKETS}'.split(',')]

print(f'Tracing: model_dir={model_dir}, trace_dir={trace_dir}, tp={tp}, buckets={buckets}')
pipeline = OmniVoiceNeuronPipeline(
    model_dir=model_dir,
    trace_dir=trace_dir,
    bucket_sizes=buckets,
    force_trace=True,
    tp_degree=tp,
)
print('Tracing complete.')
"

    TRACE_EXIT=$?
    if [ $TRACE_EXIT -ne 0 ]; then
        echo "ERROR: Model tracing failed (exit code: $TRACE_EXIT)"
        docker logs "$TRACE_CONTAINER" 2>&1 | tail -30
        docker rm "$TRACE_CONTAINER" 2>/dev/null
        exit 1
    fi

    docker commit \
        --change 'ENTRYPOINT ["python3", "server.py"]' \
        --change 'CMD ["--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/model", "--trace-dir", "/app/neuron_traces", "--voices-dir", "/app/voices"]' \
        --change 'ENV OMNIVOICE_MODEL_DIR=/app/model' \
        --change 'ENV OMNIVOICE_TRACE_DIR=/app/neuron_traces' \
        --change 'ENV OMNIVOICE_VOICES_DIR=/app/voices' \
        --change "ENV TP_DEGREE=${TP_DEGREE}" \
        --change 'ENV NEURON_RT_NUM_CORES=2' \
        --change 'ENV OMP_NUM_THREADS=4' \
        --change 'EXPOSE 8000' \
        --change 'WORKDIR /app' \
        "$TRACE_CONTAINER" \
        "${IMAGE_NAME}:${IMAGE_TAG}"

    docker rm "$TRACE_CONTAINER" 2>/dev/null
    docker rmi "${IMAGE_NAME}:${BASE_TAG}" 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "  Build Complete"
echo "=========================================="
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Launch:"
echo "  ./launch.sh --port 8000"
echo ""
