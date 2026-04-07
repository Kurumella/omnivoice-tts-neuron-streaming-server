#!/bin/bash
# ==========================================================================
# launch.sh -- Launch the OmniVoice Neuron streaming server
#
# Modes:
#   1. Docker mode (default if Docker image exists):
#      ./launch.sh --port 8000
#
#   2. Native mode (runs directly with Python):
#      ./launch.sh --native --port 8000
#
# Environment variables:
#   OMNIVOICE_MODEL_DIR  - path to model weights directory
#   OMNIVOICE_TRACE_DIR  - path to Neuron trace cache
#   OMNIVOICE_VOICES_DIR - path to voice reference audio samples
#   OMNIVOICE_PORT       - server port (default: 8000)
#   TP_DEGREE            - tensor parallelism degree (default: 2)
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
IMAGE_NAME="${OMNIVOICE_IMAGE_NAME:-omnivoice-neuron-streaming-server}"
IMAGE_TAG="${OMNIVOICE_IMAGE_TAG:-latest}"
PORT="${OMNIVOICE_PORT:-8000}"
MODEL_DIR="${OMNIVOICE_MODEL_DIR:-}"
TRACE_DIR="${OMNIVOICE_TRACE_DIR:-${SCRIPT_DIR}/neuron_traces}"
VOICES_DIR="${OMNIVOICE_VOICES_DIR:-${SCRIPT_DIR}/voices}"
TP_DEGREE="${TP_DEGREE:-2}"
BUCKETS="${OMNIVOICE_BUCKETS:-256,512,768,1024}"
NUM_STEPS="${OMNIVOICE_NUM_STEPS:-8}"
NATIVE=false

# Parse arguments
ORIG_ARGS=("$@")
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --native)
            NATIVE=true
            shift
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --trace-dir)
            TRACE_DIR="$2"
            shift 2
            ;;
        --voices-dir)
            VOICES_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        --buckets)
            BUCKETS="$2"
            shift 2
            ;;
        --num-steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
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

MODEL_DIR=$(resolve_model_dir)

echo "=========================================="
echo "  OmniVoice Neuron Streaming Server"
echo "=========================================="
echo "  Model dir:  ${MODEL_DIR:-NOT SET (baked into Docker image)}"
echo "  Trace dir:  ${TRACE_DIR}"
echo "  Voices dir: ${VOICES_DIR}"
echo "  Port:       ${PORT}"
echo "  TP degree:  ${TP_DEGREE}"
echo "  Buckets:    ${BUCKETS}"
echo "  Num steps:  ${NUM_STEPS}"
echo "=========================================="

if [ "$NATIVE" = true ]; then
    # -------------------------------------------------------------------------
    # Native mode: run directly with Python
    # -------------------------------------------------------------------------

    if [ -z "$MODEL_DIR" ] || [ ! -d "$MODEL_DIR" ]; then
        echo "ERROR: Model directory not found: ${MODEL_DIR}"
        echo ""
        echo "Either:"
        echo "  1. Download: ./download_model.sh"
        echo "  2. Specify: ./launch.sh --native --model-dir /path/to/model"
        echo "  3. Set OMNIVOICE_MODEL_DIR environment variable"
        exit 1
    fi

    echo "  Mode: Native (direct Python)"
    echo "=========================================="

    export NEURON_RT_NUM_CORES=2
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OMNIVOICE_MODEL_DIR="$MODEL_DIR"
    export OMNIVOICE_TRACE_DIR="$TRACE_DIR"
    export OMNIVOICE_VOICES_DIR="$VOICES_DIR"
    export TP_DEGREE="$TP_DEGREE"

    # Use conda env for neuronx-distributed + torch-neuronx, fallback to system python
    CONDA_ENV="${OMNIVOICE_CONDA_ENV:-}"
    if [[ -n "$CONDA_ENV" ]] && command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
        echo "  Using conda env: $CONDA_ENV"
        exec conda run --no-capture-output -n "$CONDA_ENV" \
            env NEURON_RT_NUM_CORES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            python src/server.py \
            --port "$PORT" \
            --model-dir "$MODEL_DIR" \
            --trace-dir "$TRACE_DIR" \
            --voices-dir "$VOICES_DIR" \
            --tp-degree "$TP_DEGREE" \
            --buckets "$BUCKETS" \
            --num-steps "$NUM_STEPS" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    else
        echo "  Using system Python: $(python3 --version 2>&1)"
        exec python3 src/server.py \
            --port "$PORT" \
            --model-dir "$MODEL_DIR" \
            --trace-dir "$TRACE_DIR" \
            --voices-dir "$VOICES_DIR" \
            --tp-degree "$TP_DEGREE" \
            --buckets "$BUCKETS" \
            --num-steps "$NUM_STEPS" \
            "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
    fi
else
    # -------------------------------------------------------------------------
    # Docker mode
    # -------------------------------------------------------------------------

    echo "  Mode: Docker (${IMAGE_NAME}:${IMAGE_TAG})"
    echo "=========================================="

    if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" &>/dev/null; then
        echo "Docker image not found. Building..."
        ./build.sh
    fi

    DOCKER_VOLUMES=()

    if [ -n "${OMNIVOICE_MODEL_DIR:-}" ] || [[ " ${ORIG_ARGS[*]:-} " == *" --model-dir "* ]]; then
        if [ -n "$MODEL_DIR" ] && [ -d "$MODEL_DIR" ]; then
            DOCKER_VOLUMES+=(-v "${MODEL_DIR}:/app/model:ro")
            echo "  Override: mounting external model dir ${MODEL_DIR}"
        fi
    fi

    if [ -n "${OMNIVOICE_TRACE_DIR:-}" ] || [[ " ${ORIG_ARGS[*]:-} " == *" --trace-dir "* ]]; then
        mkdir -p "$TRACE_DIR"
        DOCKER_VOLUMES+=(-v "${TRACE_DIR}:/app/neuron_traces")
        echo "  Override: mounting external trace dir ${TRACE_DIR}"
    fi

    if [ -n "${OMNIVOICE_VOICES_DIR:-}" ] || [[ " ${ORIG_ARGS[*]:-} " == *" --voices-dir "* ]]; then
        if [ -d "$VOICES_DIR" ]; then
            DOCKER_VOLUMES+=(-v "${VOICES_DIR}:/app/voices:ro")
            echo "  Override: mounting external voices dir ${VOICES_DIR}"
        fi
    fi

    DOCKER_TTY_FLAGS=()
    if [ -t 0 ]; then
        DOCKER_TTY_FLAGS+=(-it)
    fi

    DOCKER_DEVICE_FLAGS=()
    if [ -e /dev/neuron0 ]; then
        DOCKER_DEVICE_FLAGS+=(--device=/dev/neuron0)
    else
        echo "  WARNING: /dev/neuron0 not found. Container may fail without Neuron hardware."
    fi

    exec docker run --rm \
        "${DOCKER_TTY_FLAGS[@]+"${DOCKER_TTY_FLAGS[@]}"}" \
        --name omnivoice-neuron-streaming-server \
        -p "${PORT}:8000" \
        "${DOCKER_DEVICE_FLAGS[@]+"${DOCKER_DEVICE_FLAGS[@]}"}" \
        "${DOCKER_VOLUMES[@]+"${DOCKER_VOLUMES[@]}"}" \
        -e NEURON_RT_NUM_CORES=2 \
        -e OMP_NUM_THREADS=4 \
        -e MKL_NUM_THREADS=4 \
        -e TP_DEGREE="${TP_DEGREE}" \
        "${IMAGE_NAME}:${IMAGE_TAG}" \
        --host 0.0.0.0 --port 8000 \
        --tp-degree "${TP_DEGREE}" \
        --buckets "${BUCKETS}" \
        --num-steps "${NUM_STEPS}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
