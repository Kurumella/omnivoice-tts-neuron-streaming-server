# Standalone Dockerfile for OmniVoice Neuron streaming server.
# For the full two-phase build with Neuron tracing, use build.sh instead.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential git wget curl \
    libsndfile1 ffmpeg \
    libarchive13 \
    gnupg2 software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3*/EXTERNALLY-MANAGED \
    && rm -rf /usr/lib/python3/dist-packages/blinker*

# ---------------------------------------------------------------------------
# Neuron SDK (runtime + tools)
# ---------------------------------------------------------------------------
RUN . /etc/os-release && \
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add - && \
    echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/neuron.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        aws-neuronx-collectives \
        aws-neuronx-runtime-lib \
        aws-neuronx-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
# Install Neuron packages
RUN pip3 install --no-cache-dir \
    neuronx-cc \
    torch-neuronx \
    neuronx_distributed \
    --extra-index-url https://pip.repos.neuron.amazonaws.com

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# OmniVoice model package (may install its own torch version)
RUN pip3 install --no-cache-dir omnivoice

# Re-install Neuron torch + torchaudio to ensure correct versions after omnivoice
RUN pip3 install --no-cache-dir torch-neuronx \
    --extra-index-url https://pip.repos.neuron.amazonaws.com && \
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])") && \
    pip3 install --no-cache-dir "torchaudio==${TORCH_VER}" \
    --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
COPY src/ ./

# ---------------------------------------------------------------------------
# Model weights -- baked into the image
# ---------------------------------------------------------------------------
COPY .build_model/ ./model/

# ---------------------------------------------------------------------------
# Voice reference samples
# ---------------------------------------------------------------------------
COPY .build_voices/ ./voices/

RUN mkdir -p /app/neuron_traces

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV OMNIVOICE_MODEL_DIR=/app/model
ENV OMNIVOICE_TRACE_DIR=/app/neuron_traces
ENV OMNIVOICE_VOICES_DIR=/app/voices
ENV OMNIVOICE_PYTHON_CMD="python3 -u"
ENV TP_DEGREE=2
ENV PATH="/opt/aws/neuron/bin:${PATH}"
ENV NEURON_RT_NUM_CORES=2
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python3", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--model-dir", "/app/model", "--trace-dir", "/app/neuron_traces", "--voices-dir", "/app/voices"]
