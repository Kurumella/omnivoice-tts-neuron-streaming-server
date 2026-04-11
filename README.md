# OmniVoice Neuron Streaming Server

OpenAI-compatible streaming TTS server running **OmniVoice** (k2-fsa) on **AWS Trainium (trn1)** NeuronCores with tensor parallelism (TP=2) and **zero-shot voice cloning**.

> **Important:** This project is designed exclusively for **AWS Trainium (trn1) and Inferentia2 (inf2) instances** powered by AWS Neuron SDK. It does **not** use NVIDIA CUDA GPUs. All model inference runs on NeuronCores via `neuronx_distributed` and `torch-neuronx`.

## Features

- **OpenAI-compatible API** (`POST /v1/audio/speech`) -- drop-in replacement for OpenAI TTS
- **AWS Neuron native** -- runs on Trainium/Inferentia2 NeuronCores, not CUDA GPUs
- **Zero-shot voice cloning** -- clone any voice from a short reference audio sample
- **Streaming audio** -- chunk-level PCM delivery for low time-to-first-audio
- **Multiple output formats** -- PCM, WAV, and MP3
- **Tensor parallelism** -- Qwen3-0.6B backbone sharded across 2 NeuronCores via `neuronx_distributed`
- **600+ languages** -- multilingual TTS supporting Hindi, Telugu, Kannada, Tamil, Marathi, Bengali, Malayalam, English, Chinese, Japanese, Korean, French, German, Spanish, and hundreds more
- **Single-stage generation** -- discrete masked diffusion generates speech directly from text (no separate vocoder)
- **OpenAI voice mapping** -- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` mapped to bundled reference audio
- **Custom voices** -- drop any WAV/MP3/FLAC file into the voices directory

## Prerequisites

- AWS **trn1.2xlarge** instance (1 Neuron device, 2 NeuronCores) -- this project does **not** support NVIDIA GPUs
- **Neuron SDK 2.x** (`aws-neuronx-runtime-lib`, `aws-neuronx-tools`, `torch-neuronx`, `neuronx_distributed`)
- **Docker** (recommended) or conda environment with PyTorch + Neuron packages

## Getting Started

### Step 1: Clone the repository

```bash
git clone https://github.com/aws-samples/omnivoice-tts-neuron-streaming-server.git
cd omnivoice-tts-neuron-streaming-server
```

### Step 2: Download the model

```bash
# Download k2-fsa/OmniVoice from HuggingFace (~1.2 GB)
./download_model.sh

# Or download to a specific directory
./download_model.sh --output-dir ./model

# Check if already downloaded
./download_model.sh --check
```

### Step 3: Build the Docker image

```bash
# Full build with Neuron tracing (recommended, ~10-20 min)
./build.sh

# Or download model and build in one step
./build.sh --download

# Build without tracing (traces generated on first launch)
./build.sh --skip-trace
```

The build runs in two phases:

1. **Phase 1** (`docker build`): Installs Neuron SDK, Python deps, copies code + model weights + voice samples
2. **Phase 2** (`docker run` + `docker commit`): Traces Qwen3 backbone (TP=2, bucketed) on NeuronCores, commits traced container as final image

Phase 2 requires `/dev/neuron0` (trn1/inf2 instance). Use `--skip-trace` to build on non-Neuron hardware.

### Step 4: Launch the server

```bash
# Docker mode (default)
./launch.sh --port 8000

# With custom voice samples
./launch.sh --port 8000 --voices-dir /data/my_voices

# Native mode (no Docker, requires Neuron packages)
OMNIVOICE_CONDA_ENV=my_neuron_env ./launch.sh --native --port 8000
```

### Step 5: Test the server

```bash
# Health check
curl -s http://localhost:8000/health | python3 -m json.tool

# Generate speech
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world!", "voice": "alloy", "response_format": "wav", "stream": false}' \
  --output hello.wav
```

## API Reference

### `POST /v1/audio/speech`

Generate speech from text. Compatible with [OpenAI's TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body:**

```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3",
  "stream": false,
  "language": "en"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | *required* | Text to synthesize (max 4096 chars) |
| `model` | string | -- | Ignored (always uses OmniVoice) |
| `voice` | string | `"alloy"` | OpenAI voice alias, custom voice name, or path to reference audio |
| `response_format` | string | `"pcm"` | `pcm`, `wav`, or `mp3` |
| `stream` | bool | `true` | Stream audio chunks or wait for complete response |
| `language` | string | `"en"` | Language code (`en`, `zh`, `ja`, `ko`, `fr`, `de`, etc.) |
| `speed` | float | `1.0` | Playback speed multiplier |

**Response (streaming, `stream: true`):**
- Content-Type: `audio/pcm`
- Body: Raw PCM16 s16le chunks (24kHz, mono, 16-bit)
- Headers: `X-Request-Id`, `X-Queue-Time`, `X-Sample-Rate`

**Response (non-streaming, `stream: false`):**
- Content-Type: depends on `response_format` (`audio/pcm`, `audio/wav`, or `audio/mpeg`)
- Body: Complete audio data
- Headers: `X-Request-Id`, `X-Inference-Time`, `X-Audio-Duration`, `X-Sample-Rate`

### `GET /v1/audio/voices`

List available voices (builtin + custom) and supported languages.

### `GET /health`

Server health and queue status.

### `GET /metrics`

Performance metrics (average inference time, RTF, throughput).

## Usage Examples

### cURL

```bash
# English (MP3, non-streaming)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the OmniVoice text to speech system.",
    "voice": "alloy",
    "response_format": "mp3",
    "stream": false,
    "language": "en"
  }' \
  --output english.mp3

# Hindi (हिन्दी)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "भारत की जनसंख्या पूरे यूरोप और अफ्रीका महाद्वीप की जनसंख्या से अधिक है।",
    "voice": "ballad",
    "response_format": "wav",
    "stream": false,
    "language": "hi"
  }' \
  --output hindi.wav

# Telugu (తెలుగు)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "తెలుగు భాష భారతదేశంలో విస్తృతంగా మాట్లాడబడుతుంది.",
    "voice": "ash",
    "response_format": "wav",
    "stream": false,
    "language": "te"
  }' \
  --output telugu.wav

# Kannada (ಕನ್ನಡ)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "ಕರ್ನಾಟಕ ರಾಜ್ಯದಲ್ಲಿ ಕನ್ನಡ ಭಾಷೆಯನ್ನು ಮಾತನಾಡುತ್ತಾರೆ.",
    "voice": "coral",
    "response_format": "wav",
    "stream": false,
    "language": "kn"
  }' \
  --output kannada.wav

# Tamil (தமிழ்)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "தமிழ் மொழி உலகின் பழமையான மொழிகளில் ஒன்றாகும்.",
    "voice": "sage",
    "response_format": "wav",
    "stream": false,
    "language": "ta"
  }' \
  --output tamil.wav

# Marathi (मराठी)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "मराठी भाषा महाराष्ट्र राज्याची अधिकृत भाषा आहे.",
    "voice": "vale",
    "response_format": "wav",
    "stream": false,
    "language": "mr"
  }' \
  --output marathi.wav

# Bengali (বাংলা)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "বাংলা ভাষা পূর্ব ভারত ও বাংলাদেশে বহুল প্রচলিত.",
    "voice": "verse",
    "response_format": "wav",
    "stream": false,
    "language": "bn"
  }' \
  --output bengali.wav

# Malayalam (മലയാളം)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "മലയാളം ഭാഷ കേരള സംസ്ഥാനത്തിന്റെ ഔദ്യോഗിക ഭാഷയാണ്.",
    "voice": "lumen",
    "response_format": "wav",
    "stream": false,
    "language": "ml"
  }' \
  --output malayalam.wav

# French (streaming)
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Bonjour, ceci est un test de synthèse vocale en français.",
    "voice": "fable",
    "stream": true,
    "language": "fr"
  }' \
  --output french.pcm

# List available voices and languages
curl -s http://localhost:8000/v1/audio/voices | python3 -m json.tool
```

### Python (requests)

```python
import requests

# Non-streaming MP3
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Hello world!",
        "voice": "alloy",
        "response_format": "mp3",
        "stream": False,
    },
)
with open("output.mp3", "wb") as f:
    f.write(resp.content)

# Streaming PCM
resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello world!", "voice": "echo", "stream": True},
    stream=True,
)
with open("output.pcm", "wb") as f:
    for chunk in resp.iter_content(chunk_size=4096):
        f.write(chunk)
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="Hello from OmniVoice on NeuronCores!",
    response_format="mp3",
) as response:
    response.stream_to_file("output.mp3")
```

## Voice Mapping

OmniVoice uses **zero-shot voice cloning** -- each voice is a short reference audio sample. OpenAI voice names map to bundled reference audio:

| Voice | Reference Audio | Language |
|-------|----------------|----------|
| `alloy` | `en_sample.wav` | English |
| `echo` | `en_sample.wav` | English |
| `fable` | `fr_sample.wav` | French |
| `onyx` | `de_sample.wav` | German |
| `nova` | `es_sample.wav` | Spanish |
| `shimmer` | `ja-sample.wav` | Japanese |
| `ballad` | `hi_sample.wav` | Hindi |
| `ash` | `te_sample.wav` | Telugu |
| `coral` | `kn_sample.wav` | Kannada |
| `sage` | `ta_sample.wav` | Tamil |
| `vale` | `mr_sample.wav` | Marathi |
| `verse` | `bn_sample.wav` | Bengali |
| `breeze` | `gu_sample.wav` | Gujarati |
| `ember` | `pa_sample.wav` | Panjabi |
| `lumen` | `ml_sample.wav` | Malayalam |

### Custom Voices

Add your own voice by placing a WAV, MP3, or FLAC file in the `voices/` directory:

```bash
cp my_voice.wav voices/
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello!", "voice": "my_voice", "stream": false, "response_format": "wav"}' \
  --output output.wav
```

For best results, use 6-15 seconds of clean speech audio at 24kHz or higher.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIVOICE_MODEL_DIR` | Auto-detect | Path to OmniVoice model weights |
| `OMNIVOICE_TRACE_DIR` | `./neuron_traces` | Neuron compiled model cache |
| `OMNIVOICE_VOICES_DIR` | `./voices` | Path to reference voice audio samples |
| `OMNIVOICE_PORT` | `8000` | Server port |
| `OMNIVOICE_BUCKETS` | `256,512,768,1024` | Backbone bucket sizes for Neuron tracing |
| `OMNIVOICE_NUM_STEPS` | `8` | Denoising steps (2=fastest RTF~12x, 8=balanced, 32=best quality) |
| `TP_DEGREE` | `2` | Tensor parallelism degree |
| `OMNIVOICE_CONDA_ENV` | *(unset)* | Conda environment name with Neuron packages |
| `OMNIVOICE_PYTHON_CMD` | *(unset)* | Override Python command for tracing subprocesses |
| `NEURON_RT_NUM_CORES` | `2` | NeuronCores to use |
| `OMP_NUM_THREADS` | `4` | CPU threads for codec decode parallelism |

### CLI Arguments

```bash
python src/server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-dir /path/to/model \
  --trace-dir ./neuron_traces \
  --voices-dir ./voices \
  --tp-degree 2 \
  --buckets 256,512,768,1024 \
  --num-steps 8
```

## Architecture

```
+-----------------------------------------------------------+
|                      Flask Server                          |
|  POST /v1/audio/speech  GET /health  GET /metrics          |
|                                                            |
|  +----------------+    +------------------------------+    |
|  | HTTP Threads   |--->|    Request Queue (FIFO)      |    |
|  | (concurrent)   |    |    max_size=64               |    |
|  +----------------+    +--------------+---------------+    |
|                                       |                    |
|                        +--------------v---------------+    |
|                        |     Worker Thread (single)    |    |
|                        |  OmniVoiceNeuronPipeline TP=2|    |
|                        +--------------+---------------+    |
+-------------------------------|----------------------------+
                                |
            +-------------------|-----------------------+
            |              Neuron Device                 |
            |  +--------------+  +--------------+       |
            |  | NeuronCore 0 |  | NeuronCore 1 |       |
            |  |  (TP rank 0) |  |  (TP rank 1) |       |
            |  +--------------+  +--------------+       |
            |                                           |
            |  Qwen3-0.6B (28 layers)  TP=2 sharded     |
            |  GQA: Q=16, KV=8 heads   Bidirectional    |
            +-------------------------------------------+
```

TP=2 uses both NeuronCores for every inference call. Only one request can execute at a time. The queue pattern lets Flask accept HTTP requests concurrently while the worker processes them sequentially.

### Model Components

| Component | Location | Details |
|-----------|----------|---------|
| **Qwen3-0.6B Backbone** | NeuronCores (TP=2) | 28 layers, 1024 hidden, GQA (Q=16, KV=8), SwiGLU MLP |
| **Audio Embeddings** | CPU | nn.Embedding(8*1025, 1024) -- sums 8 codebook embeddings |
| **Audio Heads** | CPU | nn.Linear(1024, 8*1025) -- projects to per-codebook logits |
| **HiggsAudioV2 Tokenizer** | CPU | Encodes/decodes waveforms to 8-codebook discrete tokens |
| **Text Tokenizer** | CPU | Qwen3 subword tokenizer |
| **Duration Estimator** | CPU | Rule-based, Unicode-aware per-script duration estimation |

### Inference Flow

```
Text ──> Tokenize ──> Duration Estimate ──> Build Input Sequence
                                                    │
     ┌──────────────────────────────────────────────┘
     │
     v
[style | text | ref_audio | MASK * T_target]    (input_ids: 8 x total_len)
     │
     v  (repeat N=8 steps, default)
┌────────────────────────────────────────────┐
│  1. Embed: text_embed + audio_embed (CPU)  │
│  2. Forward: Qwen3 backbone (Neuron TP=2)  │
│  3. Project: audio_heads -> logits (CPU)   │
│  4. CFG: cond + scale*(cond - uncond)      │
│  5. Unmask: select top-k confident tokens  │
└────────────────────────────────────────────┘
     │
     v
Generated Tokens (8 codebooks x T_target)
     │
     v
HiggsAudioV2 Decode ──> 24kHz Waveform ──> HTTP Response
```

### TP Sharding Strategy (Qwen3-0.6B)

| Layer | Sharding | Details |
|-------|----------|---------|
| `q_proj` | ColumnParallel | 16 Q heads -> 8 per core |
| `k_proj` | ColumnParallel | 8 KV heads -> 4 per core |
| `v_proj` | ColumnParallel | 8 KV heads -> 4 per core |
| `o_proj` | RowParallel | All-reduce across cores |
| `gate_proj` | ColumnParallel | 3072 intermediate -> 1536 per core |
| `up_proj` | ColumnParallel | 3072 intermediate -> 1536 per core |
| `down_proj` | RowParallel | All-reduce across cores |
| `RMSNorm` | Replicated | Identical on both cores |

## Project Structure

```
omnivoice-tts-neuron-streaming-server/
├── LICENSE                          # Apache-2.0
├── README.md
├── requirements.txt                 # Python dependencies
├── requirements-dev.txt             # Dev dependencies
├── Dockerfile                       # Docker build definition
├── build.sh                         # Two-phase Docker build (with Neuron tracing)
├── launch.sh                        # Docker / native launch orchestrator
├── download_model.sh                # Download model from HuggingFace
├── .gitignore
├── .dockerignore
├── voices/                          # Reference audio samples for voice cloning
├── src/
│   ├── server.py                    # Flask server with OpenAI-compatible API
│   ├── omnivoice_neuron_pipeline.py # Core TTS pipeline (Neuron + CPU)
│   ├── tp_qwen3_model.py           # TP Qwen3 transformer for Neuron
│   ├── trace_tp_qwen3.py           # Subprocess: trace Qwen3 per bucket (batch=1 + batch=2)
│   └── nki_kernels.py              # NKI fused kernels (RMSNorm, SwiGLU, RoPE)
├── tests/
│   ├── __init__.py
│   ├── test_server.py               # Functional test suite
│   └── test_concurrency.py          # Performance & concurrency tests
└── neuron_traces/                   # Cached Neuron compiled models (auto-created)
```

## Testing

```bash
# Run functional tests (requires running server)
python tests/test_server.py

# Run performance and concurrency tests
python tests/test_concurrency.py --quick
```

## Performance

Measured on **trn1.2xlarge** (1 Neuron device, 2 NeuronCores, TP=2), Neuron SDK 2.x, BF16 precision, **8 denoising steps**, `OMP_NUM_THREADS=4`:

| Metric | Short (5 words) | Medium (16 words) | Long (51 words) |
|--------|-----------------|--------------------|--------------------|
| **Inference Time** | ~0.29s | ~0.55s | ~1.77s |
| **Audio Duration** | ~1.8s | ~5.3s | ~17.2s |
| **RTF** | ~6.1x | ~9.5x | ~9.7x |

**Average RTF: ~9.0x** (range 6-10x across text lengths), **throughput: ~1.77 req/s** (8 steps).

Concurrency scaling (medium text, streaming):

| Concurrency | Throughput | Latency P50 | Latency P95 |
|-------------|-----------|-------------|-------------|
| 1 | 1.77 req/s | ~0.56s | ~0.59s |
| 2 | 1.77 req/s | ~1.08s | ~1.18s |
| 4 | 1.79 req/s | ~1.64s | ~2.24s |
| 8 | 1.80 req/s | ~2.79s | ~4.43s |
| 16 | 1.76 req/s | ~5.05s | ~9.10s |

### Denoising Steps: Quality vs Speed

The `--num-steps` parameter controls the number of masked diffusion iterations. More steps produce higher quality audio but increase latency. Measured on medium-length text (~16 words):

| Steps | RTF | Latency | Quality | Recommended Use |
|-------|-----|---------|---------|-----------------|
| **2** | ~18x | ~0.30s | Low -- fast but may have artifacts, less natural prosody | Real-time streaming, latency-critical apps, previews |
| **4** | ~14x | ~0.38s | Moderate -- noticeable improvement over 2 steps | Balanced speed/quality for interactive use |
| **8** | ~9.5x | ~0.55s | Good -- clear speech, natural prosody | **Default.** Production TTS, voice assistants |
| **16** | ~5x | ~1.05s | Very good -- high fidelity, accurate voice cloning | High-quality content generation |
| **32** | ~2.5x | ~2.10s | **Best** -- maximum quality, most faithful voice cloning | Studio-quality output, offline batch processing |

```bash
# Best quality (32 steps) -- recommended for production audio
python src/server.py --num-steps 32

# Balanced quality/speed (8 steps)
python src/server.py --num-steps 8

# Maximum speed (2 steps) -- RTF ~12x
python src/server.py --num-steps 2
```

**Notes:**
- RTF (Real-Time Factor) = audio duration / inference time. Higher is better.
- Throughput is bounded by TP=2 (both NeuronCores used per request). Sustained **1.77 req/s** under 30s stress test (61 requests, 8 concurrent clients).
- Voice clone prompts are cached after first use, eliminating repeated reference encoding.
- First launch includes Neuron trace compilation (~25 min for 4 buckets). Subsequent launches load from cache (~60s).
- All numbers measured with 8 denoising steps (default). Use `--num-steps 2` for maximum speed (~18x RTF).

### Key Optimizations

The following optimizations achieved a **4.4x throughput improvement** (0.40 -> 1.77 req/s) and **RTF from ~2x to ~9.5x** (at 8 steps):

| Optimization | Impact | Details |
|-------------|--------|---------|
| **Reduced denoising steps** (32 -> 8 default) | ~4x fewer Neuron calls | OmniVoice's masked diffusion converges well with fewer steps; 8 steps balances quality and speed |
| **Multi-threaded CPU decode** (`OMP_NUM_THREADS=4`) | ~4x faster codec | HiggsAudioV2 codec decode was the bottleneck; parallelizing reduced fixed overhead from ~1.14s to ~0.26s |
| **Batch=1 sequential CFG** | Optimal for TP=2 | Both batch=1 and batch=2 traces compiled per bucket; batch=1 sequential used at runtime (faster on trn1.2xlarge due to TP all-reduce overhead with batch=2) |
| **Neuron compiler flags** | Optimized compiled graphs | `--model-type=transformer -O2`, ccop compute overlap, pipeline tiling, vectorized DMA |
| **NKI fused kernels** (available) | Future acceleration | Custom NKI kernels for RMSNorm, SwiGLU, and RoPE in `nki_kernels.py`; currently disabled as the Neuron compiler's `-O2` already fuses these ops |
| **Bucketed sequence traces** | No recompilation | 4 bucket sizes (256-1024) cover all sequence lengths efficiently |

### Bucket Sizes

Qwen3 backbone uses bucketed Neuron traces for efficient sequence handling:

| Bucket | Max Sequence Length | Typical Use |
|--------|-------------------|-------------|
| 256 | 256 tokens | Short sentences |
| 512 | 512 tokens | Medium text |
| 768 | 768 tokens | Paragraphs |
| 1024 | 1024 tokens | Long paragraphs |

## Troubleshooting

### `libfabric.so.1: cannot open shared object file`
Harmless warning during TP tracing. EFA not needed for single-instance TP.

### First launch is slow (10-20 min)
Models must be compiled to Neuron IR on first run. Subsequent launches load from `neuron_traces/` cache. Use `./build.sh` (without `--skip-trace`) to bake traces into the Docker image.

### `Server not ready` in tests
Wait for pipeline initialization to complete. Check server logs for "pipeline ready" message.

### Out of NeuronCore memory
Reduce bucket sizes: `--buckets 256,512` (drop 768, 1024).

### Voice cloning quality issues
- Use 6-15 seconds of clean speech audio as reference
- Sample rate should be 24kHz or higher
- Avoid background noise or music
- Mono audio works best

### Model not found
```bash
./download_model.sh --check
./download_model.sh
```

## OmniVoice Model

OmniVoice is a single-stage, non-autoregressive TTS model based on discrete masked diffusion. Key features:

- **0.8B parameters** (Qwen3-0.6B backbone + audio embeddings + heads)
- **HiggsAudioV2** 8-codebook discrete audio tokenizer at 24kHz
- **Bidirectional attention** with iterative unmasking (2-32 steps, default 8)
- **Classifier-free guidance** for improved generation quality
- **600+ languages** trained on 581k hours of multilingual audio, including Indian languages: Hindi, Telugu, Kannada, Tamil, Marathi, Bengali, Gujarati, Panjabi, Malayalam, Odia, Assamese, Urdu, Kashmiri, Sindhi, Manipuri, Santali, Maithili, Dogri, Konkani, Bodo, Mizo, Nepali, Tulu, and more

Technical report: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688)

## License

This project is licensed under the [Apache License 2.0](LICENSE).

**Note:** The bundled audio tokenizer (HiggsAudioV2) is licensed under the [Boson Higgs Audio 2 Community License](model/audio_tokenizer/LICENSE), which includes the following restrictions:
- **Attribution required:** Products and services using HiggsAudioV2 must prominently display "Built with Higgs Materials licensed from Boson AI USA, Inc."
- **Commercial use threshold:** If your product or service exceeds 100,000 annual active users, you must request an expanded license from Boson AI.
- **Output restriction:** You may not use outputs of the Higgs Materials to improve other large language models.

Please review the [full license](model/audio_tokenizer/LICENSE) before use.

## Acknowledgments

- [OmniVoice (k2-fsa)](https://github.com/k2-fsa/OmniVoice) -- the underlying TTS model with zero-shot voice cloning
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/) -- enabling efficient inference on Trainium and Inferentia2

