"""
OpenAI-compatible streaming TTS server for OmniVoice on NeuronCores.

Endpoint: POST /v1/audio/speech
Compatible with OpenAI's TTS API format.

Architecture:
  - OmniVoiceNeuronPipeline with neuronx_distributed TP=2 (both NeuronCores)
  - Thread-safe request queue for concurrent HTTP handling
  - Streaming audio response (chunked PCM/WAV/MP3)
  - Single worker thread processes requests sequentially on Neuron hardware
  - Zero-shot voice cloning via reference audio samples

Usage:
  python server.py [--port 8000] [--host 0.0.0.0]
"""

import os
import sys
import io
import time
import wave
import json
import logging
import argparse
import threading
import queue
import uuid
from concurrent.futures import Future

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, Response, jsonify, stream_with_context

logger = logging.getLogger("omnivoice-server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 24000
MAX_QUEUE_SIZE = 64
MAX_TEXT_LENGTH = 4096
TP_DEGREE = int(os.environ.get("TP_DEGREE", "2"))

_MODEL_REPO = "k2-fsa/OmniVoice"

def _resolve_model_dir():
    """Auto-resolve OmniVoice model directory."""
    env = os.environ.get("OMNIVOICE_MODEL_DIR")
    if env and os.path.exists(env):
        return env
    for candidate in [
        os.path.join(os.path.dirname(_PROJECT_ROOT), "model"),
        os.path.join(_PROJECT_ROOT, "model"),
        os.path.expanduser("~/OmniVoice"),
    ]:
        if os.path.exists(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
            return candidate
    return None

DEFAULT_MODEL_DIR = _resolve_model_dir()
DEFAULT_TRACE_DIR = os.environ.get(
    "OMNIVOICE_TRACE_DIR", os.path.join(_PROJECT_ROOT, "neuron_traces")
)
DEFAULT_VOICES_DIR = os.environ.get(
    "OMNIVOICE_VOICES_DIR", os.path.join(os.path.dirname(_PROJECT_ROOT), "voices")
)

# OpenAI voice names map to bundled reference audio samples
VOICE_TO_SAMPLE = {
    "alloy": "en_sample.wav",
    "echo": "en_sample.wav",
    "fable": "fr_sample.wav",
    "onyx": "de_sample.wav",
    "nova": "es_sample.wav",
    "shimmer": "ja-sample.wav",
    "ballad": "hi_sample.wav",
    "ash": "te_sample.wav",
    "coral": "kn_sample.wav",
    "sage": "ta_sample.wav",
    "vale": "mr_sample.wav",
    "verse": "bn_sample.wav",
    "breeze": "gu_sample.wav",
    "ember": "pa_sample.wav",
    "lumen": "ml_sample.wav",
}

# OmniVoice supports 600+ languages via lang_id codes.
# This map covers all ISO 639-1 codes from OmniVoice docs plus Indian languages.
# Any lang_id from https://github.com/k2-fsa/OmniVoice/blob/master/docs/lang_id_name_map.tsv
# is accepted -- unmapped codes are passed through directly.
LANGUAGE_MAP = {
    # Major world languages
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ru": "Russian", "nl": "Dutch", "cs": "Czech", "ar": "Arabic",
    "zh": "Chinese", "hu": "Hungarian", "ko": "Korean", "ja": "Japanese",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish",
    "el": "Greek", "ro": "Romanian", "bg": "Bulgarian", "uk": "Ukrainian",
    "sr": "Serbian", "hr": "Croatian", "bs": "Bosnian", "sk": "Slovak",
    "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
    "is": "Icelandic", "mt": "Maltese", "sq": "Albanian", "mk": "Macedonian",
    "ka": "Georgian", "hy": "Armenian", "he": "Hebrew", "fa": "Persian",
    "az": "Azerbaijani", "kk": "Kazakh", "ky": "Kirghiz", "uz": "Uzbek",
    "tk": "Turkmen", "tt": "Tatar", "ba": "Bashkir", "mn": "Mongolian",
    "my": "Burmese", "km": "Khmer", "lo": "Lao", "si": "Sinhala",
    "dv": "Dhivehi", "am": "Amharic", "ti": "Tigrinya", "om": "Oromo",
    "so": "Somali", "sw": "Swahili", "ha": "Hausa", "ig": "Igbo",
    "yo": "Yoruba", "zu": "Zulu", "xh": "Xhosa", "af": "Afrikaans",
    "sn": "Shona", "ny": "Chichewa", "rw": "Kinyarwanda", "lg": "Ganda",
    "wo": "Wolof", "eu": "Basque", "ca": "Catalan", "gl": "Galician",
    "oc": "Occitan", "cy": "Welsh", "ga": "Irish", "gv": "Manx",
    "br": "Breton", "eo": "Esperanto", "ia": "Interlingua", "fy": "Western Frisian",
    "lb": "Luxembourgish", "sc": "Sardinian", "an": "Aragonese",
    "mi": "Maori", "haw": "Hawaiian", "jv": "Javanese", "bo": "Tibetan",
    "sa": "Sanskrit", "ug": "Uighur", "ps": "Pushto", "ht": "Haitian",
    "be": "Belarusian", "cv": "Chuvash", "os": "Ossetic",
    "rm": "Romansh", "ln": "Lingala", "ff": "Fulah",
    "gn": "Guarani", "tn": "Tswana",
    # Indian languages (Scheduled + major regional)
    "hi": "Hindi", "te": "Telugu", "kn": "Kannada", "ta": "Tamil",
    "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati", "pa": "Panjabi",
    "ml": "Malayalam", "or": "Odia", "ory": "Odia", "as": "Assamese",
    "ur": "Urdu", "ks": "Kashmiri", "sd": "Sindhi", "mni": "Manipuri",
    "sat": "Santali", "mai": "Maithili", "bho": "Bhojpuri", "dgo": "Dogri",
    "knn": "Konkani", "gom": "Goan Konkani", "brx": "Bodo", "lus": "Mizo",
    "npi": "Nepali", "tcy": "Tulu", "anp": "Angika", "bns": "Bundeli",
    "bra": "Braj", "gbm": "Garhwali", "bjj": "Kanauji", "mtr": "Mewari",
    "noe": "Nimadi", "dty": "Dotyali", "hoj": "Hadothi", "jns": "Jaunsari",
    "bhb": "Bhili", "bft": "Balti", "trp": "Kok Borok", "kfe": "Kota",
    "kfk": "Kinnauri", "sip": "Sikkimese", "the": "Chitwania Tharu",
    "hno": "Northern Hindko", "skr": "Saraiki", "phr": "Pahari-Potwari",
    "gju": "Gujari", "khw": "Khowar", "scl": "Shina", "bsh": "Kati",
    "ks": "Kashmiri", "dcc": "Deccan",
    # Chinese varieties
    "yue": "Cantonese", "nan": "Min Nan Chinese",
    # Arabic varieties
    "arb": "Standard Arabic", "arz": "Egyptian Arabic", "ary": "Moroccan Arabic",
    "ars": "Najdi Arabic", "acm": "Mesopotamian Arabic", "apc": "Levantine Arabic",
    "afb": "Gulf Arabic", "acw": "Hijazi Arabic",
    # Other notable languages
    "ceb": "Cebuano", "fil": "Filipino", "kab": "Kabyle", "ckb": "Central Kurdish",
    "kmr": "Northern Kurdish", "pcm": "Nigerian Pidgin", "tok": "Toki Pona",
}


def resolve_voice(voice: str, voices_dir: str) -> str:
    """Map OpenAI voice name to reference audio file path."""
    v = voice.strip().lower()

    if v in VOICE_TO_SAMPLE:
        sample_file = VOICE_TO_SAMPLE[v]
        for base in [voices_dir, os.path.join(DEFAULT_MODEL_DIR or "", "samples")]:
            path = os.path.join(base, sample_file)
            if os.path.exists(path):
                return path

    if voices_dir:
        for ext in ["", ".wav", ".mp3", ".flac"]:
            path = os.path.join(voices_dir, v + ext)
            if os.path.exists(path):
                return path

    if os.path.exists(voice):
        return voice

    for base in [voices_dir, os.path.join(DEFAULT_MODEL_DIR or "", "samples")]:
        path = os.path.join(base, "en_sample.wav")
        if os.path.exists(path):
            return path

    return voice


def resolve_language(lang: str) -> str:
    """Map language code to OmniVoice language name."""
    lang = lang.strip().lower()
    return LANGUAGE_MAP.get(lang, lang)


# ============================================================================
# Pipeline Manager
# ============================================================================

class PipelineManager:
    """Thread-safe manager for OmniVoiceNeuronPipeline.

    TP=2 uses both NeuronCores, so only one inference runs at a time.
    Requests are queued and processed sequentially by a worker thread.
    """

    def __init__(self, model_dir, trace_dir, voices_dir, tp_degree=2,
                 bucket_sizes=None, num_steps=8, force_trace=False):
        self._pipeline = None
        self._model_dir = model_dir
        self._trace_dir = trace_dir
        self._voices_dir = voices_dir
        self._tp_degree = tp_degree
        self._bucket_sizes = bucket_sizes or [256, 512, 768, 1024]
        self._num_steps = num_steps
        self._force_trace = force_trace
        self._work_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._worker_thread = None
        self._ready = threading.Event()
        self._shutting_down = False
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_audio_seconds": 0.0,
            "total_inference_seconds": 0.0,
        }

    def start(self):
        """Start worker thread and wait for pipeline to load."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._worker_thread.start()
        self._ready.wait()

    def precache_voices(self):
        """Pre-cache voice clone prompts for default voices."""
        voices_dir = self._voices_dir
        cached = 0
        for voice_name, sample_file in VOICE_TO_SAMPLE.items():
            ref_path = None
            for base in [voices_dir, os.path.join(self._model_dir or "", "samples")]:
                path = os.path.join(base, sample_file)
                if os.path.exists(path):
                    ref_path = path
                    break
            if ref_path and self._pipeline:
                try:
                    self._pipeline.cache_voice_prompt(ref_path)
                    cached += 1
                except Exception as e:
                    logger.warning("Failed to pre-cache voice %s: %s", voice_name, e)
        if cached:
            logger.info("Pre-cached %d/%d default voices", cached, len(VOICE_TO_SAMPLE))

    def _worker_loop(self):
        """Load pipeline, then process queued requests."""
        logger.info("Worker: loading OmniVoiceNeuronPipeline (TP=%d)...", self._tp_degree)
        t0 = time.perf_counter()

        from omnivoice_neuron_pipeline import OmniVoiceNeuronPipeline
        self._pipeline = OmniVoiceNeuronPipeline(
            model_dir=self._model_dir,
            trace_dir=self._trace_dir,
            bucket_sizes=self._bucket_sizes,
            force_trace=self._force_trace,
            tp_degree=self._tp_degree,
            num_steps=self._num_steps,
        )

        elapsed = time.perf_counter() - t0
        logger.info("Worker: pipeline ready in %.1fs", elapsed)
        self._ready.set()

        while not self._shutting_down:
            try:
                work_item = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if work_item is None:
                break

            if isinstance(work_item, tuple) and len(work_item) == 2 and work_item[0] == "streaming":
                _, task_fn = work_item
                task_fn()
                continue

            request_id, params, future = work_item
            try:
                with self._stats_lock:
                    self._stats["active_requests"] += 1
                result = self._run_inference(request_id, params)
                future.set_result(result)
                with self._stats_lock:
                    self._stats["active_requests"] -= 1
                    self._stats["completed_requests"] += 1
            except Exception as e:
                future.set_exception(e)
                with self._stats_lock:
                    self._stats["active_requests"] -= 1
                    self._stats["failed_requests"] += 1

    def _run_inference(self, request_id, params):
        """Execute a single TTS inference."""
        text = params["input"]
        ref_wav = params.get("ref_wav")
        language = params.get("language", "en")
        speed = params.get("speed", 1.0)

        logger.info("[%s] Generating: lang=%s ref=%s text=%s...",
                     request_id, language,
                     os.path.basename(ref_wav) if ref_wav else "none",
                     text[:80])
        t_start = time.perf_counter()

        wav = self._pipeline.infer(
            text=text,
            ref_audio=ref_wav if ref_wav and os.path.exists(ref_wav) else None,
            language=language,
            speed=speed,
        )

        t_end = time.perf_counter()
        inference_time = t_end - t_start
        sr = self._pipeline.SAMPLE_RATE
        audio_duration = len(wav) / sr if len(wav) > 0 else 0.0

        with self._stats_lock:
            self._stats["total_audio_seconds"] += audio_duration
            self._stats["total_inference_seconds"] += inference_time

        logger.info(
            "[%s] Done: %.2fs wall, %.2fs audio, RTF=%.1fx",
            request_id, inference_time, audio_duration,
            audio_duration / inference_time if inference_time > 0 else 0,
        )

        return {
            "wav": wav,
            "sample_rate": sr,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "request_id": request_id,
        }

    def _run_streaming_inference(self, request_id, params):
        """Execute TTS with chunk-level streaming."""
        text = params["input"]
        ref_wav = params.get("ref_wav")
        language = params.get("language", "en")
        speed = params.get("speed", 1.0)

        logger.info("[%s] Streaming: lang=%s ref=%s text=%s...",
                     request_id, language,
                     os.path.basename(ref_wav) if ref_wav else "none",
                     text[:80])
        t_start = time.perf_counter()
        t_first_byte = None
        total_samples = 0
        sr = self._pipeline.SAMPLE_RATE

        for wav_chunk in self._pipeline.infer_streaming(
            text=text,
            ref_audio=ref_wav if ref_wav and os.path.exists(ref_wav) else None,
            language=language,
            speed=speed,
        ):
            if len(wav_chunk) == 0:
                continue

            if t_first_byte is None:
                t_first_byte = time.perf_counter() - t_start
                logger.info("[%s] TTFA: %.1fms", request_id, t_first_byte * 1000)

            total_samples += len(wav_chunk)
            yield wav_chunk

        t_end = time.perf_counter()
        inference_time = t_end - t_start
        audio_duration = total_samples / sr if total_samples > 0 else 0.0

        with self._stats_lock:
            self._stats["total_audio_seconds"] += audio_duration
            self._stats["total_inference_seconds"] += inference_time
            self._stats["completed_requests"] += 1
            self._stats["active_requests"] -= 1

        logger.info(
            "[%s] Stream done: %.2fs wall, %.2fs audio, TTFA=%.1fms",
            request_id, inference_time, audio_duration,
            (t_first_byte or 0.0) * 1000,
        )

    def submit(self, params):
        """Submit a blocking TTS request. Returns (Future, request_id)."""
        request_id = str(uuid.uuid4())[:8]
        future = Future()

        with self._stats_lock:
            self._stats["total_requests"] += 1

        try:
            self._work_queue.put_nowait((request_id, params, future))
        except queue.Full:
            future.set_exception(
                RuntimeError("Server overloaded: request queue full")
            )

        return future, request_id

    def submit_streaming(self, params):
        """Submit a streaming TTS request. Returns (generator, request_id)."""
        request_id = str(uuid.uuid4())[:8]
        chunk_queue = queue.Queue(maxsize=64)

        with self._stats_lock:
            self._stats["total_requests"] += 1
            self._stats["active_requests"] += 1

        def _worker_task():
            try:
                for chunk in self._run_streaming_inference(request_id, params):
                    chunk_queue.put(("chunk", chunk))
                chunk_queue.put(("done", None))
            except Exception as e:
                chunk_queue.put(("error", str(e)))

        try:
            self._work_queue.put_nowait(("streaming", _worker_task))
        except queue.Full:
            with self._stats_lock:
                self._stats["active_requests"] -= 1
                self._stats["failed_requests"] += 1
            raise RuntimeError("Server overloaded: request queue full")

        def _chunk_generator():
            while True:
                msg_type, data = chunk_queue.get()
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise RuntimeError(f"Inference error: {data}")
                elif msg_type == "chunk":
                    yield data

        return _chunk_generator(), request_id

    def get_stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def get_queue_size(self):
        return self._work_queue.qsize()

    def is_ready(self):
        return self._ready.is_set()

    def get_languages(self):
        if self._pipeline:
            return self._pipeline.get_supported_languages()
        return []

    def get_voices_dir(self):
        return self._voices_dir

    def shutdown(self):
        self._shutting_down = True
        self._work_queue.put(None)


# ============================================================================
# Audio Encoding
# ============================================================================

def encode_pcm16(wav_np):
    """float32 [-1,1] -> raw PCM16 s16le bytes (24kHz mono)."""
    pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm16.tobytes()


def encode_wav(wav_np, sample_rate=SAMPLE_RATE):
    """float32 -> complete WAV file bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def encode_mp3(wav_np, sample_rate=SAMPLE_RATE):
    """float32 -> MP3 bytes (uses lameenc, no ffmpeg needed)."""
    import lameenc
    pcm16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(pcm16.tobytes())
    mp3_data += encoder.flush()
    return bytes(mp3_data)


# ============================================================================
# Flask Application
# ============================================================================

app = Flask(__name__)
pipeline_manager: PipelineManager = None


@app.route("/v1/audio/speech", methods=["POST"])
def create_speech():
    """OpenAI-compatible TTS endpoint.

    Request body:
        {
            "model": "tts-1",           // ignored (always uses OmniVoice)
            "input": "Hello world",      // required: text to synthesize
            "voice": "alloy",            // optional: voice name or ref audio file
            "response_format": "pcm",    // optional: pcm, wav, mp3
            "speed": 1.0,               // optional: playback speed
            "stream": true,              // optional: streaming mode
            "language": "en"             // optional: language code
        }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return _error_response("Invalid JSON", "invalid_request_error", 400)

    text = data.get("input", "").strip()
    if not text:
        return _error_response("'input' is required", "invalid_request_error", 400)
    if len(text) > MAX_TEXT_LENGTH:
        return _error_response(
            f"Input too long (max {MAX_TEXT_LENGTH} chars)",
            "invalid_request_error", 400,
        )

    voice = data.get("voice", "alloy")
    response_format = data.get("response_format", "pcm")
    stream = data.get("stream", True)
    language = data.get("language", "en")
    speed = float(data.get("speed", 1.0))

    ref_wav = resolve_voice(voice, pipeline_manager.get_voices_dir())
    lang_code = resolve_language(language)

    format_map = {
        "pcm": ("audio/pcm", encode_pcm16),
        "raw": ("audio/pcm", encode_pcm16),
        "wav": ("audio/wav", encode_wav),
        "mp3": ("audio/mpeg", encode_mp3),
    }
    if response_format not in format_map:
        return _error_response(
            f"Unsupported response_format: {response_format}. Use: pcm, wav, mp3",
            "invalid_request_error", 400,
        )
    content_type, encoder = format_map[response_format]

    params = {
        "input": text,
        "ref_wav": ref_wav,
        "language": lang_code,
        "speed": speed,
    }

    t_queued = time.perf_counter()

    if stream:
        try:
            chunk_gen, req_id = pipeline_manager.submit_streaming(params)
        except RuntimeError as e:
            return _error_response(str(e), "server_error", 503)

        if response_format == "mp3":
            import lameenc

            def generate():
                enc = lameenc.Encoder()
                enc.set_bit_rate(128)
                enc.set_in_sample_rate(SAMPLE_RATE)
                enc.set_channels(1)
                enc.set_quality(2)
                for wav_chunk in chunk_gen:
                    pcm16 = (np.clip(wav_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    mp3_bytes = enc.encode(pcm16.tobytes())
                    if mp3_bytes:
                        yield bytes(mp3_bytes)
                tail = enc.flush()
                if tail:
                    yield bytes(tail)

        elif response_format == "wav":
            def generate():
                all_pcm = bytearray()
                for wav_chunk in chunk_gen:
                    pcm16 = (np.clip(wav_chunk, -1.0, 1.0) * 32767).astype(np.int16)
                    all_pcm.extend(pcm16.tobytes())
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(bytes(all_pcm))
                yield buf.getvalue()

        else:
            def generate():
                for wav_chunk in chunk_gen:
                    yield encode_pcm16(wav_chunk)

        headers = {
            "X-Request-Id": req_id,
            "X-Queue-Time": f"{time.perf_counter() - t_queued:.3f}",
            "X-Sample-Rate": str(SAMPLE_RATE),
        }
        return Response(
            stream_with_context(generate()),
            mimetype=content_type,
            headers=headers,
        )
    else:
        future, req_id = pipeline_manager.submit(params)

        try:
            result = future.result(timeout=300)
        except queue.Full:
            return _error_response("Server overloaded", "server_error", 503)
        except Exception as e:
            return _error_response(str(e), "server_error", 500)

        wav = result["wav"]

        if response_format == "wav":
            audio_bytes = encode_wav(wav, result["sample_rate"])
        elif response_format == "mp3":
            audio_bytes = encode_mp3(wav, result["sample_rate"])
        else:
            audio_bytes = encode_pcm16(wav)

        headers = {
            "X-Request-Id": req_id,
            "X-Inference-Time": f"{result['inference_time']:.3f}",
            "X-Audio-Duration": f"{result['audio_duration']:.3f}",
            "X-Queue-Time": f"{time.perf_counter() - t_queued:.3f}",
            "X-Sample-Rate": str(result["sample_rate"]),
        }
        return Response(audio_bytes, mimetype=content_type, headers=headers)


@app.route("/v1/audio/voices", methods=["GET"])
def list_voices():
    """List available voices and languages."""
    voices = []

    for oai_name, sample_file in VOICE_TO_SAMPLE.items():
        voices.append({
            "voice_id": oai_name,
            "name": oai_name,
            "ref_audio": sample_file,
            "type": "builtin",
        })

    voices_dir = pipeline_manager.get_voices_dir()
    if voices_dir and os.path.isdir(voices_dir):
        for f in sorted(os.listdir(voices_dir)):
            if f.endswith((".wav", ".mp3", ".flac")):
                name = os.path.splitext(f)[0]
                if name.lower() not in VOICE_TO_SAMPLE:
                    voices.append({
                        "voice_id": name,
                        "name": name,
                        "ref_audio": f,
                        "type": "custom",
                    })

    return jsonify({
        "voices": voices,
        "languages": pipeline_manager.get_languages(),
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    stats = pipeline_manager.get_stats()
    return jsonify({
        "status": "ready" if pipeline_manager.is_ready() else "loading",
        "queue_depth": pipeline_manager.get_queue_size(),
        **stats,
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Detailed performance metrics."""
    stats = pipeline_manager.get_stats()
    completed = stats["completed_requests"]
    avg_inference = (
        stats["total_inference_seconds"] / completed if completed > 0 else 0
    )
    avg_audio = (
        stats["total_audio_seconds"] / completed if completed > 0 else 0
    )
    avg_rtf = (
        stats["total_audio_seconds"] / stats["total_inference_seconds"]
        if stats["total_inference_seconds"] > 0 else 0
    )
    return jsonify({
        **stats,
        "queue_depth": pipeline_manager.get_queue_size(),
        "avg_inference_time": round(avg_inference, 3),
        "avg_audio_duration": round(avg_audio, 3),
        "avg_rtf": round(avg_rtf, 1),
    })


@app.route("/", methods=["GET"])
def index():
    """Service info."""
    return jsonify({
        "service": "OmniVoice Neuron Streaming Server",
        "version": "1.0",
        "model": "k2-fsa/OmniVoice",
        "accelerator": f"neuronx_distributed TP={TP_DEGREE}",
        "api": "OpenAI-compatible",
        "endpoints": [
            "POST /v1/audio/speech",
            "GET  /v1/audio/voices",
            "GET  /health",
            "GET  /metrics",
        ],
    })


def _error_response(message, error_type, status_code):
    """Return an OpenAI-compatible error response."""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
        }
    }), status_code


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OmniVoice OpenAI-compatible Neuron Streaming Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Path to OmniVoice model directory",
    )
    parser.add_argument(
        "--trace-dir", default=DEFAULT_TRACE_DIR,
        help="Path to Neuron trace cache",
    )
    parser.add_argument(
        "--voices-dir", default=DEFAULT_VOICES_DIR,
        help="Path to reference voice audio samples",
    )
    parser.add_argument(
        "--tp-degree", type=int, default=TP_DEGREE,
        help="Tensor parallelism degree (default: 2)",
    )
    parser.add_argument(
        "--buckets", default="256,512,768,1024",
        help="Comma-separated backbone bucket sizes",
    )
    parser.add_argument(
        "--num-steps", type=int, default=8,
        help="Number of denoising steps (default: 8 balanced; 2=fastest, 32=best quality)",
    )
    parser.add_argument(
        "--force-trace", action="store_true",
        help="Force retracing of Neuron models (use after code changes)",
    )
    args = parser.parse_args()

    if not args.model_dir or not os.path.exists(args.model_dir):
        logger.error(
            "Model directory not found: %s\n"
            "Download the model first:\n"
            "  ./download_model.sh\n"
            "  # or: huggingface-cli download k2-fsa/OmniVoice --local-dir ./model",
            args.model_dir,
        )
        sys.exit(1)

    bucket_sizes = [int(b) for b in args.buckets.split(",")]

    global pipeline_manager
    pipeline_manager = PipelineManager(
        model_dir=args.model_dir,
        trace_dir=args.trace_dir,
        voices_dir=args.voices_dir,
        tp_degree=args.tp_degree,
        bucket_sizes=bucket_sizes,
        num_steps=args.num_steps,
        force_trace=args.force_trace,
    )

    logger.info("Starting pipeline initialization...")
    pipeline_manager.start()
    pipeline_manager.precache_voices()

    languages = pipeline_manager.get_languages()
    logger.info("Available languages: %s", languages[:10])
    logger.info("Voices directory: %s", args.voices_dir)
    logger.info(
        "Server starting on %s:%d (TP=%d, buckets=%s, steps=%d)",
        args.host, args.port, args.tp_degree, bucket_sizes, args.num_steps,
    )

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
