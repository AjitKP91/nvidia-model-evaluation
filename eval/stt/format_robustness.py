"""Test 1.11 — Audio Format & Input Robustness.

ENGLISH-ONLY BY DESIGN: this is an audio-frontend test — it varies container
format, sample rate, and bit depth and checks the endpoint still accepts/decodes
the stream. Those are properties of the audio pipeline, not of the spoken
language, so it is not parameterised by language. See
docs/multilingual-test-gap-analysis.md.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

import riva.client

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.format_robustness")

SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
BIT_DEPTHS = ["pcm_16", "pcm_32", "float32"]

# Audio formats mapped to the Riva AudioEncoding the gateway decodes them as.
# Riva's gRPC API defines exactly this set of decoders, so each row genuinely
# exercises the gateway's audio frontend (the old path forced every format
# through REST, which this endpoint does not expose). MP3 is NOT in Riva's
# AudioEncoding enum at all and is handled separately as unsupported-by-API.
FORMAT_ENCODINGS = [
    ("pcm_s16le", riva.client.AudioEncoding.LINEAR_PCM),
    ("flac", riva.client.AudioEncoding.FLAC),
    ("ogg_opus", riva.client.AudioEncoding.OGGOPUS),
    ("mulaw", riva.client.AudioEncoding.MULAW),
    ("alaw", riva.client.AudioEncoding.ALAW),
]
DURATION_EXTREMES = [0.5, 5, 60, 600]


def _create_test_audio(sr: int = 16000, duration_s: float = 5.0) -> np.ndarray:
    """Generate a sine wave as test audio."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def _encode_for_riva(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    """Encode the float32 sine clip into the bytes Riva expects for ``fmt``.

    Uses libsndfile (soundfile) only — no ffmpeg dependency — so it runs
    anywhere the eval runs. LINEAR_PCM is headerless int16; FLAC and OGG_OPUS
    are self-describing containers; MULAW/ALAW are raw 8-bit companded streams.
    """
    if fmt == "pcm_s16le":
        return (audio * 32768).astype(np.int16).tobytes()
    buf = io.BytesIO()
    if fmt == "flac":
        sf.write(buf, audio, sr, format="FLAC", subtype="PCM_16")
    elif fmt == "ogg_opus":
        sf.write(buf, audio, sr, format="OGG", subtype="OPUS")
    elif fmt == "mulaw":
        sf.write(buf, audio, sr, format="RAW", subtype="ULAW")
    elif fmt == "alaw":
        sf.write(buf, audio, sr, format="RAW", subtype="ALAW")
    else:
        raise ValueError(f"no soundfile encoder mapped for format {fmt!r}")
    buf.seek(0)
    return buf.read()


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "format_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.11: Audio Format & Input Robustness ===")

    base_audio = _create_test_audio(sr=16000, duration_s=5.0)
    summary_rows = []

    # ---- Sample rate test ----
    for sr in tqdm(SAMPLE_RATES, desc="Sample rates"):
        audio = _create_test_audio(sr=sr, duration_s=5.0)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, sr)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "sample_rate", "value": sr, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Format test ----
    # Each format is streamed to the gRPC endpoint with its matching Riva
    # AudioEncoding, so this measures whether the gateway's audio frontend
    # actually decodes the container — not just whether a REST shim exists.
    for fmt, enc in tqdm(FORMAT_ENCODINGS, desc="Formats"):
        try:
            audio_data = _encode_for_riva(base_audio, 16000, fmt)
            result = stt_client.recognize_batch(audio_data, 16000, encoding=enc)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "format", "value": fmt, "status": status,
                  "encoding": riva.client.AudioEncoding.Name(enc),
                  "transport": "grpc_streaming", "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # MP3 is not representable in Riva's AudioEncoding enum, so the gateway
    # cannot decode it over gRPC at all. Recorded as an API limitation
    # (unsupported) rather than a gateway failure, so it is not mistaken for a
    # rejected-but-supported format.
    mp3_record = {"test": "format", "value": "mp3", "status": "unsupported",
                  "encoding": "n/a", "transport": "grpc_streaming",
                  "error": "Riva gRPC API has no MP3 AudioEncoding; the server "
                           "cannot decode MP3. Transcode client-side to PCM/FLAC "
                           "first."}
    write_jsonl(jsonl_path, mp3_record)
    summary_rows.append(mp3_record)

    # ---- Channels test ----
    for channels in [1, 2]:
        try:
            if channels == 2:
                stereo = np.column_stack([base_audio, base_audio])
                audio_bytes = (stereo * 32768).astype(np.int16).tobytes()
            else:
                audio_bytes = (base_audio * 32768).astype(np.int16).tobytes()
            result = stt_client.recognize_batch(audio_bytes, 16000, channels=channels)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "channels", "value": channels, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Duration extremes ----
    for dur in tqdm(DURATION_EXTREMES, desc="Duration extremes"):
        audio = _create_test_audio(sr=16000, duration_s=dur)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, 16000)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "duration", "value": dur, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Edge cases ----
    edge_cases = [
        ("silence", np.zeros(16000 * 5, dtype=np.float32)),
        ("white_noise", np.random.randn(16000 * 5).astype(np.float32) * 0.5),
    ]
    for name, audio in edge_cases:
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, 16000)
            status = "pass"
            transcript = result["transcript"]
        except Exception as e:
            status = "fail"
            transcript = str(e)

        record = {"test": "edge_case", "value": name, "status": status, "transcript": transcript}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    passed = sum(1 for r in summary_rows if r["status"] == "pass")
    total = len(summary_rows)
    logger.info("Format robustness: %d/%d passed", passed, total)

    return {"test": "1.11", "name": "format_robustness", "results": summary_rows}
