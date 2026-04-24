"""Phase 0 — Discovery & Infrastructure.

Verify connectivity, confirm response schema, determine supported audio
formats, and establish baseline numbers for all subsequent tests.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.config import Config, load_config
from eval.stt.client import STTClient
from eval.tts.client import TTSClient
from eval.utils import write_jsonl, setup_logging

logger = logging.getLogger("eval.phase0")

SMOKE_AUDIO_TEXT = "This is a short test sentence for smoke testing."


def _make_sine_wav(path: Path, duration_s: float = 2.0, sr: int = 16000) -> None:
    """Generate a short sine-wave WAV for connectivity checks."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), audio, sr)


# ------------------------------------------------------------------
# 0.1  Connectivity & Auth
# ------------------------------------------------------------------

def check_connectivity(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.1 Connectivity & Auth ===")
    results: dict = {"grpc_stt": None, "grpc_tts": None, "rest_stt": None, "rest_tts": None}

    sine_path = out_dir / "sine_test.wav"
    _make_sine_wav(sine_path, duration_s=2.0, sr=16000)
    audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))

    # gRPC STT
    try:
        t0 = time.perf_counter()
        resp = stt_client.recognize_batch(audio_bytes, sr)
        latency = time.perf_counter() - t0
        results["grpc_stt"] = {"status": "ok", "latency_s": latency, "transcript_len": len(resp["transcript"])}
        logger.info("gRPC STT: OK (%.2fs)", latency)
    except Exception as e:
        results["grpc_stt"] = {"status": "error", "error": str(e)}
        logger.error("gRPC STT failed: %s", e)

    # REST STT
    try:
        t0 = time.perf_counter()
        resp = stt_client.recognize_batch_rest(audio_bytes, sr)
        latency = time.perf_counter() - t0
        results["rest_stt"] = {"status": "ok", "latency_s": latency}
        logger.info("REST STT: OK (%.2fs)", latency)
    except Exception as e:
        results["rest_stt"] = {"status": "error", "error": str(e)}
        logger.error("REST STT failed: %s", e)

    # gRPC TTS
    try:
        t0 = time.perf_counter()
        resp = tts_client.synthesize_batch("Hello, testing connectivity.")
        latency = time.perf_counter() - t0
        results["grpc_tts"] = {"status": "ok", "latency_s": latency, "audio_len_bytes": len(resp["audio_bytes"])}
        logger.info("gRPC TTS: OK (%.2fs)", latency)
    except Exception as e:
        results["grpc_tts"] = {"status": "error", "error": str(e)}
        logger.error("gRPC TTS failed: %s", e)

    # REST TTS
    try:
        t0 = time.perf_counter()
        resp = tts_client.synthesize_batch_rest("Hello, testing connectivity.")
        latency = time.perf_counter() - t0
        results["rest_tts"] = {"status": "ok", "latency_s": latency}
        logger.info("REST TTS: OK (%.2fs)", latency)
    except Exception as e:
        results["rest_tts"] = {"status": "error", "error": str(e)}
        logger.error("REST TTS failed: %s", e)

    # Network baseline
    try:
        import requests as req
        t0 = time.perf_counter()
        req.head(stt_client.stt_cfg.rest_endpoint, timeout=10)
        results["network_rtt_s"] = time.perf_counter() - t0
    except Exception:
        results["network_rtt_s"] = None

    return results


# ------------------------------------------------------------------
# 0.2  STT Schema Discovery
# ------------------------------------------------------------------

def discover_stt_schema(stt_client: STTClient, out_dir: Path) -> dict:
    logger.info("=== 0.2 STT Schema Discovery ===")

    sine_path = out_dir / "sine_test.wav"
    if not sine_path.exists():
        _make_sine_wav(sine_path, duration_s=5.0, sr=16000)
    audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))

    result = stt_client.recognize_batch(audio_bytes, sr)

    schema = {
        "has_transcript": bool(result.get("transcript")),
        "has_confidence": result.get("confidence") is not None,
        "has_words": bool(result.get("words")),
        "word_fields": (
            list(result["words"][0].keys()) if result.get("words") else []
        ),
        "sample_transcript": result.get("transcript", "")[:200],
        "batch_latency_s": result.get("elapsed_s"),
    }

    logger.info("STT batch schema: %s", json.dumps(schema, indent=2))
    return schema


def discover_stt_streaming_schema(stt_client: STTClient, out_dir: Path) -> dict:
    logger.info("=== 0.2b STT Streaming Schema Discovery ===")

    sine_path = out_dir / "sine_test.wav"
    if not sine_path.exists():
        _make_sine_wav(sine_path, duration_s=5.0, sr=16000)

    result = stt_client.stream_recognize(str(sine_path))

    schema = {
        "has_partials": bool(result.get("partials")),
        "num_partials": len(result.get("partials", [])),
        "has_finals": bool(result.get("finals")),
        "has_stability": (
            result["partials"][0].get("stability") is not None
            if result.get("partials")
            else False
        ),
        "ttfw": result.get("ttfw"),
        "finalization_latency": result.get("finalization_latency"),
        "streaming_rtf": result.get("streaming_rtf"),
        "final_transcript": result.get("transcript", "")[:200],
    }

    logger.info("STT streaming schema: %s", json.dumps(schema, indent=2, default=str))
    return schema


# ------------------------------------------------------------------
# 0.3  TTS Schema Discovery
# ------------------------------------------------------------------

def discover_tts_schema(tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.3 TTS Schema Discovery ===")

    text = "Hello, this is a test."

    # Batch gRPC
    batch_result = tts_client.synthesize_batch(text)
    audio_bytes = batch_result["audio_bytes"]

    # Streaming gRPC
    stream_result = tts_client.synthesize_stream(text)

    # Save sample audio
    sample_path = out_dir / "tts_sample.wav"
    audio_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(str(sample_path), audio_arr, tts_client.sample_rate)

    schema = {
        "batch_audio_bytes": len(audio_bytes),
        "batch_audio_duration_s": batch_result["audio_duration"],
        "batch_latency_s": batch_result["elapsed_s"],
        "sample_rate": tts_client.sample_rate,
        "stream_ttfb": stream_result.get("ttfb"),
        "stream_n_chunks": stream_result.get("n_chunks"),
        "stream_audio_duration_s": stream_result["audio_duration"],
        "stream_total_latency_s": stream_result["elapsed_s"],
        "sample_saved": str(sample_path),
    }

    logger.info("TTS schema: %s", json.dumps(schema, indent=2, default=str))
    return schema


# ------------------------------------------------------------------
# 0.4  Parameter Exploration
# ------------------------------------------------------------------

def explore_parameters(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.4 Parameter Exploration ===")
    results: dict = {"stt_sample_rates": {}, "tts_ssml": None}

    sine_16k = out_dir / "sine_test.wav"
    if not sine_16k.exists():
        _make_sine_wav(sine_16k, duration_s=2.0, sr=16000)

    for test_sr in [8000, 16000, 22050, 44100, 48000]:
        sine_path = out_dir / f"sine_{test_sr}.wav"
        _make_sine_wav(sine_path, duration_s=2.0, sr=test_sr)
        audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))
        try:
            resp = stt_client.recognize_batch(audio_bytes, sr)
            results["stt_sample_rates"][test_sr] = "ok"
        except Exception as e:
            results["stt_sample_rates"][test_sr] = f"error: {e}"

    # SSML test
    try:
        ssml_text = '<speak>Hello <break time="500ms"/> world.</speak>'
        resp = tts_client.synthesize_batch(ssml_text)
        results["tts_ssml"] = "supported" if resp["audio_duration"] > 0 else "no_audio"
    except Exception as e:
        results["tts_ssml"] = f"error: {e}"

    logger.info("Parameters: %s", json.dumps(results, indent=2))
    return results


# ------------------------------------------------------------------
# 0.5  Smoke Tests
# ------------------------------------------------------------------

def smoke_tests(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.5 Smoke Tests ===")
    results = {}

    # STT 10s audio
    sine_10s = out_dir / "sine_10s.wav"
    _make_sine_wav(sine_10s, duration_s=10.0, sr=16000)
    audio_bytes, sr = stt_client.audio_to_bytes(str(sine_10s))
    try:
        resp = stt_client.recognize_batch(audio_bytes, sr)
        results["stt_10s"] = {
            "status": "pass" if resp["elapsed_s"] < 30 else "fail",
            "latency_s": resp["elapsed_s"],
        }
    except Exception as e:
        results["stt_10s"] = {"status": "fail", "error": str(e)}

    # STT 60s audio
    sine_60s = out_dir / "sine_60s.wav"
    _make_sine_wav(sine_60s, duration_s=60.0, sr=16000)
    audio_bytes_60, sr = stt_client.audio_to_bytes(str(sine_60s))
    try:
        resp = stt_client.recognize_batch(audio_bytes_60, sr)
        results["stt_60s"] = {"status": "pass", "latency_s": resp["elapsed_s"]}
    except Exception as e:
        results["stt_60s"] = {"status": "fail", "error": str(e)}

    # TTS 20 words (REST)
    try:
        resp = tts_client.synthesize_batch_rest(
            "The quick brown fox jumps over the lazy dog near the river bank on a warm sunny afternoon."
        )
        dur = resp["audio_duration"]
        results["tts_rest"] = {
            "status": "pass" if dur > 1.0 else "fail",
            "audio_duration_s": dur,
        }
    except Exception as e:
        results["tts_rest"] = {"status": "fail", "error": str(e)}

    # TTS gRPC streaming
    try:
        resp = tts_client.synthesize_stream(
            "The quick brown fox jumps over the lazy dog near the river bank on a warm sunny afternoon."
        )
        results["tts_grpc_stream"] = {
            "status": "pass" if resp.get("ttfb") and resp["ttfb"] < 5.0 else "fail",
            "ttfb_s": resp.get("ttfb"),
        }
    except Exception as e:
        results["tts_grpc_stream"] = {"status": "fail", "error": str(e)}

    for name, r in results.items():
        logger.info("Smoke %s: %s", name, r["status"])

    return results


# ------------------------------------------------------------------
# 0.6  Cold-Start vs Warm Latency
# ------------------------------------------------------------------

def cold_start_test(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.6 Cold-Start Test ===")
    logger.info(
        "NOTE: For accurate cold-start measurement, run this immediately after "
        "a fresh deployment or after >30 min idle."
    )

    sine_path = out_dir / "sine_10s.wav"
    if not sine_path.exists():
        _make_sine_wav(sine_path, duration_s=10.0, sr=16000)
    audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))

    results: dict = {}

    # STT cold start
    t0 = time.perf_counter()
    stt_client.recognize_batch(audio_bytes, sr)
    t_first = time.perf_counter() - t0

    warm_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        stt_client.recognize_batch(audio_bytes, sr)
        warm_times.append(time.perf_counter() - t0)

    t_warm = np.mean(warm_times)
    results["stt"] = {
        "t_first_s": t_first,
        "t_warm_mean_s": float(t_warm),
        "cold_ratio": t_first / t_warm if t_warm > 0 else None,
    }

    # TTS cold start
    text = "This is a cold start test sentence for the TTS model."
    t0 = time.perf_counter()
    tts_client.synthesize_batch(text)
    t_first = time.perf_counter() - t0

    warm_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        tts_client.synthesize_batch(text)
        warm_times.append(time.perf_counter() - t0)

    t_warm = np.mean(warm_times)
    results["tts"] = {
        "t_first_s": t_first,
        "t_warm_mean_s": float(t_warm),
        "cold_ratio": t_first / t_warm if t_warm > 0 else None,
    }

    for model, r in results.items():
        status = "PASS" if (r["cold_ratio"] or 999) < 5 else "FLAG"
        logger.info(
            "%s cold-start: first=%.2fs, warm=%.2fs, ratio=%.1f (%s)",
            model.upper(), r["t_first_s"], r["t_warm_mean_s"],
            r["cold_ratio"] or -1, status,
        )

    return results


# ------------------------------------------------------------------
# Run all Phase 0
# ------------------------------------------------------------------

def run(config: Config | None = None) -> dict:
    if config is None:
        config = load_config()

    setup_logging(config.evaluation.log_level)
    out_dir = Path(config.evaluation.results_dir) / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    tts_client = TTSClient(config)

    results = {}
    results["connectivity"] = check_connectivity(stt_client, tts_client, out_dir)
    results["stt_schema"] = discover_stt_schema(stt_client, out_dir)
    results["stt_streaming_schema"] = discover_stt_streaming_schema(stt_client, out_dir)
    results["tts_schema"] = discover_tts_schema(tts_client, out_dir)
    results["parameters"] = explore_parameters(stt_client, tts_client, out_dir)
    results["smoke_tests"] = smoke_tests(stt_client, tts_client, out_dir)
    results["cold_start"] = cold_start_test(stt_client, tts_client, out_dir)

    output_path = out_dir / "discovery.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Phase 0 results written to %s", output_path)

    return results


if __name__ == "__main__":
    run()
