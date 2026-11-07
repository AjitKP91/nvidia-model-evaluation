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


_SCHEMA_PROBE_TEXT = "The quick brown fox jumps over the lazy dog."


def _speech_probe_clip(stt_client: STTClient, tts_client: TTSClient,
                       out_dir: Path) -> tuple[bytes, int, Path]:
    """Synthesize a short known utterance for schema discovery.

    The schema probes ask "does the response carry a transcript / words /
    finals field?" — that can only be answered with *speech*. A sine wave
    transcribes to nothing, so the content-dependent fields come back empty
    and the schema looks (misleadingly) unsupported. We synthesize a real
    sentence via the live TTS endpoint and transcribe that instead. Falls
    back to a sine only if TTS is unavailable (recorded as the degraded case).

    Returns ``(audio_bytes, sample_rate, wav_path)`` — bytes for the batch
    probe, the WAV path for the streaming probe (which reads a file).
    """
    wav_path = out_dir / "speech_probe.wav"
    try:
        synth = tts_client.synthesize_batch(_SCHEMA_PROBE_TEXT)
        audio_bytes = synth["audio_bytes"]
        sr = tts_client.sample_rate
        arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sf.write(str(wav_path), arr, sr)
        return audio_bytes, sr, wav_path
    except Exception as e:
        logger.warning("Schema-probe TTS failed, falling back to sine: %s", e)
        sine_path = out_dir / "sine_test.wav"
        if not sine_path.exists():
            _make_sine_wav(sine_path, duration_s=5.0, sr=16000)
        audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))
        return audio_bytes, sr, sine_path


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

def discover_stt_schema(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.2 STT Schema Discovery ===")

    # Use real synthesized speech, not a sine — the sine yields an empty
    # transcript and no words, which makes the response schema look
    # unsupported when it is not.
    audio_bytes, sr, clip_path = _speech_probe_clip(stt_client, tts_client, out_dir)
    is_speech = clip_path.name != "sine_test.wav"

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
        "probe_audio": "speech" if is_speech else "sine_fallback",
    }

    logger.info("STT batch schema: %s", json.dumps(schema, indent=2))
    return schema


def discover_stt_streaming_schema(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    logger.info("=== 0.2b STT Streaming Schema Discovery ===")

    _, _, clip_path = _speech_probe_clip(stt_client, tts_client, out_dir)
    is_speech = clip_path.name != "sine_test.wav"

    result = stt_client.stream_recognize(str(clip_path))

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
        "probe_audio": "speech" if is_speech else "sine_fallback",
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

    # SSML test — acceptance AND functional effect. A <break> should lengthen
    # the audio vs the same words with no tag; a longer clip is the honored
    # signal (acceptance alone, L218-era, didn't prove the gateway did anything).
    try:
        plain_text = "<speak>Hello world.</speak>"
        ssml_text = '<speak>Hello <break time="1s"/> world.</speak>'
        plain = tts_client.synthesize_batch(plain_text)
        variant = tts_client.synthesize_batch(ssml_text)
        plain_dur = plain["audio_duration"]
        ssml_dur = variant["audio_duration"]
        ratio = ssml_dur / plain_dur if plain_dur > 0 else None
        accepted = ssml_dur > 0
        effect = ratio is not None and ratio >= 1.15
        results["tts_ssml"] = {
            "acceptance": "supported" if accepted else "no_audio",
            "plain_duration_s": round(plain_dur, 3),
            "ssml_duration_s": round(ssml_dur, 3),
            "duration_ratio": round(ratio, 3) if ratio else None,
            "break_effect": "honored" if effect else ("accepted_no_effect" if accepted else "no_audio"),
        }
    except Exception as e:
        results["tts_ssml"] = {"acceptance": f"error: {e}", "break_effect": "error"}

    logger.info("Parameters: %s", json.dumps(results, indent=2, default=str))
    return results


# ------------------------------------------------------------------
# 0.4b  STT Serving-Feature Acceptance Snapshot
# ------------------------------------------------------------------

def probe_stt_features(stt_client: STTClient, tts_client: TTSClient, out_dir: Path) -> dict:
    """Fast per-flag acceptance snapshot for the Riva STT serving features.

    One recognize call per feature flag against a short synthesized utterance,
    recording two layers: ``accepted`` (the client set the field and the call
    returned without error) and ``populated`` (the response actually carried the
    corresponding data — word times, >1 alternative, a speaker_tag). The scored
    Tests 1.14–1.18 do the depth; this is the cheap gateway-honor glance that
    lives in discovery.json. A real spoken clip is used so the "populated" check
    is meaningful (a sine wave yields no words).
    """
    logger.info("=== 0.4b STT Feature Acceptance Snapshot ===")
    features: dict = {}

    # A short real utterance so word/alternative/speaker fields can populate.
    try:
        synth = tts_client.synthesize_batch(
            "The quick brown fox jumps over the lazy dog."
        )
        audio_bytes = synth["audio_bytes"]
        sr = tts_client.sample_rate
    except Exception as e:
        logger.warning("Feature-probe TTS failed, falling back to sine: %s", e)
        sine_path = out_dir / "sine_test.wav"
        if not sine_path.exists():
            _make_sine_wav(sine_path, duration_s=3.0, sr=16000)
        audio_bytes, sr = stt_client.audio_to_bytes(str(sine_path))

    def _probe(name: str, populated_fn, **kwargs) -> None:
        try:
            resp = stt_client.recognize_batch(audio_bytes, sr, **kwargs)
            features[name] = {"accepted": True, "populated": bool(populated_fn(resp))}
        except Exception as e:
            features[name] = {"accepted": False, "error": str(e)[:200]}

    _probe(
        "word_timestamps",
        lambda r: any((w.get("start_time") or 0) > 0 or (w.get("end_time") or 0) > 0
                      for w in r.get("words", [])),
        enable_word_times=True,
    )
    _probe(
        "nbest",
        lambda r: len(r.get("alternatives", [])) > 1,
        max_alternatives=5,
    )
    _probe(
        "word_boosting",
        lambda r: bool(r.get("transcript")),
        speech_contexts=[{"phrases": ["Nemotron", "Magpie"], "boost": 20.0}],
    )
    _probe(
        "profanity_filter",
        lambda r: r.get("transcript") is not None,
        profanity_filter=True,
    )
    _probe(
        "diarization",
        lambda r: any(w.get("speaker_tag") is not None for w in r.get("words", [])),
        diarization=True, max_speaker_count=2,
    )

    logger.info("STT feature snapshot: %s", json.dumps(features, indent=2, default=str))
    return features


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

def _run_subphase(name: str, fn, out_dir: Path, results: dict, *args, **kwargs) -> None:
    """Run a sub-phase and record its result (or full exception) in results[name].

    On success: results[name] is whatever fn returned.
    On failure: results[name] = {"phase": name, "error": "<str(exc)>",
                                 "error_type": "<ExcClass>",
                                 "traceback": "<full traceback>"}.
    The full traceback also lands in run.log via logger.exception so the raw
    RPC error, HTTP status, upstream envoy message, etc. are all preserved.

    Persists results/phase0/discovery.json after each sub-phase so a crash
    midway leaves everything up to that point on disk. Callers pass ``fn``
    plus its positional and keyword arguments verbatim.
    """
    import traceback as _tb
    try:
        results[name] = fn(*args, **kwargs)
    except Exception as e:
        tb_str = _tb.format_exc()
        logger.exception("Sub-phase %s FAILED: %s", name, e)
        results[name] = {
            "phase": name,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb_str,
        }
    # Persist incrementally — one file rewrite per sub-phase; cheap because
    # results is tiny (<50 KB). If Python crashes right after this, whatever
    # completed is on disk.
    _write_discovery(out_dir, results)


def _write_discovery(out_dir: Path, results: dict) -> None:
    """Atomic-ish write of discovery.json — write to .tmp then rename."""
    output_path = out_dir / "discovery.json"
    tmp_path = out_dir / "discovery.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp_path.replace(output_path)


def _language_acceptance_subphase(stt_client: STTClient, out_dir: Path) -> dict:
    """Probe every registered model language against the live endpoint (streamed
    FLEURS, ~3 clips each, idempotent). Shares its output dir with Test 1.13 so the
    "ok" verdicts are reused and the probe stays cheap on every later run."""
    from eval.stt.language_acceptance import probe

    # out_dir here is results/phase0; keep the matrix with the STT results.
    accept_dir = out_dir.parent / "stt" / "language_acceptance"
    result = probe(stt_client, accept_dir)
    return {"phase": "language_acceptance", **result["summary"],
            "matrix_csv": str(accept_dir / "language_acceptance.csv")}


def run(config: Config | None = None) -> dict:
    if config is None:
        config = load_config()

    setup_logging(config.evaluation.log_level)
    out_dir = Path(config.evaluation.results_dir) / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Client construction can itself fail (missing token, invalid config).
    # Wrap it so we still write a usable discovery.json instead of crashing.
    results: dict = {}
    try:
        stt_client = STTClient(config)
    except Exception as e:
        logger.exception("Failed to construct STTClient: %s", e)
        results["stt_client_init"] = {
            "phase": "stt_client_init",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        stt_client = None

    try:
        tts_client = TTSClient(config)
    except Exception as e:
        logger.exception("Failed to construct TTSClient: %s", e)
        results["tts_client_init"] = {
            "phase": "tts_client_init",
            "error": str(e),
            "error_type": type(e).__name__,
        }
        tts_client = None

    if stt_client is None or tts_client is None:
        _write_discovery(out_dir, results)
        logger.error(
            "Phase 0 aborted: could not construct client(s). "
            "Partial discovery.json written to %s", out_dir / "discovery.json",
        )
        return results

    # Each sub-phase is isolated — a failure records its error and moves on.
    _run_subphase("connectivity",           check_connectivity,           out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("stt_schema",             discover_stt_schema,          out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("stt_streaming_schema",   discover_stt_streaming_schema, out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("tts_schema",             discover_tts_schema,          out_dir, results, tts_client, out_dir)
    _run_subphase("parameters",             explore_parameters,           out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("stt_features",           probe_stt_features,           out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("smoke_tests",            smoke_tests,                  out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("cold_start",             cold_start_test,              out_dir, results, stt_client, tts_client, out_dir)
    _run_subphase("language_acceptance",    _language_acceptance_subphase, out_dir, results, stt_client, out_dir)

    logger.info("Phase 0 results written to %s", out_dir / "discovery.json")

    # Summary line so the operator doesn't have to open the JSON to know
    # which sub-phases passed. Failed ones already logged full tracebacks.
    passed, failed = [], []
    for name, r in results.items():
        if isinstance(r, dict) and r.get("error"):
            failed.append(name)
        else:
            passed.append(name)
    logger.info("Phase 0 summary: %d passed, %d failed", len(passed), len(failed))
    if failed:
        logger.info("  Failed sub-phases: %s", ", ".join(failed))
        logger.info("  See discovery.json (error/traceback fields) and run.log for details.")

    for wav in out_dir.glob("*.wav"):
        wav.unlink(missing_ok=True)
    logger.info("Phase 0 temporary WAV files deleted.")

    return results


if __name__ == "__main__":
    run()
