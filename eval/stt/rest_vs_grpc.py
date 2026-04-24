"""Test 1.4 — REST vs gRPC Comparison."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import compute_percentiles, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.rest_vs_grpc")

N_REPS = 50


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "rest_vs_grpc"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)

    logger.info("=== Test 1.4: REST vs gRPC Comparison ===")

    # Pick a 30s utterance
    ds = load_dataset("librispeech_asr", "clean", split="test", token=True)
    ref_example = None
    for ex in ds:
        dur = len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"]
        if 25 < dur < 35:
            ref_example = ex
            break

    if ref_example is None:
        ref_example = ds[0]

    audio_array = np.array(ref_example["audio"]["array"], dtype=np.float32)
    sr = ref_example["audio"]["sampling_rate"]
    audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

    # Use mkstemp to avoid NamedTemporaryFile dual-open conflict.
    # Write int16 PCM so wave.open() in stream_recognize can parse it.
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    sf.write(wav_path, audio_int16, sr, subtype="PCM_16")

    # ---- Batch comparison ----
    grpc_batch_latencies = []
    rest_batch_latencies = []

    logger.info("Batch comparison: %d reps per interface", N_REPS)
    for i in tqdm(range(N_REPS), desc="gRPC batch"):
        try:
            r = stt_client.recognize_batch(audio_bytes, sr)
            grpc_batch_latencies.append(r["elapsed_s"])
            write_jsonl(results_dir / "batch.jsonl", {"interface": "grpc", "rep": i, "elapsed_s": r["elapsed_s"]})
        except Exception as e:
            logger.warning("gRPC batch %d failed: %s", i, e)

    for i in tqdm(range(N_REPS), desc="REST batch"):
        try:
            r = stt_client.recognize_batch_rest(audio_bytes, sr)
            rest_batch_latencies.append(r["elapsed_s"])
            write_jsonl(results_dir / "batch.jsonl", {"interface": "rest", "rep": i, "elapsed_s": r["elapsed_s"]})
        except Exception as e:
            logger.warning("REST batch %d failed: %s", i, e)

    # ---- Streaming comparison ----
    grpc_stream_ttfw = []
    grpc_stream_final = []
    rest_stream_latencies = []

    logger.info("Streaming comparison: %d reps per interface", N_REPS)
    for i in tqdm(range(N_REPS), desc="gRPC streaming"):
        try:
            r = stt_client.stream_recognize(wav_path)
            if r["ttfw"] is not None:
                grpc_stream_ttfw.append(r["ttfw"])
            if r["finalization_latency"] is not None:
                grpc_stream_final.append(r["finalization_latency"])
            write_jsonl(results_dir / "streaming.jsonl", {
                "interface": "grpc", "rep": i,
                "ttfw": r["ttfw"], "finalization_latency": r["finalization_latency"],
            })
        except Exception as e:
            logger.warning("gRPC stream %d failed: %s", i, e)

    for i in tqdm(range(N_REPS), desc="REST streaming"):
        try:
            r = stt_client.stream_recognize_rest(wav_path)
            rest_stream_latencies.append(r["elapsed_s"])
            write_jsonl(results_dir / "streaming.jsonl", {
                "interface": "rest", "rep": i, "elapsed_s": r["elapsed_s"],
                "ttfb": r.get("ttfb"),
            })
        except Exception as e:
            logger.warning("REST stream %d failed: %s", i, e)

    Path(wav_path).unlink(missing_ok=True)

    # ---- Summary ----
    summary = []
    if grpc_batch_latencies:
        summary.append({"mode": "batch", "interface": "grpc", **compute_percentiles(grpc_batch_latencies)})
    if rest_batch_latencies:
        summary.append({"mode": "batch", "interface": "rest", **compute_percentiles(rest_batch_latencies)})
    if grpc_stream_ttfw:
        summary.append({"mode": "streaming_ttfw", "interface": "grpc", **compute_percentiles(grpc_stream_ttfw)})
    if grpc_stream_final:
        summary.append({"mode": "streaming_final", "interface": "grpc", **compute_percentiles(grpc_stream_final)})
    if rest_stream_latencies:
        summary.append({"mode": "streaming_total", "interface": "rest", **compute_percentiles(rest_stream_latencies)})

    save_summary_csv(results_dir / "summary.csv", summary)

    for row in summary:
        logger.info("%s %s: P50=%.3fs P99=%.3fs", row["mode"], row["interface"], row.get("p50", 0), row.get("p99", 0))

    return {"test": "1.4", "name": "rest_vs_grpc", "results": summary}
