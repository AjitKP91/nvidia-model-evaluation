"""Test 1.2 — Batch Processing Performance (RTF / RTFx)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import compute_percentiles, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.performance")

DURATION_BUCKETS = [10, 30, 60, 300, 600]
SAMPLES_PER_BUCKET = 20


def _pick_utterances_by_duration(ds, target_s: float, n: int, tolerance: float = 0.3):
    """Select utterances whose duration is close to target_s."""
    candidates = []
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        dur = len(audio["array"]) / audio["sampling_rate"]
        if abs(dur - target_s) / target_s < tolerance:
            candidates.append((i, ex, dur))
        if len(candidates) >= n * 3:
            break
    candidates.sort(key=lambda x: abs(x[2] - target_s))
    return candidates[:n]


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "performance"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.2: Batch Processing Performance ===")

    ds = load_dataset("librispeech_asr", "clean", split="test.clean", token=True)

    all_results = []

    for bucket_s in DURATION_BUCKETS:
        logger.info("Duration bucket: %d s", bucket_s)
        utterances = _pick_utterances_by_duration(ds, bucket_s, SAMPLES_PER_BUCKET)

        if not utterances:
            logger.warning("No utterances found for %d s bucket", bucket_s)
            continue

        for idx, example, dur in tqdm(utterances, desc=f"{bucket_s}s"):
            audio = example["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()
            audio_duration = len(audio_array) / sr

            result = stt_client.recognize_batch(audio_bytes, sr)
            elapsed = result["elapsed_s"]

            record = {
                "id": f"perf_{bucket_s}s_{idx}",
                "duration_bucket_s": bucket_s,
                "audio_duration_s": audio_duration,
                "elapsed_s": elapsed,
                "rtf": elapsed / audio_duration,
                "rtfx": audio_duration / elapsed,
                "concurrency": 1,
            }
            write_jsonl(jsonl_path, record)
            all_results.append(record)

    # ---- Concurrency test ----
    logger.info("Running concurrency test...")
    ref_utterances = _pick_utterances_by_duration(ds, 30, 1)
    if ref_utterances:
        _, ref_ex, _ = ref_utterances[0]
        ref_audio = np.array(ref_ex["audio"]["array"], dtype=np.float32)
        ref_sr = ref_ex["audio"]["sampling_rate"]
        ref_bytes = (ref_audio * 32768).astype(np.int16).tobytes()
        ref_dur = len(ref_audio) / ref_sr

        concurrency_results = []
        for n_workers in config.evaluation.stt_concurrency_levels:
            logger.info("Concurrency N=%d", n_workers)
            latencies = []

            def _call(_):
                r = stt_client.recognize_batch(ref_bytes, ref_sr)
                return r["elapsed_s"]

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_call, i) for i in range(n_workers * 5)]
                for f in as_completed(futures):
                    try:
                        latencies.append(f.result())
                    except Exception as e:
                        logger.warning("Concurrent call failed: %s", e)

            if latencies:
                stats = compute_percentiles(latencies)
                record = {
                    "concurrency": n_workers,
                    "n_calls": len(latencies),
                    **stats,
                    "rtf_mean": stats["mean"] / ref_dur,
                }
                concurrency_results.append(record)
                write_jsonl(results_dir / "concurrency.jsonl", record)

        save_summary_csv(results_dir / "concurrency_summary.csv", concurrency_results)

    # ---- Summary per bucket ----
    summary_rows = []
    for bucket_s in DURATION_BUCKETS:
        bucket_results = [r for r in all_results if r["duration_bucket_s"] == bucket_s]
        if not bucket_results:
            continue
        rtfs = [r["rtf"] for r in bucket_results]
        rtfxs = [r["rtfx"] for r in bucket_results]
        latencies = [r["elapsed_s"] for r in bucket_results]
        stats = compute_percentiles(latencies)
        summary_rows.append({
            "bucket_s": bucket_s,
            "n_calls": len(bucket_results),
            "rtf_mean": round(np.mean(rtfs), 4),
            "rtfx_mean": round(np.mean(rtfxs), 2),
            **{f"latency_{k}": round(v, 3) for k, v in stats.items()},
        })

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.2", "name": "batch_performance", "results": summary_rows}
