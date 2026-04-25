"""Test 2.5 — Streaming TTFB & RTF (REST vs gRPC)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_text_length_buckets
from eval.tts.client import TTSClient
from eval.utils import compute_percentiles, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.latency")

N_REPS_PER_BUCKET = 50


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "latency"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"
    completed = get_completed_ids(jsonl_path)

    logger.info("=== Test 2.5: Streaming TTFB & RTF ===")

    buckets = get_text_length_buckets()

    summary_rows = []

    for bucket_name, texts in buckets.items():
        for interface in ["grpc", "rest"]:
            logger.info("Bucket=%s Interface=%s", bucket_name, interface)
            ttfb_values = []
            rtf_values = []
            elapsed_values = []

            for rep in tqdm(range(N_REPS_PER_BUCKET), desc=f"{bucket_name}/{interface}"):
                item_id = f"{bucket_name}_{interface}_{rep}"
                if item_id in completed:
                    continue
                text = texts[rep % len(texts)]

                try:
                    if interface == "grpc":
                        result = tts_client.synthesize_stream(text)
                    else:
                        result = tts_client.synthesize_stream_rest(text)

                    ttfb = result.get("ttfb")
                    rtf = result.get("rtf")
                    elapsed = result.get("elapsed_s")

                    if ttfb is not None:
                        ttfb_values.append(ttfb)
                    if rtf is not None:
                        rtf_values.append(rtf)
                    if elapsed is not None:
                        elapsed_values.append(elapsed)

                    write_jsonl(jsonl_path, {
                        "id": f"{bucket_name}_{interface}_{rep}",
                        "bucket": bucket_name,
                        "interface": interface,
                        "text_len": len(text),
                        "ttfb": ttfb,
                        "rtf": rtf,
                        "elapsed_s": elapsed,
                        "audio_duration_s": result.get("audio_duration"),
                    })
                except Exception as e:
                    logger.warning("Failed %s/%s rep %d: %s", bucket_name, interface, rep, e)

            if ttfb_values:
                row = {
                    "bucket": bucket_name,
                    "interface": interface,
                    "n_calls": len(ttfb_values),
                    "ttfb": compute_percentiles(ttfb_values),
                    "rtf": compute_percentiles(rtf_values) if rtf_values else {},
                    "elapsed": compute_percentiles(elapsed_values) if elapsed_values else {},
                }
                summary_rows.append(row)
                logger.info("  TTFB P50=%.0fms P99=%.0fms, RTF P50=%.3f",
                    row["ttfb"].get("p50", 0) * 1000,
                    row["ttfb"].get("p99", 0) * 1000,
                    row["rtf"].get("p50", 0) if row["rtf"] else 0,
                )

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "2.5", "name": "streaming_ttfb_rtf", "results": summary_rows}
