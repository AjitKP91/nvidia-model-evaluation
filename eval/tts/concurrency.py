"""Test 2.6 — Throughput & Concurrency."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.tts.client import TTSClient
from eval.utils import compute_percentiles, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.concurrency")

TEST_TEXT = (
    "The artificial intelligence revolution is transforming industries across the globe, "
    "from healthcare and finance to transportation and entertainment."
)

_GRPC_CALL_TIMEOUT_S = 60


async def _rest_request(session: aiohttp.ClientSession, url: str, headers: dict, payload: dict) -> dict:
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            data = await resp.read()
            elapsed = time.perf_counter() - start
            return {"elapsed_s": elapsed, "status": resp.status, "audio_len": len(data)}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"elapsed_s": elapsed, "status": -1, "error": str(e)}


async def _run_concurrent_rest(config: Config, n_concurrent: int, n_total: int) -> list[dict]:
    headers = {
        config.tts.auth_header: f"Bearer {config.riva.auth_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "text": TEST_TEXT,
        "voice": {"name": config.tts.voice_name},
        "language_code": config.tts.language_code,
        "sample_rate_hz": config.tts.sample_rate,
    }

    sem = asyncio.Semaphore(n_concurrent)
    results = []

    async def bounded_request(session):
        async with sem:
            return await _rest_request(session, config.tts.rest_endpoint, headers, payload)

    timeout = aiohttp.ClientTimeout(total=config.tts.request_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_request(session) for _ in range(n_total)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return [
        r if isinstance(r, dict) else {"elapsed_s": 0, "status": -1, "error": str(r)}
        for r in results
    ]


def _run_concurrent_grpc(tts_client: TTSClient, n_concurrent: int, n_total: int) -> tuple[list[dict], float]:
    """Returns (results, wall_elapsed_s)."""
    results: list[dict] = []
    wall_start = time.perf_counter()

    def _call(_):
        return tts_client.synthesize_batch(TEST_TEXT)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(_call, i) for i in range(n_total)]
        for f in concurrent.futures.as_completed(futures):
            try:
                r = f.result(timeout=_GRPC_CALL_TIMEOUT_S)
                results.append({"elapsed_s": r["elapsed_s"], "status": 200, "audio_duration": r["audio_duration"]})
            except concurrent.futures.TimeoutError:
                results.append({"elapsed_s": _GRPC_CALL_TIMEOUT_S, "status": -1, "error": "timeout"})
            except Exception as e:
                results.append({"elapsed_s": 0, "status": -1, "error": str(e)})

    wall_elapsed = time.perf_counter() - wall_start
    return results, wall_elapsed


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "concurrency"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.6: Throughput & Concurrency ===")

    n_total_per_level = 50
    summary_rows = []

    for n_concurrent in config.evaluation.tts_concurrency_levels:
        logger.info("Concurrency N=%d", n_concurrent)

        # gRPC concurrency
        logger.info("  gRPC...")
        grpc_results, grpc_wall = _run_concurrent_grpc(tts_client, n_concurrent, n_total_per_level)
        grpc_latencies = [r["elapsed_s"] for r in grpc_results if r["status"] == 200]
        grpc_errors = sum(1 for r in grpc_results if r["status"] != 200)
        grpc_rps = len(grpc_latencies) / grpc_wall if grpc_wall > 0 else 0

        if grpc_latencies:
            grpc_stats = compute_percentiles(grpc_latencies)
            row = {
                "interface": "grpc",
                "concurrency": n_concurrent,
                "n_calls": len(grpc_results),
                "n_success": len(grpc_latencies),
                "n_errors": grpc_errors,
                "error_rate": round(grpc_errors / len(grpc_results), 4),
                "rps": round(grpc_rps, 2),
                **{f"latency_{k}": round(v, 3) for k, v in grpc_stats.items()},
            }
            summary_rows.append(row)
            write_jsonl(jsonl_path, row)

        # REST concurrency
        logger.info("  REST...")
        try:
            rest_results = asyncio.run(
                _run_concurrent_rest(config, n_concurrent, n_total_per_level)
            )
            rest_latencies = [r["elapsed_s"] for r in rest_results if r.get("status") == 200]
            rest_errors = sum(1 for r in rest_results if r.get("status") != 200)

            if rest_latencies:
                rest_stats = compute_percentiles(rest_latencies)
                rest_total_time = sum(rest_latencies) / n_concurrent if n_concurrent > 0 else sum(rest_latencies)
                row = {
                    "interface": "rest",
                    "concurrency": n_concurrent,
                    "n_calls": len(rest_results),
                    "n_success": len(rest_latencies),
                    "n_errors": rest_errors,
                    "error_rate": round(rest_errors / len(rest_results), 4),
                    "rps": round(len(rest_latencies) / rest_total_time, 2) if rest_total_time > 0 else 0,
                    **{f"latency_{k}": round(v, 3) for k, v in rest_stats.items()},
                }
                summary_rows.append(row)
                write_jsonl(jsonl_path, row)
        except Exception as e:
            logger.warning("REST concurrency test failed: %s", e)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    for row in summary_rows:
        logger.info(
            "%s N=%d: P50=%.0fms P99=%.0fms RPS=%.1f errors=%d",
            row["interface"], row["concurrency"],
            row.get("latency_p50", 0) * 1000, row.get("latency_p99", 0) * 1000,
            row.get("rps", 0), row.get("n_errors", 0),
        )

    return {"test": "2.6", "name": "concurrency", "results": summary_rows}
