"""Test 1.2 — Batch Processing Performance (RTF / RTFx)."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes, resolve_languages
from eval.utils import compute_percentiles, load_dataset_tmp, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.performance")

DURATION_BUCKETS = [10, 30, 60, 300, 600]
SAMPLES_PER_BUCKET = 20
# Per-language FLEURS clip budget for opt-in Tier-B spot-checks (clips are short,
# so only the small duration buckets populate).
PER_LANG_PERF_LIMIT = 120


def _pick_utterances_by_duration(examples: list, target_s: float, n: int, tolerance: float = 0.3):
    """Select utterances whose duration is close to target_s."""
    candidates = []
    for i, ex in enumerate(examples):
        audio = ex["audio"]
        dur = len(audio["array"]) / audio["sampling_rate"]
        if abs(dur - target_s) / target_s < tolerance:
            candidates.append((i, ex, dur))
    candidates.sort(key=lambda x: abs(x[2] - target_s))
    return candidates[:n]


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "performance"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.2: Batch Processing Performance ===")

    with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean") as examples:
        pass

    all_results = []

    for bucket_s in DURATION_BUCKETS:
        logger.info("Duration bucket: %d s", bucket_s)
        utterances = _pick_utterances_by_duration(examples, bucket_s, SAMPLES_PER_BUCKET)

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
    ref_utterances = _pick_utterances_by_duration(examples, 30, 1)
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

    # ---- Opt-in Tier-B spot-check languages (FLEURS) ----
    spot = resolve_languages(getattr(config.stt, "spot_check_languages", []) or [])
    lang_rows: list[dict] = []
    for lang in spot:
        try:
            lang_rows.extend(_run_language_perf(lang, config, stt_client, results_dir))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Language %s failed in 1.2: %s", lang["key"], e)
    if lang_rows:
        save_summary_csv(results_dir / "summary_by_language.csv", lang_rows)
        summary_rows = summary_rows + lang_rows

    return {"test": "1.2", "name": "batch_performance", "results": summary_rows}


def _run_language_perf(lang: dict, config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """Opt-in per-language performance spot-check on FLEURS.

    Measures RTF/RTFx per duration bucket exactly like the English pass, and
    additionally reports output cost — transcript characters and whitespace
    tokens per audio-second — which exposes the autoregressive decoder-cost
    variance across scripts (CJK output is far denser per second than Latin).
    Languages that return empty transcripts record a ``no_output`` row.
    """
    key = lang["key"]
    jsonl_path = results_dir / f"perf_{key}.jsonl"
    logger.info("--- 1.2 spot-check language=%s (FLEURS %s) ---", key, lang["fleurs"])

    try:
        with fleurs_load(lang, limit=PER_LANG_PERF_LIMIT) as examples:
            pass
    except Exception as e:
        logger.error("FLEURS load failed for %s: %s", key, e)
        return [{"language": key, "bucket_s": "n/a", "status": "load_failed"}]

    if not examples:
        return [{"language": key, "bucket_s": "n/a", "status": "load_failed"}]

    rows: list[dict] = []
    non_empty = 0
    for bucket_s in DURATION_BUCKETS:
        picks = _pick_utterances_by_duration(examples, bucket_s, SAMPLES_PER_BUCKET)
        if not picks:
            continue
        rtfs, rtfxs, lat, chars_per_s, toks_per_s = [], [], [], [], []
        for idx, example, dur in picks:
            audio, sr, _ = get_audio_and_ref(example)
            audio_duration = len(audio) / sr
            try:
                result = stt_client.recognize_batch(pcm16_bytes(audio), sr, language_code=lang["model_lang"])
            except Exception as e:
                logger.warning("perf %s bucket %ds item %d failed: %s", key, bucket_s, idx, e)
                continue
            elapsed = result["elapsed_s"]
            transcript = result.get("transcript", "") or ""
            if transcript.strip():
                non_empty += 1
            rtfs.append(elapsed / audio_duration)
            rtfxs.append(audio_duration / elapsed)
            lat.append(elapsed)
            chars_per_s.append(len(transcript) / audio_duration if audio_duration else 0)
            toks_per_s.append(len(transcript.split()) / audio_duration if audio_duration else 0)
            write_jsonl(jsonl_path, {
                "id": f"perf_{key}_{bucket_s}s_{idx}",
                "language": key,
                "duration_bucket_s": bucket_s,
                "audio_duration_s": audio_duration,
                "elapsed_s": elapsed,
                "rtf": elapsed / audio_duration,
                "rtfx": audio_duration / elapsed,
                "transcript_chars": len(transcript),
                "transcript_tokens": len(transcript.split()),
            })
        if rtfs:
            stats = compute_percentiles(lat)
            rows.append({
                "language": key,
                "bucket_s": bucket_s,
                "n_calls": len(rtfs),
                "rtf_mean": round(float(np.mean(rtfs)), 4),
                "rtfx_mean": round(float(np.mean(rtfxs)), 2),
                "chars_per_audio_s": round(float(np.mean(chars_per_s)), 2),
                "tokens_per_audio_s": round(float(np.mean(toks_per_s)), 2),
                **{f"latency_{k}": round(v, 3) for k, v in stats.items()},
            })

    if non_empty == 0:
        logger.warning("Language %s returns empty transcripts; recording no_output", key)
        return [{"language": key, "bucket_s": "n/a", "status": "no_output"}]
    return rows
