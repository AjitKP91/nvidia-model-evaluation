"""Test 1.14 — Word-Level Timestamps (serving-layer feature honor check).

The Riva proto exposes ``enable_word_time_offsets`` and every ``WordInfo`` carries
``start_time``/``end_time``. The open question is whether the SAP AI Core gateway
*honors* it: a FastConformer streaming model can accept the flag and still return
all-zero offsets. This test transcribes clean LibriSpeech clips with
``enable_word_times=True`` and checks that the returned word timings are
populated, non-zero, monotonic, and span the clip — recording a
client-accepted / gateway-honored / ignored verdict.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import get_completed_ids, load_dataset_tmp, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.word_timestamps")

N_CLIPS = 30


def _monotonic(words: list[dict]) -> bool:
    starts = [w.get("start_time") or 0 for w in words]
    return all(b >= a for a, b in zip(starts, starts[1:]))


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "word_timestamps"
    results_dir.mkdir(parents=True, exist_ok=True)
    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.14: Word-Level Timestamps ===")

    try:
        with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=N_CLIPS) as items:
            pass
    except Exception as e:
        logger.error("LibriSpeech load failed: %s", e)
        return {"test": "1.14", "name": "word_timestamps", "results": {"verdict": "load_failed"}}

    completed = get_completed_ids(jsonl_path)

    for i, ex in enumerate(tqdm(items, desc="word-timestamps")):
        item_id = f"wt_{i}"
        if item_id in completed:
            continue
        audio = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        duration_s = len(audio) / sr
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, sr, enable_word_times=True)
            words = result.get("words", [])
            hyp_words = result["transcript"].split()
            ends = [w.get("end_time") or 0 for w in words]
            max_end = max(ends) if ends else 0
            nonzero = any(e > 0 for e in ends)
            covering = bool(max_end) and (0.5 * duration_s <= max_end <= 1.5 * duration_s)
            write_jsonl(jsonl_path, {
                "id": item_id,
                "n_hyp_words": len(hyp_words),
                "n_timestamped_words": len(words),
                "has_words": len(words) > 0,
                "nonzero": nonzero,
                "monotonic": _monotonic(words) if words else False,
                "max_end_time": round(float(max_end), 3),
                "duration_s": round(float(duration_s), 3),
                "covering": covering,
            })
        except Exception as e:
            logger.warning("Failed %s: %s", item_id, e)

    recs = [r for r in read_jsonl(jsonl_path) if "has_words" in r]
    n = len(recs)
    if n == 0:
        summary = {"verdict": "no_data", "n_clips": 0}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.14", "name": "word_timestamps", "results": summary}

    n_words = sum(1 for r in recs if r.get("has_words"))
    n_nonzero = sum(1 for r in recs if r.get("nonzero"))
    n_monotonic = sum(1 for r in recs if r.get("monotonic"))
    n_covering = sum(1 for r in recs if r.get("covering"))
    mean_words = float(np.mean([r.get("n_timestamped_words", 0) for r in recs]))

    nonzero_pct = n_nonzero / n
    if n_nonzero == 0:
        verdict = "ignored"          # flag accepted, but all offsets zero
    elif nonzero_pct >= 0.8 and n_covering / n >= 0.5:
        verdict = "honored"
    else:
        verdict = "partial"

    summary = {
        "verdict": verdict,
        "n_clips": n,
        "clips_with_words": n_words,
        "nonzero_pct": round(nonzero_pct, 4),
        "monotonic_pct": round(n_monotonic / n, 4),
        "covering_pct": round(n_covering / n, 4),
        "mean_timestamped_words": round(mean_words, 2),
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("Word-timestamps verdict=%s nonzero=%.0f%% covering=%.0f%%",
                verdict, nonzero_pct * 100, n_covering / n * 100)
    return {"test": "1.14", "name": "word_timestamps", "results": summary}
