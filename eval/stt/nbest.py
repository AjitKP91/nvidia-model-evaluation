"""Test 1.15 — N-Best Alternatives (serving-layer feature honor check).

``RecognitionConfig.max_alternatives`` asks the recognizer to return the top-N
competing hypotheses per segment. A streaming FastConformer served behind the
gateway may accept the field and still return a single alternative. This test
requests ``max_alternatives=5`` on clean LibriSpeech clips and records the
distribution of ``len(alternatives)`` and whether the lower-ranked hypotheses
actually differ from the top one — recording accepted / honored / ignored.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import get_completed_ids, load_dataset_tmp, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.nbest")

N_CLIPS = 30
N_BEST = 5


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "nbest"
    results_dir.mkdir(parents=True, exist_ok=True)
    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.15: N-Best Alternatives (max_alternatives=%d) ===", N_BEST)

    try:
        with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=N_CLIPS) as items:
            pass
    except Exception as e:
        logger.error("LibriSpeech load failed: %s", e)
        return {"test": "1.15", "name": "nbest", "results": {"verdict": "load_failed"}}

    completed = get_completed_ids(jsonl_path)

    for i, ex in enumerate(tqdm(items, desc="nbest")):
        item_id = f"nb_{i}"
        if item_id in completed:
            continue
        audio = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, sr, max_alternatives=N_BEST)
            alts = result.get("alternatives", []) or []
            alt_texts = [(a.get("transcript") or "").strip() for a in alts]
            distinct = len({t for t in alt_texts if t})
            write_jsonl(jsonl_path, {
                "id": item_id,
                "requested": N_BEST,
                "n_alternatives": len(alts),
                "n_distinct": distinct,
                "alt_differs": distinct > 1,
                "top": alt_texts[0] if alt_texts else "",
            })
        except Exception as e:
            logger.warning("Failed %s: %s", item_id, e)

    recs = [r for r in read_jsonl(jsonl_path) if "n_alternatives" in r]
    n = len(recs)
    if n == 0:
        summary = {"verdict": "no_data", "n_clips": 0}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.15", "name": "nbest", "results": summary}

    dist = Counter(r["n_alternatives"] for r in recs)
    max_returned = max(r["n_alternatives"] for r in recs)
    n_multi = sum(1 for r in recs if r["n_alternatives"] > 1)
    n_differ = sum(1 for r in recs if r.get("alt_differs"))

    if max_returned <= 1:
        verdict = "ignored"          # field accepted, gateway collapses to 1-best
    elif n_differ >= 0.5 * n_multi and n_multi > 0:
        verdict = "honored"
    else:
        verdict = "partial"

    summary = {
        "verdict": verdict,
        "n_clips": n,
        "requested": N_BEST,
        "max_returned": max_returned,
        "mean_returned": round(float(np.mean([r["n_alternatives"] for r in recs])), 3),
        "clips_multi_alt": n_multi,
        "clips_alts_differ": n_differ,
        "distribution": "; ".join(f"{k}:{v}" for k, v in sorted(dist.items())),
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("N-best verdict=%s max_returned=%d (requested %d)", verdict, max_returned, N_BEST)
    return {"test": "1.15", "name": "nbest", "results": summary}
