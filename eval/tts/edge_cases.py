"""Test 2.7 — Edge Cases & Input Robustness."""
from __future__ import annotations

import logging
from pathlib import Path

import jiwer
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_edge_cases
from eval.tts.client import TTSClient
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.edge_cases")


def _score_utmos_single(audio_path: str) -> float | None:
    try:
        from utmos import UTMOSScore
        scorer = UTMOSScore()
        return float(scorer.score(audio_path))
    except Exception:
        return None


def _round_trip_wer(text: str, audio_path: str) -> float | None:
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return float(jiwer.wer(
            text, result["text"],
            truth_transform=NORMALIZE_FOR_WER,
            hypothesis_transform=NORMALIZE_FOR_WER,
        ))
    except Exception:
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "edge_cases"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.7: Edge Cases & Input Robustness ===")

    edge_cases = get_edge_cases()
    logger.info("Testing %d edge cases", len(edge_cases))

    completed = get_completed_ids(jsonl_path)
    summary_rows = []

    for i, case in enumerate(tqdm(edge_cases, desc="Edge cases")):
        item_id = f"edge_{i}"
        if item_id in completed:
            continue
        text = case["text"]
        category = case["category"]
        wav_path = results_dir / f"edge_{i:04d}.wav"

        record = {
            "id": f"edge_{i}",
            "category": category,
            "text": text[:200],
            "status": None,
            "error": None,
            "utmos": None,
            "round_trip_wer": None,
            "audio_duration_s": None,
        }

        try:
            result = tts_client.save_synthesis(text, str(wav_path))
            record["status"] = "pass"
            record["audio_duration_s"] = result["audio_duration"]

            if result["audio_duration"] > 0.1:
                record["utmos"] = _score_utmos_single(str(wav_path))
                record["round_trip_wer"] = _round_trip_wer(text, str(wav_path))
            else:
                record["status"] = "degraded"
                record["error"] = "audio too short"

        except Exception as e:
            record["status"] = "fail"
            record["error"] = str(e)

        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # Aggregate by category
    category_summary = {}
    for row in summary_rows:
        cat = row["category"]
        if cat not in category_summary:
            category_summary[cat] = {"total": 0, "pass": 0, "fail": 0, "degraded": 0}
        category_summary[cat]["total"] += 1
        category_summary[cat][row["status"] or "fail"] += 1

    agg_rows = []
    for cat, counts in sorted(category_summary.items()):
        agg_rows.append({
            "category": cat,
            **counts,
            "pass_rate": counts["pass"] / counts["total"] if counts["total"] > 0 else 0,
        })

    save_summary_csv(results_dir / "summary.csv", agg_rows)

    total = len(summary_rows)
    passed = sum(1 for r in summary_rows if r["status"] == "pass")
    failed = sum(1 for r in summary_rows if r["status"] == "fail")
    logger.info("Edge cases: %d/%d passed, %d failed", passed, total, failed)

    return {"test": "2.7", "name": "edge_cases", "results": agg_rows}
