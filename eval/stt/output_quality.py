"""Test 1.9 — Output Quality (Punctuation / Capitalisation / Numbers)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import get_completed_ids, load_dataset_tmp, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.output_quality")

PUNCT_MARKS = [".", ",", "?", "!"]


def _extract_punct_labels(text: str, marks: list[str]) -> list[tuple[int, str]]:
    """Return list of (position, mark) for each punctuation mark in text."""
    labels = []
    for i, char in enumerate(text):
        if char in marks:
            labels.append((i, char))
    return labels


def _compute_punct_f1(ref_text: str, hyp_text: str) -> dict:
    """Compute per-mark punctuation precision, recall, F1."""
    results = {}
    for mark in PUNCT_MARKS:
        ref_count = ref_text.count(mark)
        hyp_count = hyp_text.count(mark)

        # Simple token-level comparison
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        min_len = min(len(ref_words), len(hyp_words))

        tp = fp = fn = 0
        for j in range(min_len):
            ref_has = mark in ref_words[j]
            hyp_has = mark in hyp_words[j]
            if ref_has and hyp_has:
                tp += 1
            elif hyp_has and not ref_has:
                fp += 1
            elif ref_has and not hyp_has:
                fn += 1

        # Remaining words
        for j in range(min_len, len(ref_words)):
            if mark in ref_words[j]:
                fn += 1
        for j in range(min_len, len(hyp_words)):
            if mark in hyp_words[j]:
                fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[mark] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }

    return results


def _compute_capitalisation_accuracy(ref_text: str, hyp_text: str) -> float:
    """Fraction of correctly capitalised tokens."""
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    min_len = min(len(ref_words), len(hyp_words))
    if min_len == 0:
        return 0.0

    correct = 0
    total = 0
    for j in range(min_len):
        rw = ref_words[j].strip(".,!?;:")
        hw = hyp_words[j].strip(".,!?;:")
        if rw.lower() == hw.lower():
            total += 1
            if rw == hw:
                correct += 1

    return correct / total if total > 0 else 0.0


def _compute_number_accuracy(ref_text: str, hyp_text: str) -> dict:
    """Check if numeric patterns are correctly formatted."""
    ref_numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', ref_text)
    hyp_numbers = re.findall(r'\b\d[\d,]*\.?\d*\b', hyp_text)

    if not ref_numbers:
        return {"accuracy": 1.0, "ref_count": 0, "hyp_count": 0}

    matched = sum(1 for n in ref_numbers if n in hyp_text)
    return {
        "accuracy": matched / len(ref_numbers),
        "ref_count": len(ref_numbers),
        "hyp_count": len(hyp_numbers),
    }


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "output_quality"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.9: Output Quality ===")

    # Use punctuated/cased datasets
    datasets_to_test = [
        ("spgispeech", ("kensho/spgispeech", None), "SPGISpeech", "val"),
        # revdotcom/earnings22 removed — builder downloads from dead third-party URLs
    ]

    all_punct_results: dict[str, list[dict]] = {m: [] for m in PUNCT_MARKS}
    all_cap_scores = []
    all_num_scores = []
    summary_rows = []
    completed = get_completed_ids(jsonl_path)

    for ds_name, (hf_path, hf_name), label, split in datasets_to_test:
        logger.info("Processing %s", label)
        try:
            with load_dataset_tmp(hf_path, split, name=hf_name, limit=500) as subset:
                pass
        except Exception as e:
            logger.error("Failed to load %s: %s", ds_name, e)
            continue

        for i, ex in enumerate(tqdm(subset, desc=ds_name)):
            item_id = f"{ds_name}_{i}"
            if item_id in completed:
                continue
            audio = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            ref = ex.get("text") or ex.get("norm_text") or ex.get("transcript") or ex.get("sentence", "")
            audio_bytes = (audio * 32768).astype(np.int16).tobytes()

            try:
                # Do NOT normalize — we need punctuation and casing
                result = stt_client.recognize_batch(
                    audio_bytes, sr, enable_punctuation=True
                )
                hyp = result["transcript"]

                punct = _compute_punct_f1(ref, hyp)
                cap_acc = _compute_capitalisation_accuracy(ref, hyp)
                num_acc = _compute_number_accuracy(ref, hyp)

                for mark in PUNCT_MARKS:
                    all_punct_results[mark].append(punct[mark])
                all_cap_scores.append(cap_acc)
                all_num_scores.append(num_acc["accuracy"])

                write_jsonl(jsonl_path, {
                    "id": f"{ds_name}_{i}",
                    "dataset": ds_name,
                    "reference": ref,
                    "hypothesis": hyp,
                    "punct_f1": {m: punct[m]["f1"] for m in PUNCT_MARKS},
                    "capitalisation_accuracy": cap_acc,
                    "number_accuracy": num_acc["accuracy"],
                })
            except Exception as e:
                logger.warning("Failed %s item %d: %s", ds_name, i, e)

    # Aggregate punctuation F1
    for mark in PUNCT_MARKS:
        if all_punct_results[mark]:
            f1s = [r["f1"] for r in all_punct_results[mark]]
            avg_f1 = np.mean(f1s)
            summary_rows.append({
                "metric": f"Punctuation F1 ({mark})",
                "value": round(avg_f1, 4),
                "n_samples": len(f1s),
            })
            logger.info("Punctuation '%s' F1: %.3f", mark, avg_f1)

    if all_cap_scores:
        avg_cap = np.mean(all_cap_scores)
        summary_rows.append({"metric": "Capitalisation Accuracy", "value": round(avg_cap, 4)})
        logger.info("Capitalisation accuracy: %.2f%%", avg_cap * 100)

    if all_num_scores:
        avg_num = np.mean(all_num_scores)
        summary_rows.append({"metric": "Number Formatting Accuracy", "value": round(avg_num, 4)})
        logger.info("Number formatting accuracy: %.2f%%", avg_num * 100)

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.9", "name": "output_quality", "results": summary_rows}
