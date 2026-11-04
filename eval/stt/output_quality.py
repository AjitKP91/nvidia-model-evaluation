"""Test 1.9 — Output Quality (Punctuation / Capitalisation / Numbers)."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes, resolve_languages
from eval.stt.text_quality_rules import rules_for
from eval.utils import get_completed_ids, is_cer_language, load_dataset_tmp, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.output_quality")

PUNCT_MARKS = [".", ",", "?", "!"]
# Per-language FLEURS clip budget for the opt-in Tier-A multilingual pass.
PER_LANG_OQ_LIMIT = 200


def _extract_punct_labels(text: str, marks: list[str]) -> list[tuple[int, str]]:
    """Return list of (position, mark) for each punctuation mark in text."""
    labels = []
    for i, char in enumerate(text):
        if char in marks:
            labels.append((i, char))
    return labels


def _compute_punct_f1(ref_text: str, hyp_text: str, marks: list[str] | None = None,
                      char_level: bool = False) -> dict:
    """Compute per-mark punctuation precision, recall, F1.

    English default (marks=PUNCT_MARKS, char_level=False) reproduces the original
    token-level behaviour exactly. For CER languages (no word boundaries), pass
    char_level=True so marks are aligned per character instead of per space-token.
    """
    if marks is None:
        marks = PUNCT_MARKS
    tokenize = (lambda t: list(t)) if char_level else (lambda t: t.split())
    results = {}
    for mark in marks:
        ref_count = ref_text.count(mark)
        hyp_count = hyp_text.count(mark)

        # Token-level comparison (words, or characters for CER languages).
        ref_words = tokenize(ref_text)
        hyp_words = tokenize(hyp_text)
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


def _compute_capitalisation_accuracy(ref_text: str, hyp_text: str, has_case: bool = True):
    """Fraction of correctly capitalised tokens, or None for unicameral scripts.

    has_case=False (ja, zh, th, ar, ko) returns None (N/A) rather than a false 0,
    because those scripts have no upper/lower case distinction.
    """
    if not has_case:
        return None
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


def _compute_number_accuracy(ref_text: str, hyp_text: str, ascii_digits: bool = True) -> dict:
    """Check if numeric patterns are correctly formatted.

    ascii_digits=False (th, ar — scripts that normally use native digit glyphs)
    returns accuracy=None (N/A), because the ASCII-digit regex would mis-count.
    """
    if not ascii_digits:
        return {"accuracy": None, "ref_count": 0, "hyp_count": 0}
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
    """Test 1.9 entry point.

    Always runs the unchanged English SPGISpeech pass. If
    ``stt.multilingual_extra_languages`` is non-empty, each of those languages is
    additionally scored on FLEURS ``raw_transcription`` with language-aware rules
    (eval/stt/text_quality_rules.py): capitalisation is N/A for unicameral
    scripts, number accuracy is N/A for native-digit scripts, and the correct
    punctuation marks are counted per language. English output is byte-for-byte
    identical when the list is empty.
    """
    results_dir = Path(config.evaluation.results_dir) / "stt" / "output_quality"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)

    logger.info("=== Test 1.9: Output Quality ===")

    summary_rows = _run_english(config, stt_client, results_dir)

    extra = resolve_languages(getattr(config.stt, "multilingual_extra_languages", []) or [])
    for lang in extra:
        try:
            summary_rows.extend(_run_language(lang, config, stt_client, results_dir))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Language %s failed in 1.9: %s", lang["key"], e)
            summary_rows.append({"language": lang["key"], "metric": "load_failed", "value": None})

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.9", "name": "output_quality", "results": summary_rows}


def _run_english(config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """The original English SPGISpeech output-quality pass (unchanged)."""
    jsonl_path = results_dir / "calls.jsonl"

    # Use punctuated/cased datasets
    datasets_to_test = [
        ("spgispeech", ("kensho/spgispeech", "S"), "SPGISpeech", "validation"),
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

    return summary_rows


def _run_language(lang: dict, config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """Opt-in per-language output-quality pass on FLEURS raw_transcription.

    Uses language-aware rules: the right punctuation marks, capitalisation N/A
    for unicameral scripts, number accuracy N/A for native-digit scripts, and
    character-level punctuation alignment for CER languages. Languages that
    return empty transcripts record a graceful ``no_output`` row.
    """
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"
    rules = rules_for(key)
    marks = rules["punct_marks"]
    char_level = is_cer_language(key)
    logger.info("--- 1.9 language=%s (FLEURS %s, marks=%s, case=%s) ---",
                key, lang["fleurs"], "".join(marks) or "none", rules["has_case"])

    try:
        with fleurs_load(lang, limit=PER_LANG_OQ_LIMIT) as examples:
            pass
    except Exception as e:
        logger.error("FLEURS load failed for %s: %s", key, e)
        return [{"language": key, "metric": "load_failed", "value": None}]

    if not examples:
        return [{"language": key, "metric": "load_failed", "value": None}]

    punct_results: dict[str, list[dict]] = {m: [] for m in marks}
    cap_scores: list[float] = []
    num_scores: list[float] = []
    non_empty = 0
    completed = get_completed_ids(jsonl_path)

    for i, ex in enumerate(tqdm(examples, desc=f"oq:{key}")):
        item_id = f"{key}_{i}"
        if item_id in completed:
            continue
        # raw_transcription keeps casing/punctuation/numerals (needed here).
        audio, sr, ref = get_audio_and_ref(ex, field="raw_transcription")
        try:
            result = stt_client.recognize_batch(
                pcm16_bytes(audio), sr, enable_punctuation=True, language_code=lang["model_lang"]
            )
            hyp = result["transcript"]
            if (hyp or "").strip():
                non_empty += 1

            punct = _compute_punct_f1(ref, hyp, marks=marks, char_level=char_level)
            cap_acc = _compute_capitalisation_accuracy(ref, hyp, has_case=rules["has_case"])
            num_acc = _compute_number_accuracy(ref, hyp, ascii_digits=rules["ascii_digits"])

            for mark in marks:
                punct_results[mark].append(punct[mark])
            if cap_acc is not None:
                cap_scores.append(cap_acc)
            if num_acc["accuracy"] is not None:
                num_scores.append(num_acc["accuracy"])

            write_jsonl(jsonl_path, {
                "id": item_id,
                "language": key,
                "reference": ref,
                "hypothesis": hyp,
                "punct_f1": {m: punct[m]["f1"] for m in marks},
                "capitalisation_accuracy": cap_acc,
                "number_accuracy": num_acc["accuracy"],
            })
        except Exception as e:
            logger.warning("Failed %s item %d: %s", key, i, e)

    # Endpoint produced nothing for this language — honest no_output row.
    if non_empty == 0:
        logger.warning("Language %s returns empty transcripts; recording no_output", key)
        return [{"language": key, "metric": "no_output", "value": None}]

    rows: list[dict] = []
    for mark in marks:
        if punct_results[mark]:
            f1s = [r["f1"] for r in punct_results[mark]]
            rows.append({
                "language": key,
                "metric": f"Punctuation F1 ({mark})",
                "value": round(float(np.mean(f1s)), 4),
                "n_samples": len(f1s),
            })
    rows.append({
        "language": key,
        "metric": "Capitalisation Accuracy",
        "value": round(float(np.mean(cap_scores)), 4) if cap_scores else "n/a",
    })
    rows.append({
        "language": key,
        "metric": "Number Formatting Accuracy",
        "value": round(float(np.mean(num_scores)), 4) if num_scores else "n/a",
    })
    return rows
