"""Test 1.10 — Confidence Score Calibration."""
from __future__ import annotations

import logging
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_dataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.confidence")


def _compute_word_correctness(reference: str, hypothesis: str) -> list[tuple[str, int]]:
    """Return per-hypothesis-word correctness labels (1=correct, 0=error).
    Insertions are included with label=0. Deletions are excluded (no hyp token).
    """
    output = jiwer.process_words(reference, hypothesis)
    labels = []
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            for _ in range(chunk.ref_end_idx - chunk.ref_start_idx):
                labels.append(1)
        elif chunk.type == "substitute":
            for _ in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                labels.append(0)
        elif chunk.type == "insert":
            for _ in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                labels.append(0)
        # deletions: no hypothesis word → excluded
    return labels


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "confidence"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.10: Confidence Score Calibration ===")

    # Load 1000+ utterances from LibriSpeech + GigaSpeech
    ls_ds = load_dataset("esb/datasets", "librispeech", split="test", token=True)
    ls_subset = list(ls_ds.select(range(min(700, len(ls_ds)))))

    try:
        gs_ds = load_dataset("esb/datasets", "gigaspeech", split="test", token=True)
        gs_subset = list(gs_ds.select(range(min(300, len(gs_ds)))))
    except Exception:
        gs_subset = []
        logger.warning("GigaSpeech not available; using LibriSpeech only")

    all_items = [(f"ls_{i}", ex) for i, ex in enumerate(ls_subset)]
    all_items += [(f"gs_{i}", ex) for i, ex in enumerate(gs_subset)]

    all_confidences = []
    all_labels = []
    deletion_count = 0
    total_ref_words = 0
    completed = get_completed_ids(jsonl_path)

    for item_id, example in tqdm(all_items, desc="Confidence calibration"):
        if item_id in completed:
            continue
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("norm_text") or example.get("text", "")
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        try:
            result = stt_client.recognize_batch(audio_bytes, sr, enable_word_times=True)
            hyp = result["transcript"]
            confidence = result["confidence"]

            if confidence is None or confidence == 0:
                continue

            # Per-word correctness
            word_labels = _compute_word_correctness(ref, hyp)
            hyp_words = hyp.split()
            ref_words = ref.split()
            total_ref_words += len(ref_words)
            deletion_count += max(0, len(ref_words) - len(hyp_words))

            # Per-word confidence (if available from words field)
            word_confs = result.get("words", [])
            if word_confs and len(word_confs) == len(hyp_words):
                for wc, label in zip(word_confs, word_labels):
                    conf = wc.get("confidence")
                    if conf is not None:
                        all_confidences.append(conf)
                        all_labels.append(label)
            else:
                # Fall back to utterance-level confidence
                for label in word_labels:
                    all_confidences.append(confidence)
                    all_labels.append(label)

            write_jsonl(jsonl_path, {
                "id": item_id,
                "reference": ref,
                "hypothesis": hyp,
                "utterance_confidence": confidence,
                "n_correct_words": sum(word_labels),
                "n_total_hyp_words": len(word_labels),
            })
        except Exception as e:
            logger.warning("Failed %s: %s", item_id, e)

    if len(all_confidences) < 100:
        logger.warning("Not enough confidence data (%d points)", len(all_confidences))
        return {"test": "1.10", "name": "confidence_calibration", "results": {}}

    confidences = np.array(all_confidences)
    labels = np.array(all_labels)

    # ECE (bin-weighted)
    n_bins = 10
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, confidences, n_bins=n_bins, strategy="uniform"
    )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_details = []
    for b in range(len(fraction_of_positives)):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidences >= lo) & (confidences < hi)
        n_b = mask.sum()
        if n_b > 0:
            weight = n_b / len(confidences)
            gap = abs(fraction_of_positives[b] - mean_predicted_value[b])
            ece += weight * gap
            bin_details.append({
                "bin": f"[{lo:.1f}, {hi:.1f})",
                "n": int(n_b),
                "accuracy": round(float(fraction_of_positives[b]), 4),
                "confidence": round(float(mean_predicted_value[b]), 4),
                "gap": round(float(gap), 4),
            })

    # AUROC
    try:
        auroc = roc_auc_score(labels, confidences)
    except ValueError:
        auroc = None

    # Brier Score
    brier = brier_score_loss(labels, confidences)

    deletion_rate = deletion_count / total_ref_words if total_ref_words > 0 else 0

    summary = {
        "ece": round(float(ece), 4),
        "auroc": round(float(auroc), 4) if auroc else None,
        "brier_score": round(float(brier), 4),
        "n_words_evaluated": len(confidences),
        "deletion_rate": round(deletion_rate, 4),
        "bins": bin_details,
    }

    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("ECE=%.4f, AUROC=%.4f, Brier=%.4f, Deletion rate=%.2f%%",
        ece, auroc or 0, brier, deletion_rate * 100)

    return {"test": "1.10", "name": "confidence_calibration", "results": summary}
