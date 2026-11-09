"""Test 1.10 — Confidence Score Calibration."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_dataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes, resolve_languages
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, load_dataset_tmp, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.confidence")

# Per-language FLEURS clip budget for opt-in Tier-B calibration spot-checks.
PER_LANG_CONF_LIMIT = 300

# Languages written without spaces between words (scriptio continua). Per-word
# confidence calibration aligns words split on whitespace, which is meaningless
# for these scripts — the model emits per-character tokens that can never line
# up with a phrase-spaced reference, so every "word" scores wrong and the
# resulting ECE/AUROC/Brier are artefacts, not calibration. We record a
# graceful status for these instead of a fabricated number.
NO_WORD_BOUNDARY_LANGS = {"ja", "zh", "cmn", "yue", "wuu", "th", "lo", "my", "km"}

# Strip everything that isn't a word char or whitespace (punctuation, quotes,
# etc.). Applied per-token so the word count is preserved exactly.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _norm_tokens(text: str) -> list[str]:
    """Lowercase + depunctuate each whitespace token, preserving token count.

    Confidence calibration pairs each *hypothesis* word with its per-word
    confidence from the model, so the correctness label list must stay aligned
    1:1 with ``hyp.split()``. We therefore normalize token-by-token (never
    collapse or split tokens): a token that reduces to empty keeps its
    lowercased raw form as a placeholder so the position — and the count — is
    held. This mirrors the lowercase+strip-punctuation that the WER tests apply
    via ``NORMALIZE_FOR_WER`` so "correct" means the same thing here as there,
    without which ALL-CAPS/punctuated references score as ~all-wrong.
    """
    out = []
    for tok in text.split():
        norm = _PUNCT_RE.sub("", tok.lower()).strip()
        out.append(norm if norm else tok.lower())
    return out


def _compute_word_correctness(reference: str, hypothesis: str) -> list[int]:
    """Return per-hypothesis-word correctness labels (1=correct, 0=error).
    Insertions are included with label=0. Deletions are excluded (no hyp token).

    Both sides are normalized (lowercase + punctuation stripped) before
    alignment so case/punctuation differences between the reference corpus and
    the model's cased, punctuated output are not counted as word errors.
    """
    ref_norm = " ".join(_norm_tokens(reference))
    hyp_norm = " ".join(_norm_tokens(hypothesis))
    output = jiwer.process_words(ref_norm, hyp_norm)
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
    with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=700) as ls_subset:
        pass

    try:
        with load_dataset_tmp("speechcolab/gigaspeech", "test", name="xs", limit=300) as gs_subset:
            pass
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

    summary = _calibration_summary(all_confidences, all_labels, deletion_count, total_ref_words)
    if summary is None:
        return {"test": "1.10", "name": "confidence_calibration", "results": {}}

    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("ECE=%.4f, AUROC=%.4f, Brier=%.4f, Deletion rate=%.2f%%",
        summary["ece"], summary["auroc"] or 0, summary["brier_score"], summary["deletion_rate"] * 100)

    # ---- Opt-in Tier-B spot-check languages (FLEURS) ----
    spot = resolve_languages(getattr(config.stt, "spot_check_languages", []) or [])
    per_language: dict[str, dict] = {}
    for lang in spot:
        try:
            per_language[lang["key"]] = _run_language_conf(lang, config, stt_client, results_dir)
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Language %s failed in 1.10: %s", lang["key"], e)
            per_language[lang["key"]] = {"status": "error", "error": str(e)}
    if per_language:
        summary["per_language"] = per_language

    return {"test": "1.10", "name": "confidence_calibration", "results": summary}


def _calibration_summary(all_confidences, all_labels, deletion_count, total_ref_words) -> dict | None:
    """Compute ECE / AUROC / Brier from paired confidence & correctness arrays.

    Returns None when there are fewer than 100 confidence points (same guard the
    English pass always used). Deterministic — extracting it preserves the
    original English output exactly.
    """
    if len(all_confidences) < 100:
        logger.warning("Not enough confidence data (%d points)", len(all_confidences))
        return None

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

    return {
        "ece": round(float(ece), 4),
        "auroc": round(float(auroc), 4) if auroc else None,
        "brier_score": round(float(brier), 4),
        "n_words_evaluated": len(confidences),
        "deletion_rate": round(deletion_rate, 4),
        "bins": bin_details,
    }


def _run_language_conf(lang: dict, config: Config, stt_client: STTClient, results_dir: Path) -> dict:
    """Opt-in per-language confidence-calibration spot-check on FLEURS.

    Item IDs are language-tagged (``{key}_{i}``) and written to a per-language
    JSONL so resume never collides across languages. Reuses the same
    ECE/AUROC/Brier logic as the English pass. Languages that return empty
    transcripts or too few confidence points record a graceful status instead of
    a fabricated calibration.
    """
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"
    logger.info("--- 1.10 spot-check language=%s (FLEURS %s) ---", key, lang["fleurs"])

    if key in NO_WORD_BOUNDARY_LANGS:
        logger.warning(
            "Language %s has no whitespace word boundaries; per-word confidence "
            "calibration is not meaningful — recording no_word_boundaries", key)
        return {"status": "no_word_boundaries"}

    try:
        with fleurs_load(lang, limit=PER_LANG_CONF_LIMIT) as examples:
            pass
    except Exception as e:
        logger.error("FLEURS load failed for %s: %s", key, e)
        return {"status": "load_failed"}

    if not examples:
        return {"status": "load_failed"}

    all_confidences, all_labels = [], []
    deletion_count = total_ref_words = 0
    non_empty = 0
    completed = get_completed_ids(jsonl_path)

    for i, example in enumerate(tqdm(examples, desc=f"conf:{key}")):
        item_id = f"{key}_{i}"
        if item_id in completed:
            continue
        audio, sr, ref = get_audio_and_ref(example)
        try:
            result = stt_client.recognize_batch(pcm16_bytes(audio), sr, enable_word_times=True, language_code=lang["model_lang"])
            hyp = result["transcript"]
            confidence = result["confidence"]
            if (hyp or "").strip():
                non_empty += 1
            if confidence is None or confidence == 0:
                continue

            word_labels = _compute_word_correctness(ref, hyp)
            hyp_words = hyp.split()
            ref_words = ref.split()
            total_ref_words += len(ref_words)
            deletion_count += max(0, len(ref_words) - len(hyp_words))

            word_confs = result.get("words", [])
            if word_confs and len(word_confs) == len(hyp_words):
                for wc, label in zip(word_confs, word_labels):
                    conf = wc.get("confidence")
                    if conf is not None:
                        all_confidences.append(conf)
                        all_labels.append(label)
            else:
                for label in word_labels:
                    all_confidences.append(confidence)
                    all_labels.append(label)

            write_jsonl(jsonl_path, {
                "id": item_id,
                "language": key,
                "reference": ref,
                "hypothesis": hyp,
                "utterance_confidence": confidence,
                "n_correct_words": sum(word_labels),
                "n_total_hyp_words": len(word_labels),
            })
        except Exception as e:
            logger.warning("Failed %s: %s", item_id, e)

    if non_empty == 0:
        logger.warning("Language %s returns empty transcripts; recording no_output", key)
        return {"status": "no_output"}

    summary = _calibration_summary(all_confidences, all_labels, deletion_count, total_ref_words)
    if summary is None:
        return {"status": "insufficient_data", "n_points": len(all_confidences)}
    save_summary_csv(results_dir / f"{key}_summary.csv", [summary])
    logger.info("[%s] ECE=%.4f, AUROC=%.4f, Brier=%.4f", key,
                summary["ece"], summary["auroc"] or 0, summary["brier_score"])
    return summary
