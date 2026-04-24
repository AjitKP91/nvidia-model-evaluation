"""Test 1.6 — Accent & Dialect Robustness."""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, bootstrap_ci, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.accent")

MIN_UTTERANCES_PER_GROUP = 50


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "accent"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.6: Accent & Dialect Robustness ===")

    # ---- Common Voice EN (accent metadata) ----
    try:
        cv_ds = load_dataset(
            "mozilla-foundation/common_voice_17_0", "en",
            split="test", trust_remote_code=True, token=True,
        )
    except Exception as e:
        logger.error("Common Voice EN not available: %s", e)
        cv_ds = None

    accent_groups: dict[str, list] = defaultdict(list)

    if cv_ds:
        for i, ex in enumerate(cv_ds):
            accent = ex.get("accent", "unknown") or "unknown"
            if len(accent_groups[accent]) < 150:
                accent_groups[accent].append((f"cv_{accent}_{i}", ex))

    # Filter groups with enough samples
    accent_groups = {
        k: v for k, v in accent_groups.items()
        if len(v) >= MIN_UTTERANCES_PER_GROUP and k != "unknown"
    }

    logger.info("Found %d accent groups with >= %d utterances", len(accent_groups), MIN_UTTERANCES_PER_GROUP)

    summary_rows = []
    group_wers = {}
    completed = get_completed_ids(jsonl_path)

    for accent, items in accent_groups.items():
        logger.info("Processing accent: %s (%d utterances)", accent, len(items))
        refs, hyps = [], []
        per_utt_wers = []

        for item_id, ex in tqdm(items, desc=accent):
            if item_id in completed:
                continue
            audio = ex["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            ref = ex.get("sentence", "") or ex.get("text", "")
            audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

            try:
                result = stt_client.recognize_batch(audio_bytes, sr)
                hyp = result["transcript"]
                refs.append(ref)
                hyps.append(hyp)

                utt_wer = jiwer.wer(ref, hyp, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
                per_utt_wers.append(utt_wer)

                write_jsonl(jsonl_path, {
                    "id": item_id, "accent": accent,
                    "reference": ref, "hypothesis": hyp, "wer": utt_wer,
                })
            except Exception as e:
                logger.warning("Failed %s: %s", item_id, e)

        if refs:
            agg_wer = jiwer.wer(refs, hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
            mean_wer, ci_lo, ci_hi = bootstrap_ci(per_utt_wers, n_bootstrap=config.evaluation.bootstrap_n)
            group_wers[accent] = agg_wer

            summary_rows.append({
                "accent": accent,
                "n_utterances": len(refs),
                "wer": round(agg_wer, 4),
                "wer_ci_lo": round(ci_lo, 4),
                "wer_ci_hi": round(ci_hi, 4),
            })
            logger.info("  %s: WER=%.2f%% [%.2f%%–%.2f%%]", accent, agg_wer * 100, ci_lo * 100, ci_hi * 100)

    # Fairness delta
    if group_wers:
        fairness_delta = max(group_wers.values()) - min(group_wers.values())
        logger.info("Fairness delta: %.2f pp (target < 10 pp)", fairness_delta * 100)
        for row in summary_rows:
            row["fairness_delta"] = round(fairness_delta, 4)

    # Sort by WER descending
    summary_rows.sort(key=lambda r: r["wer"], reverse=True)
    save_summary_csv(results_dir / "summary.csv", summary_rows)

    return {"test": "1.6", "name": "accent_robustness", "results": summary_rows}
