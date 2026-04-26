"""Test 1.6 — Accent & Dialect Robustness."""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, bootstrap_ci, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.accent")

_HF_DATASET = "westbrook/English_Accent_DataSet"
_MAX_PER_ACCENT = 150
_MAX_ITER = 30_000   # cap to avoid streaming the full 53k if all groups fill early
MIN_UTTERANCES_PER_GROUP = 50


def _load_accent_groups() -> dict[str, list]:
    """Stream westbrook/English_Accent_DataSet and bucket by accent, max 150 each."""
    from datasets import load_dataset  # local import — heavy dep

    logger.info("Streaming %s (up to %d rows)…", _HF_DATASET, _MAX_ITER)
    ds = load_dataset(
        _HF_DATASET,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or None,
    )

    groups: dict[str, list] = defaultdict(list)
    for i, ex in enumerate(ds):
        if i >= _MAX_ITER:
            break
        accent = (ex.get("accent") or "").strip() or "unknown"
        if accent == "unknown":
            continue
        if len(groups[accent]) < _MAX_PER_ACCENT:
            groups[accent].append((f"accent_{accent}_{i}", ex))

    logger.info(
        "Collected %d accent groups, %d total utterances",
        len(groups),
        sum(len(v) for v in groups.values()),
    )
    return groups


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "accent"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.6: Accent & Dialect Robustness ===")

    accent_groups = _load_accent_groups()

    # Filter groups without enough samples
    accent_groups = {
        k: v for k, v in accent_groups.items()
        if len(v) >= MIN_UTTERANCES_PER_GROUP
    }

    logger.info(
        "Using %d accent groups with >= %d utterances",
        len(accent_groups), MIN_UTTERANCES_PER_GROUP,
    )

    summary_rows = []
    group_wers = {}
    completed = get_completed_ids(jsonl_path)

    for accent, items in accent_groups.items():
        logger.info("Processing accent: %s (%d utterances)", accent, len(items))
        refs, hyps, per_utt_wers = [], [], []

        for item_id, ex in tqdm(items, desc=accent):
            if item_id in completed:
                continue

            audio = ex["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            ref = (ex.get("raw_text") or ex.get("text") or ex.get("sentence") or "").strip()
            if not ref:
                continue

            audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

            try:
                result = stt_client.recognize_batch(audio_bytes, sr)
                hyp = result["transcript"]
                refs.append(ref)
                hyps.append(hyp)

                utt_wer = jiwer.wer(
                    ref, hyp,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )
                per_utt_wers.append(utt_wer)

                write_jsonl(jsonl_path, {
                    "id": item_id, "accent": accent,
                    "reference": ref, "hypothesis": hyp, "wer": utt_wer,
                })
            except Exception as e:
                logger.warning("Failed %s: %s", item_id, e)

        if refs:
            agg_wer = jiwer.wer(
                refs, hyps,
                reference_transform=NORMALIZE_FOR_WER,
                hypothesis_transform=NORMALIZE_FOR_WER,
            )
            mean_wer, ci_lo, ci_hi = bootstrap_ci(per_utt_wers, n_bootstrap=config.evaluation.bootstrap_n)
            group_wers[accent] = agg_wer

            summary_rows.append({
                "accent": accent,
                "n_utterances": len(refs),
                "wer": round(agg_wer, 4),
                "wer_ci_lo": round(ci_lo, 4),
                "wer_ci_hi": round(ci_hi, 4),
            })
            logger.info(
                "  %s: WER=%.2f%% [%.2f%%–%.2f%%]",
                accent, agg_wer * 100, ci_lo * 100, ci_hi * 100,
            )

    if group_wers:
        fairness_delta = max(group_wers.values()) - min(group_wers.values())
        logger.info("Fairness delta: %.2f pp (target < 10 pp)", fairness_delta * 100)
        for row in summary_rows:
            row["fairness_delta"] = round(fairness_delta, 4)

    summary_rows.sort(key=lambda r: r["wer"], reverse=True)
    save_summary_csv(results_dir / "summary.csv", summary_rows)

    return {"test": "1.6", "name": "accent_robustness", "results": summary_rows}
