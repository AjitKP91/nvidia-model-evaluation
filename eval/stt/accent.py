"""Test 1.6 — Accent & Dialect Robustness."""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import jiwer
import librosa
import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, NORMALIZE_FOR_WER_AGG, bootstrap_ci, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.accent")

_HF_DATASET = "CSTR-Edinburgh/vctk"
_TARGET_SR = 16_000
_MAX_PER_ACCENT = 150
_MAX_ITER = 50_000
MIN_UTTERANCES_PER_GROUP = 20


def _load_accent_groups() -> dict[str, list]:
    """Load CSTR-Edinburgh/vctk and bucket by accent, max 150 each."""
    from datasets import load_dataset

    logger.info("Loading %s…", _HF_DATASET)
    ds = load_dataset(
        _HF_DATASET,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN") or None,
        trust_remote_code=True,
    )

    groups: dict[str, list] = defaultdict(list)
    for i, ex in enumerate(ds):
        if i >= _MAX_ITER:
            break
        accent = (ex.get("accent") or "").strip() or "unknown"
        if accent == "unknown":
            continue
        if len(groups[accent]) < _MAX_PER_ACCENT:
            groups[accent].append((f"vctk_{accent}_{i}", ex))

    logger.info(
        "Collected %d accent groups, %d total utterances",
        len(groups),
        sum(len(v) for v in groups.values()),
    )
    return groups


def _reconstruct_from_jsonl(jsonl_path: Path) -> dict[str, dict]:
    """Read completed results from calls.jsonl, grouped by accent."""
    if not jsonl_path.exists():
        return {}

    groups: dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": [], "wers": []})
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            accent = row.get("accent")
            ref = row.get("reference", "")
            hyp = row.get("hypothesis", "")
            wer = row.get("wer")
            if accent and ref and hyp is not None and wer is not None:
                groups[str(accent)]["refs"].append(ref)
                groups[str(accent)]["hyps"].append(hyp)
                groups[str(accent)]["wers"].append(wer)

    return dict(groups)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "accent"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.6: Accent & Dialect Robustness ===")

    accent_groups = _load_accent_groups()

    accent_groups = {
        k: v for k, v in accent_groups.items()
        if len(v) >= MIN_UTTERANCES_PER_GROUP
    }

    logger.info(
        "Using %d accent groups with >= %d utterances",
        len(accent_groups), MIN_UTTERANCES_PER_GROUP,
    )

    jsonl_cache = _reconstruct_from_jsonl(jsonl_path)
    if jsonl_cache:
        logger.info(
            "Found %d accent groups already in calls.jsonl — will reuse their results",
            len(jsonl_cache),
        )

    summary_rows = []
    group_wers = {}
    completed = get_completed_ids(jsonl_path)

    for accent, items in accent_groups.items():
        logger.info("Processing accent: %s (%d utterances)", accent, len(items))
        refs, hyps, per_utt_wers = [], [], []

        if accent in jsonl_cache:
            cached = jsonl_cache[accent]
            refs.extend(cached["refs"])
            hyps.extend(cached["hyps"])
            per_utt_wers.extend(cached["wers"])
            logger.info("  Loaded %d cached results from JSONL for accent %s", len(refs), accent)

        for item_id, ex in tqdm(items, desc=accent):
            if item_id in completed:
                continue

            audio = ex["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            # VCTK is 48 kHz — resample to 16 kHz for Parakeet
            if sr != _TARGET_SR:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=_TARGET_SR)
                sr = _TARGET_SR

            ref = (ex.get("text") or ex.get("sentence") or "").strip()
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
                reference_transform=NORMALIZE_FOR_WER_AGG,
                hypothesis_transform=NORMALIZE_FOR_WER_AGG,
            )
            mean_wer, ci_lo, ci_hi = bootstrap_ci(per_utt_wers, n_bootstrap=config.evaluation.bootstrap_n)
            group_wers[accent] = agg_wer

            summary_rows.append({
                "accent": accent,
                "n_utterances": len(refs),
                "wer": round(agg_wer, 4),
                "mean_wer": round(mean_wer, 4),
                "wer_ci_lo": round(ci_lo, 4),
                "wer_ci_hi": round(ci_hi, 4),
            })
            logger.info(
                "  %s: WER=%.2f%% (mean=%.2f%% [%.2f%%–%.2f%%])",
                accent, agg_wer * 100, mean_wer * 100, ci_lo * 100, ci_hi * 100,
            )

    if group_wers:
        fairness_delta = max(group_wers.values()) - min(group_wers.values())
        logger.info("Fairness delta: %.2f pp (target < 10 pp)", fairness_delta * 100)
        for row in summary_rows:
            row["fairness_delta"] = round(fairness_delta, 4)

    summary_rows.sort(key=lambda r: r["wer"], reverse=True)
    save_summary_csv(results_dir / "summary.csv", summary_rows)

    return {"test": "1.6", "name": "accent_robustness", "results": summary_rows}
