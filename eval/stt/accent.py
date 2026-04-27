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

_VCTK_DATASET = "CSTR-Edinburgh/vctk"
_L2ARCTIC_DATASET = "KoelLabs/L2Arctic"
_TARGET_SR = 16_000
_MAX_PER_ACCENT = 150
MIN_UTTERANCES_PER_GROUP = 20


def _load_vctk_groups() -> dict[str, list]:
    """Load CSTR-Edinburgh/vctk, bucket by accent (skip Unknown), max 150 each."""
    import itertools
    from datasets import load_dataset

    logger.info("Loading %s…", _VCTK_DATASET)
    ds = load_dataset(
        _VCTK_DATASET,
        split="train",
        token=os.environ.get("HF_TOKEN") or None,
        trust_remote_code=True,
    )

    groups: dict[str, list] = defaultdict(list)
    for i, ex in enumerate(itertools.islice(ds, 50_000)):
        accent = (ex.get("accent") or "").strip()
        if not accent or accent.lower() == "unknown":
            continue
        if len(groups[accent]) < _MAX_PER_ACCENT:
            groups[accent].append((f"vctk_{accent}_{i}", ex))

    logger.info("VCTK: %d groups, %d utterances", len(groups), sum(len(v) for v in groups.values()))
    return groups


def _load_l2arctic_groups() -> dict[str, list]:
    """Load KoelLabs/L2Arctic scripted split, bucket by native_language, max 150 each."""
    import itertools
    from datasets import load_dataset

    logger.info("Loading %s…", _L2ARCTIC_DATASET)
    ds = load_dataset(
        _L2ARCTIC_DATASET,
        split="scripted",
        token=os.environ.get("HF_TOKEN") or None,
        trust_remote_code=True,
    )

    groups: dict[str, list] = defaultdict(list)
    for i, ex in enumerate(itertools.islice(ds, 10_000)):
        accent = (ex.get("speaker_native_language") or "").strip()
        if not accent or accent.lower() == "unknown":
            continue
        if len(groups[accent]) < _MAX_PER_ACCENT:
            groups[accent].append((f"l2arctic_{accent}_{i}", ex))

    logger.info("L2-Arctic: %d groups, %d utterances", len(groups), sum(len(v) for v in groups.values()))
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


def _get_audio_and_ref(ex: dict) -> tuple[np.ndarray, int, str]:
    audio = ex["audio"]
    array = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    ref = (ex.get("text") or ex.get("sentence") or "").strip()
    return array, sr, ref


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "accent"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.6: Accent & Dialect Robustness ===")

    # Load both datasets and merge accent groups
    vctk_groups = _load_vctk_groups()
    l2arctic_groups = _load_l2arctic_groups()

    accent_groups: dict[str, list] = {}
    accent_groups.update(vctk_groups)
    accent_groups.update(l2arctic_groups)

    # Filter groups with too few samples
    accent_groups = {
        k: v for k, v in accent_groups.items()
        if len(v) >= MIN_UTTERANCES_PER_GROUP
    }

    logger.info(
        "Using %d accent groups with >= %d utterances (VCTK + L2-Arctic)",
        len(accent_groups), MIN_UTTERANCES_PER_GROUP,
    )

    jsonl_cache = _reconstruct_from_jsonl(jsonl_path)
    if jsonl_cache:
        logger.info(
            "Found %d accent groups already in calls.jsonl — reusing cached results",
            len(jsonl_cache),
        )

    summary_rows = []
    group_wers = {}
    completed = get_completed_ids(jsonl_path)

    for accent, items in accent_groups.items():
        source = "VCTK" if items[0][0].startswith("vctk_") else "L2-Arctic"
        logger.info("Processing accent: %s [%s] (%d utterances)", accent, source, len(items))
        refs, hyps, per_utt_wers = [], [], []

        if accent in jsonl_cache:
            cached = jsonl_cache[accent]
            refs.extend(cached["refs"])
            hyps.extend(cached["hyps"])
            per_utt_wers.extend(cached["wers"])
            logger.info("  Loaded %d cached results from JSONL", len(refs))

        for item_id, ex in tqdm(items, desc=f"{accent} ({source})"):
            if item_id in completed:
                continue

            audio_array, sr, ref = _get_audio_and_ref(ex)
            if not ref:
                continue

            # Resample to 16 kHz if needed (VCTK is 48 kHz; L2-Arctic is already 16 kHz)
            if sr != _TARGET_SR:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=_TARGET_SR)
                sr = _TARGET_SR

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
                    "id": item_id, "accent": accent, "source": source,
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
                "source": source,
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
