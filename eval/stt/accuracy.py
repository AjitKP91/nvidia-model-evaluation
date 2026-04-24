"""Test 1.1 — Clean Speech Accuracy.

Establish the accuracy baseline across diverse English speech styles.
"""
from __future__ import annotations

import logging
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import (
    NORMALIZE_FOR_WER,
    bootstrap_ci,
    get_completed_ids,
    save_summary_csv,
    write_jsonl,
)

logger = logging.getLogger("eval.stt.accuracy")

DATASETS = [
    {"name": "librispeech_clean", "hf": ("librispeech_asr", "clean"), "split": "test.clean", "label": "LibriSpeech clean"},
    {"name": "librispeech_other", "hf": ("librispeech_asr", "other"), "split": "test.other", "label": "LibriSpeech other"},
    {"name": "tedlium", "hf": ("LIUM/tedlium", "release3"), "split": "test", "label": "TED-LIUM v3"},
    {"name": "gigaspeech", "hf": ("speechcolab/gigaspeech", "xs"), "split": "test", "label": "GigaSpeech"},
    {"name": "spgispeech", "hf": ("kensho/spgispeech", None), "split": "val", "label": "SPGISpeech"},
    {"name": "earnings22", "hf": ("revdotcom/earnings22", None), "split": "test", "label": "Earnings-22"},
    {"name": "ami", "hf": ("edinburghcst/ami", "ihm"), "split": "test", "label": "AMI IHM"},
]

THRESHOLDS = {
    "librispeech_clean": {"pass": 0.05, "good": 0.07, "acceptable": 0.10},
    "librispeech_other": {"pass": 0.10, "good": 0.15, "acceptable": 0.20},
    "tedlium": {"pass": 0.08, "good": 0.12, "acceptable": 0.18},
    "gigaspeech": {"pass": 0.12, "good": 0.16, "acceptable": 0.22},
    "spgispeech": {"pass": 0.08, "good": 0.12, "acceptable": 0.18},
    "earnings22": {"pass": 0.15, "good": 0.22, "acceptable": 0.30},
    "ami": {"pass": 0.18, "good": 0.25, "acceptable": 0.35},
}


def _load_dataset(ds_info: dict) -> object:
    path, name = ds_info["hf"]
    kwargs = {"path": path, "split": ds_info["split"], "token": True}
    # Only Common Voice requires trust_remote_code; all other standard HF
    # datasets are built-in or use simple builders that don't need it.
    # Passing trust_remote_code=True on others causes the library to look for
    # a local dataset script (e.g. ./LIUM/tedlium/) and triggers ** pattern
    # errors with Python 3.10 pathlib on datasets that have custom scripts.
    if path.startswith("mozilla-foundation/"):
        kwargs["trust_remote_code"] = True
    if name:
        kwargs["name"] = name
    return load_dataset(**kwargs)


def _get_audio_and_ref(example: dict) -> tuple[np.ndarray, int, str]:
    audio = example["audio"]
    array = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    ref = example.get("norm_text") or example.get("text") or example.get("transcript") or example.get("sentence", "")
    return array, sr, ref


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "accuracy"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    summary_rows = []

    for ds_info in DATASETS:
        ds_name = ds_info["name"]
        logger.info("=== Test 1.1: %s ===", ds_info["label"])

        jsonl_path = results_dir / f"{ds_name}.jsonl"
        completed = get_completed_ids(jsonl_path)

        try:
            ds = _load_dataset(ds_info)
        except Exception as e:
            logger.error("Failed to load %s: %s", ds_name, e)
            continue

        references = []
        hypotheses = []
        per_utt_wers = []

        for i, example in enumerate(tqdm(ds, desc=ds_name)):
            item_id = f"{ds_name}_{i}"
            if item_id in completed:
                continue

            try:
                audio_array, sr, ref_text = _get_audio_and_ref(example)
                audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

                result = stt_client.recognize_batch(audio_bytes, sr)
                hyp = result["transcript"]

                utt_wer = jiwer.wer(
                    ref_text, hyp,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )
                utt_cer = jiwer.cer(
                    ref_text, hyp,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )

                record = {
                    "id": item_id,
                    "dataset": ds_name,
                    "reference": ref_text,
                    "hypothesis": hyp,
                    "wer": utt_wer,
                    "cer": utt_cer,
                    "confidence": result["confidence"],
                    "elapsed_s": result["elapsed_s"],
                }
                write_jsonl(jsonl_path, record)

                references.append(ref_text)
                hypotheses.append(hyp)
                per_utt_wers.append(utt_wer)
            except Exception as e:
                logger.warning("Error on %s item %d: %s", ds_name, i, e)

        if not references:
            logger.warning("No results for %s", ds_name)
            continue

        # Aggregate metrics
        agg_wer = jiwer.wer(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER,
            hypothesis_transform=NORMALIZE_FOR_WER,
        )
        agg_cer = jiwer.cer(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER,
            hypothesis_transform=NORMALIZE_FOR_WER,
        )
        agg_mer = jiwer.mer(references, hypotheses)
        agg_wil = jiwer.wil(references, hypotheses)

        # SER: fraction of utterances with at least 1 error
        ser = sum(1 for w in per_utt_wers if w > 0) / len(per_utt_wers)

        # Bootstrap CI on per-utterance WER
        mean_wer, ci_lo, ci_hi = bootstrap_ci(per_utt_wers, n_bootstrap=config.evaluation.bootstrap_n)

        threshold = THRESHOLDS.get(ds_name, {})
        if agg_wer <= threshold.get("pass", 0):
            verdict = "PASS"
        elif agg_wer <= threshold.get("acceptable", 1.0):
            verdict = "FLAG"
        else:
            verdict = "FAIL"

        row = {
            "dataset": ds_info["label"],
            "n_utterances": len(references),
            "wer": round(agg_wer, 4),
            "cer": round(agg_cer, 4),
            "mer": round(agg_mer, 4),
            "wil": round(agg_wil, 4),
            "ser": round(ser, 4),
            "wer_ci_lo": round(ci_lo, 4),
            "wer_ci_hi": round(ci_hi, 4),
            "verdict": verdict,
        }
        summary_rows.append(row)
        logger.info(
            "%s: WER=%.2f%% [%.2f%%–%.2f%%] CER=%.2f%% SER=%.2f%% → %s",
            ds_info["label"],
            agg_wer * 100, ci_lo * 100, ci_hi * 100,
            agg_cer * 100, ser * 100, verdict,
        )

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.1", "name": "clean_speech_accuracy", "results": summary_rows}
