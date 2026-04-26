"""Test 1.1 — Clean Speech Accuracy.

Establish the accuracy baseline across diverse English speech styles.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import jiwer
import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import (
    NORMALIZE_FOR_WER,
    NORMALIZE_FOR_WER_AGG,
    bootstrap_ci,
    get_completed_ids,
    load_dataset_tmp,
    read_jsonl,
    save_summary_csv,
    write_jsonl,
)

logger = logging.getLogger("eval.stt.accuracy")

DATASETS = [
    {"name": "librispeech_clean", "hf": ("librispeech_asr", "clean"), "split": "test", "label": "LibriSpeech clean"},
    {"name": "librispeech_other", "hf": ("librispeech_asr", "other"), "split": "test", "label": "LibriSpeech other"},
    {"name": "voxpopuli_en", "hf": ("facebook/voxpopuli", "en"), "split": "test", "label": "VoxPopuli EN"},
    {"name": "gigaspeech", "hf": ("speechcolab/gigaspeech", "xs"), "split": "test", "label": "GigaSpeech"},
    {"name": "spgispeech", "hf": ("kensho/spgispeech", "S"), "split": "validation", "label": "SPGISpeech"},
]

THRESHOLDS = {
    "librispeech_clean": {"pass": 0.05, "good": 0.07, "acceptable": 0.10},
    "librispeech_other": {"pass": 0.10, "good": 0.15, "acceptable": 0.20},
    "voxpopuli_en": {"pass": 0.08, "good": 0.12, "acceptable": 0.18},
    "gigaspeech": {"pass": 0.12, "good": 0.16, "acceptable": 0.22},
    "spgispeech": {"pass": 0.08, "good": 0.12, "acceptable": 0.18},
}


_GIGASPEECH_PUNCT = {"<COMMA>": ",", "<PERIOD>": ".", "<QUESTIONMARK>": "?", "<EXCLAMATIONPOINT>": "!"}
_GIGASPEECH_NOISE = re.compile(r"<(?:NOISE|MUSIC|SIL)>")
_GIGASPEECH_SKIP  = re.compile(r"^<OTHER>$")


def _normalize_gigaspeech_ref(text: str) -> str:
    for tag, punct in _GIGASPEECH_PUNCT.items():
        text = text.replace(tag, punct)
    text = _GIGASPEECH_NOISE.sub("", text).strip()
    return text


def _get_audio_and_ref(example: dict) -> tuple[np.ndarray, int, str]:
    audio = example["audio"]
    array = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    ref = (example.get("normalized_text") or
           example.get("norm_text") or
           example.get("text") or
           example.get("transcript") or
           example.get("sentence", ""))
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

        path, name = ds_info["hf"]
        extra = {"trust_remote_code": True} if path.startswith("mozilla-foundation/") else {}
        try:
            with load_dataset_tmp(path, ds_info["split"], name=name, **extra) as examples:
                pass
        except Exception as e:
            logger.error("Failed to load %s: %s", ds_name, e)
            continue

        references = []
        hypotheses = []
        per_utt_wers = []

        for i, example in enumerate(tqdm(examples, desc=ds_name)):
            item_id = f"{ds_name}_{i}"
            if item_id in completed:
                continue

            try:
                audio_array, sr, ref_text = _get_audio_and_ref(example)
                if ds_name == "gigaspeech":
                    ref_text = _normalize_gigaspeech_ref(ref_text)
                    if not ref_text or _GIGASPEECH_SKIP.match(ref_text):
                        continue
                if not ref_text.strip():
                    continue
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
            # All items were already completed on a prior run — reconstruct from JSONL.
            if jsonl_path.exists():
                for rec in read_jsonl(jsonl_path):
                    ref = rec.get("reference", "")
                    if ds_name == "gigaspeech":
                        ref = _normalize_gigaspeech_ref(ref)
                        if not ref or _GIGASPEECH_SKIP.match(ref):
                            continue
                    if ref and rec.get("hypothesis") is not None:
                        references.append(ref)
                        hypotheses.append(rec["hypothesis"])
                        # Stored per-utt WER for GigaSpeech is inflated by tags — recompute.
                        if ds_name == "gigaspeech":
                            w = jiwer.wer(ref, rec["hypothesis"],
                                          reference_transform=NORMALIZE_FOR_WER,
                                          hypothesis_transform=NORMALIZE_FOR_WER)
                            per_utt_wers.append(w)
                        else:
                            per_utt_wers.append(rec.get("wer", 0.0))
            if not references:
                logger.warning("No results for %s", ds_name)
                continue

        # Aggregate metrics — use NORMALIZE_FOR_WER_AGG (no RemoveEmptyStrings)
        # so empty refs/hyps don't shorten one list and cause a length mismatch.
        agg_wer = jiwer.wer(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER_AGG,
            hypothesis_transform=NORMALIZE_FOR_WER_AGG,
        )
        agg_cer = jiwer.cer(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER_AGG,
            hypothesis_transform=NORMALIZE_FOR_WER_AGG,
        )
        agg_mer = jiwer.mer(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER_AGG,
            hypothesis_transform=NORMALIZE_FOR_WER_AGG,
        )
        agg_wil = jiwer.wil(
            references, hypotheses,
            reference_transform=NORMALIZE_FOR_WER_AGG,
            hypothesis_transform=NORMALIZE_FOR_WER_AGG,
        )

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
