"""Test 1.7 — Long-Form Audio."""
from __future__ import annotations

import logging
import re
import tempfile
from collections import Counter
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.long_form")

CHUNK_DURATION_S = 30


def _detect_hallucinations(reference: str, hypothesis: str, ngram_size: int = 5) -> dict:
    """Flag n-gram sequences in hypothesis absent from reference."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    ref_ngrams = set()
    for i in range(len(ref_words) - ngram_size + 1):
        ref_ngrams.add(tuple(ref_words[i : i + ngram_size]))

    hallucinated_ngrams = []
    for i in range(len(hyp_words) - ngram_size + 1):
        ng = tuple(hyp_words[i : i + ngram_size])
        if ng not in ref_ngrams:
            hallucinated_ngrams.append(" ".join(ng))

    hallucinated_tokens = set()
    for ng_str in hallucinated_ngrams:
        hallucinated_tokens.update(ng_str.split())

    return {
        "hallucinated_ngrams": hallucinated_ngrams[:20],
        "hallucination_rate": len(hallucinated_tokens) / len(hyp_words) if hyp_words else 0,
    }


def _detect_repetitions(hypothesis: str, ngram_size: int = 5) -> dict:
    """Count repeated n-gram sequences."""
    words = hypothesis.lower().split()
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngrams.append(tuple(words[i : i + ngram_size]))

    counts = Counter(ngrams)
    repeated = {" ".join(k): v for k, v in counts.items() if v > 1}
    total_ngrams = len(ngrams) if ngrams else 1

    return {
        "repeated_ngrams": dict(list(repeated.items())[:20]),
        "repetition_rate": sum(v - 1 for v in counts.values() if v > 1) / total_ngrams,
    }


def _chunk_audio(audio: np.ndarray, sr: int, chunk_s: float) -> list[np.ndarray]:
    chunk_samples = int(sr * chunk_s)
    return [audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)]


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "long_form"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.7: Long-Form Audio ===")

    # Load TED-LIUM full talks
    try:
        ds = load_dataset("LIUM/tedlium", "release3", split="test", token=True)
        items = list(ds.select(range(min(50, len(ds)))))
    except Exception:
        items = []
        logger.warning("TED-LIUM not available")

    # Also create concatenated LibriSpeech files
    try:
        ls_ds = load_dataset("librispeech_asr", "clean", split="test.clean", token=True)
        concat_groups = []
        for start in range(0, min(100, len(ls_ds)), 10):
            group = list(ls_ds.select(range(start, min(start + 10, len(ls_ds)))))
            concat_groups.append(group)
    except Exception:
        concat_groups = []

    summary_rows = []

    # Process existing long-form audio
    for i, example in enumerate(tqdm(items, desc="Long-form")):
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("norm_text") or example.get("text", "")
        duration_s = len(audio) / sr

        audio_bytes = (audio * 32768).astype(np.int16).tobytes()

        try:
            # Full-file transcription
            full_result = stt_client.recognize_batch(audio_bytes, sr)
            full_hyp = full_result["transcript"]
            full_wer = jiwer.wer(ref, full_hyp, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            # Chunked transcription
            chunks = _chunk_audio(audio, sr, CHUNK_DURATION_S)
            chunk_hyps = []
            for chunk in chunks:
                chunk_bytes = (chunk * 32768).astype(np.int16).tobytes()
                cr = stt_client.recognize_batch(chunk_bytes, sr)
                chunk_hyps.append(cr["transcript"])
            chunked_hyp = " ".join(chunk_hyps)
            chunked_wer = jiwer.wer(ref, chunked_hyp, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            hall = _detect_hallucinations(ref, full_hyp)
            rep = _detect_repetitions(full_hyp)

            record = {
                "id": f"longform_{i}",
                "duration_s": duration_s,
                "full_wer": full_wer,
                "chunked_wer": chunked_wer,
                "wer_ratio": full_wer / chunked_wer if chunked_wer > 0 else None,
                "hallucination_rate": hall["hallucination_rate"],
                "repetition_rate": rep["repetition_rate"],
                "elapsed_s": full_result["elapsed_s"],
            }
            write_jsonl(jsonl_path, record)
            summary_rows.append(record)

        except Exception as e:
            logger.warning("Long-form item %d failed: %s", i, e)

    # Process concatenated LibriSpeech
    for gi, group in enumerate(tqdm(concat_groups, desc="Concatenated")):
        all_audio = []
        all_refs = []
        sr = None
        for ex in group:
            a = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            all_audio.append(a)
            all_refs.append(ex.get("norm_text") or ex.get("text", ""))

        if not all_audio:
            continue

        concat_audio = np.concatenate(all_audio)
        concat_ref = " ".join(all_refs)
        concat_bytes = (concat_audio * 32768).astype(np.int16).tobytes()
        duration_s = len(concat_audio) / sr

        try:
            full_result = stt_client.recognize_batch(concat_bytes, sr)
            full_hyp = full_result["transcript"]
            full_wer = jiwer.wer(concat_ref, full_hyp, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            hall = _detect_hallucinations(concat_ref, full_hyp)
            rep = _detect_repetitions(full_hyp)

            record = {
                "id": f"concat_{gi}",
                "duration_s": duration_s,
                "full_wer": full_wer,
                "hallucination_rate": hall["hallucination_rate"],
                "repetition_rate": rep["repetition_rate"],
            }
            write_jsonl(jsonl_path, record)
            summary_rows.append(record)
        except Exception as e:
            logger.warning("Concatenated group %d failed: %s", gi, e)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    if summary_rows:
        avg_hall = np.mean([r["hallucination_rate"] for r in summary_rows])
        avg_rep = np.mean([r["repetition_rate"] for r in summary_rows])
        logger.info("Hallucination rate: %.2f%%, Repetition rate: %.2f%%", avg_hall * 100, avg_rep * 100)

    return {"test": "1.7", "name": "long_form", "results": summary_rows}
