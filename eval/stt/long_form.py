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
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes, resolve_languages
from eval.utils import NORMALIZE_FOR_WER, is_cer_language, load_dataset_tmp, save_summary_csv, score_pair, write_jsonl

logger = logging.getLogger("eval.stt.long_form")

CHUNK_DURATION_S = 30
# Number of FLEURS clips concatenated into one long file per spot-check language.
LANG_CONCAT_GROUP = 10
# How many concatenated long files to build per spot-check language.
LANG_CONCAT_FILES = 3


def _detect_hallucinations(reference: str, hypothesis: str, ngram_size: int = 5,
                           char_level: bool = False) -> dict:
    """Flag n-gram sequences in hypothesis absent from reference.

    English default (char_level=False) tokenises on whitespace, unchanged. For
    CER languages (ja/zh/th — no word boundaries) pass char_level=True so n-grams
    are over characters; callers mark such results ``approximate`` rather than
    reporting a word-based rate that is meaningless for those scripts.
    """
    tokenize = (lambda t: list(t)) if char_level else (lambda t: t.split())
    ref_words = tokenize(reference.lower())
    hyp_words = tokenize(hypothesis.lower())

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


def _detect_repetitions(hypothesis: str, ngram_size: int = 5, char_level: bool = False) -> dict:
    """Count repeated n-gram sequences (whitespace tokens, or chars for CER)."""
    tokenize = (lambda t: list(t)) if char_level else (lambda t: t.split())
    words = tokenize(hypothesis.lower())
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

    # Load VoxPopuli EN as long-ish utterances (replaces LIUM/tedlium — not on HF Hub)
    try:
        with load_dataset_tmp("facebook/voxpopuli", "test", name="en", limit=50) as items:
            pass
    except Exception:
        items = []
        logger.warning("VoxPopuli not available")

    # Also create concatenated LibriSpeech files
    try:
        with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=100) as ls_examples:
            pass
        concat_groups = [ls_examples[i:i+10] for i in range(0, len(ls_examples), 10)]
    except Exception:
        concat_groups = []

    summary_rows = []

    # Process existing long-form audio
    for i, example in enumerate(tqdm(items, desc="Long-form")):
        audio = np.array(example["audio"]["array"], dtype=np.float32)
        sr = example["audio"]["sampling_rate"]
        ref = example.get("normalized_text") or example.get("norm_text") or example.get("text", "")
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

            # Chunked transcription — same 30s windowing as the short-clip loop
            # above. The concat files are the only ones long enough to span
            # multiple chunks, so this is where whole-file vs chunked drift is
            # actually measurable; without it chunked_wer/wer_ratio are blank.
            chunks = _chunk_audio(concat_audio, sr, CHUNK_DURATION_S)
            chunk_hyps = []
            for chunk in chunks:
                chunk_bytes = (chunk * 32768).astype(np.int16).tobytes()
                cr = stt_client.recognize_batch(chunk_bytes, sr)
                chunk_hyps.append(cr["transcript"])
            chunked_hyp = " ".join(chunk_hyps)
            chunked_wer = jiwer.wer(concat_ref, chunked_hyp, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            hall = _detect_hallucinations(concat_ref, full_hyp)
            rep = _detect_repetitions(full_hyp)

            record = {
                "id": f"concat_{gi}",
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
            logger.warning("Concatenated group %d failed: %s", gi, e)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    if summary_rows:
        avg_hall = np.mean([r["hallucination_rate"] for r in summary_rows])
        avg_rep = np.mean([r["repetition_rate"] for r in summary_rows])
        logger.info("Hallucination rate: %.2f%%, Repetition rate: %.2f%%", avg_hall * 100, avg_rep * 100)

    # ---- Opt-in Tier-B spot-check languages (FLEURS concatenation) ----
    spot = resolve_languages(getattr(config.stt, "spot_check_languages", []) or [])
    lang_rows: list[dict] = []
    for lang in spot:
        try:
            lang_rows.extend(_run_language_longform(lang, config, stt_client, results_dir))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Language %s failed in 1.7: %s", lang["key"], e)
    if lang_rows:
        save_summary_csv(results_dir / "summary_by_language.csv", lang_rows)
        summary_rows = summary_rows + lang_rows

    return {"test": "1.7", "name": "long_form", "results": summary_rows}


def _run_language_longform(lang: dict, config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """Opt-in per-language long-form spot-check on concatenated FLEURS clips.

    Builds long files by concatenating ~10 FLEURS clips (same idiom as the
    English concatenated-LibriSpeech path) and scores full-file WER/CER with
    ``score_pair``. For CER languages the hallucination/repetition detectors run
    on characters and are flagged ``approximate`` — never reported as a bogus
    word-based rate. Languages that return empty transcripts record a
    ``no_output`` row.
    """
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"
    cer = is_cer_language(key)
    logger.info("--- 1.7 spot-check language=%s (FLEURS %s, cer=%s) ---", key, lang["fleurs"], cer)

    try:
        with fleurs_load(lang, limit=LANG_CONCAT_GROUP * LANG_CONCAT_FILES) as examples:
            pass
    except Exception as e:
        logger.error("FLEURS load failed for %s: %s", key, e)
        return [{"language": key, "id": "load_failed", "full_wer": None}]

    if not examples:
        return [{"language": key, "id": "load_failed", "full_wer": None}]

    groups = [examples[i:i + LANG_CONCAT_GROUP] for i in range(0, len(examples), LANG_CONCAT_GROUP)]
    rows: list[dict] = []
    non_empty = 0
    for gi, group in enumerate(tqdm(groups, desc=f"lf:{key}")):
        all_audio, all_refs, sr = [], [], None
        for ex in group:
            a, sr, ref = get_audio_and_ref(ex)
            all_audio.append(a)
            all_refs.append(ref)
        if not all_audio:
            continue
        concat_audio = np.concatenate(all_audio)
        concat_ref = " ".join(all_refs)
        duration_s = len(concat_audio) / sr
        try:
            full_result = stt_client.recognize_batch(pcm16_bytes(concat_audio), sr, language_code=lang["model_lang"])
            full_hyp = full_result["transcript"]
            if (full_hyp or "").strip():
                non_empty += 1
            sc = score_pair([concat_ref], [full_hyp], key)
            hall = _detect_hallucinations(concat_ref, full_hyp, char_level=cer)
            rep = _detect_repetitions(full_hyp, char_level=cer)
            record = {
                "id": f"concat_{key}_{gi}",
                "language": key,
                "duration_s": duration_s,
                "full_wer": round(sc["primary_value"], 4),
                "metric": sc["primary"],
                "hallucination_rate": hall["hallucination_rate"],
                "repetition_rate": rep["repetition_rate"],
                "detector_approximate": cer,
            }
            write_jsonl(jsonl_path, record)
            rows.append(record)
        except Exception as e:
            logger.warning("Long-form %s group %d failed: %s", key, gi, e)

    if non_empty == 0:
        logger.warning("Language %s returns empty transcripts; recording no_output", key)
        return [{"language": key, "id": "no_output", "full_wer": None}]
    return rows
