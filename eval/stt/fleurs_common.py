"""Shared FLEURS helpers for per-language STT tests.

Several tests (1.2, 1.5, 1.7, 1.9, 1.10, 1.12) measure language-sensitive
behaviour by deriving conditions from the same FLEURS read-speech corpus
(google/fleurs, CC-BY-4.0). This module factors out the three pieces they all
need so there is exactly one FLEURS code path:

  * resolve_languages()  — map config language keys -> language dicts.
  * fleurs_load()        — download a FLEURS split to memory (contextmanager).
  * get_audio_and_ref()  — pull (audio array, sample rate, reference) per clip.

The model under test (nvidia/nemotron-3.5-asr-streaming-0.6b) auto-detects
language, so lang["model_lang"] is only a hint. The live acceptance probe (1.13)
shows one GA Joule language (Arabic) returns EMPTY transcripts
on this endpoint; callers must therefore treat "no output" as a graceful
no_output result, never a fake score.
"""
from __future__ import annotations

import logging

import numpy as np

from eval.stt.languages import BY_KEY
from eval.utils import load_dataset_tmp

logger = logging.getLogger("eval.stt.fleurs_common")


def resolve_languages(keys: list[str] | None) -> list[dict]:
    """Map config language keys to full language dicts (eval/stt/languages.py).

    Unknown keys are skipped with a warning. Order follows the caller's list.
    Empty/None returns an empty list (callers interpret that as English-only).
    """
    out: list[dict] = []
    for key in keys or []:
        lang = BY_KEY.get(key)
        if lang is None:
            logger.warning("Unknown language key %r (not in languages registry); skipping", key)
            continue
        out.append(lang)
    return out


def fleurs_load(lang: dict, limit: int | None = None):
    """Context manager yielding an in-memory list of FLEURS examples for a lang.

    Mirrors multilingual.py's load. Returns the load_dataset_tmp contextmanager
    so callers use `with fleurs_load(lang, limit) as examples:`. Loading errors
    propagate to the caller (so it can record a load_failed row and continue).
    """
    return load_dataset_tmp(
        "google/fleurs", "test", name=lang["fleurs"],
        limit=limit, trust_remote_code=True,
    )


def get_audio_and_ref(example: dict, field: str = "transcription") -> tuple[np.ndarray, int, str]:
    """Return (float32 audio array, sample_rate, reference) for a FLEURS clip.

    field="transcription" (default) is lightly-normalised — right for WER/CER.
    field="raw_transcription" keeps original casing/punctuation/numerals — use
    it for the output-quality test (1.9). Falls back to the other field if the
    requested one is empty.
    """
    audio = example["audio"]
    array = np.array(audio["array"], dtype=np.float32)
    sr = audio["sampling_rate"]
    if field == "raw_transcription":
        ref = example.get("raw_transcription") or example.get("transcription") or ""
    else:
        ref = example.get("transcription") or example.get("raw_transcription") or ""
    return array, sr, ref


def pcm16_bytes(audio_array: np.ndarray) -> bytes:
    """Convert a float32 [-1, 1] audio array to int16 little-endian PCM bytes.

    Matches the encoding every test uses before recognize_batch().
    """
    return (audio_array * 32768).astype(np.int16).tobytes()
