"""UTMOS22 scorer — lazy-initialised, one model per process.

Strategy: ttseval/utmos PyPI package (requires fairseq + wav2vec_small.pt from HF).
Falls back to unavailable if fairseq is not installed (fairseq is unbuildable on
pip>=24 without --no-build-isolation and a matching CUDA toolkit).
"""
from __future__ import annotations

import gc
import logging

import soundfile as sf

logger = logging.getLogger("eval.tts.utmos")

_utmos_scorer = None  # ("pypi", model) | False
_utmos_available = None


def get_utmos_scorer():
    global _utmos_scorer, _utmos_available
    if _utmos_available is False:
        return None
    if _utmos_scorer is not None:
        return _utmos_scorer

    try:
        from utmos import Score  # type: ignore[import]
        model = Score()
        _utmos_scorer = ("pypi", model)
        _utmos_available = True
        logger.info("UTMOS scorer initialised (ttseval/utmos PyPI package)")
        return _utmos_scorer
    except Exception as e:
        logger.warning("UTMOS not available: %s", e)
        _utmos_available = False
        return None


def score_utmos(audio_path: str) -> float | None:
    result = get_utmos_scorer()
    if result is None:
        return None
    kind, model = result
    try:
        if kind == "pypi":
            return float(model.calculate_wav_file(audio_path))
    except Exception as e:
        logger.warning("UTMOS scoring failed: %s", e)
    return None


def release_utmos() -> None:
    global _utmos_scorer, _utmos_available
    if _utmos_scorer is None:
        return
    _, model = _utmos_scorer
    del model
    _utmos_scorer = None
    _utmos_available = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
