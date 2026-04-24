"""Test 2.1 — Automated Naturalness (UTMOS / DNSMOS / NISQA)."""
from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_naturalness_sentences
from eval.tts.client import TTSClient
from eval.utils import compute_percentiles, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.naturalness")


def _preload_cuda_runtime() -> None:
    """Preload PyTorch's bundled CUDA runtime so UTMOS can find libcudart.

    Sets LD_LIBRARY_PATH and force-loads all libcudart variants present in
    PyTorch's lib directory. Must run before 'import utmos'.
    """
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        current_path = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib not in current_path:
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{current_path}"
        for name in sorted(os.listdir(torch_lib)):
            if name.startswith("libcudart"):
                try:
                    ctypes.CDLL(os.path.join(torch_lib, name))
                except Exception:
                    pass
    except Exception:
        pass


_preload_cuda_runtime()

# Module-level scorer cache — initialise once, reuse across all sentences.
_utmos_scorer = None
_utmos_available = None
_dnsmos_available = None


def _get_utmos_scorer():
    global _utmos_scorer, _utmos_available
    if _utmos_available is False:
        return None
    if _utmos_scorer is None:
        try:
            from utmos import UTMOSScore
            _utmos_scorer = UTMOSScore()
            _utmos_available = True
            logger.info("UTMOS scorer initialised")
        except Exception as e:
            logger.warning("UTMOS not available: %s", e)
            _utmos_available = False
    return _utmos_scorer


def _score_utmos(audio_path: str) -> float | None:
    scorer = _get_utmos_scorer()
    if scorer is None:
        return None
    try:
        return float(scorer.score(audio_path))
    except Exception as e:
        logger.warning("UTMOS scoring failed: %s", e)
        return None


def _score_dnsmos(audio_path: str) -> dict | None:
    global _dnsmos_available
    if _dnsmos_available is False:
        return None
    try:
        from speechmetrics import relative as sm
        metrics = sm.load("dnsmos")
        result = metrics(audio_path)
        _dnsmos_available = True
        return {
            "ovrl": float(result.get("dnsmos", {}).get("ovrl", 0)),
            "sig": float(result.get("dnsmos", {}).get("sig", 0)),
            "bak": float(result.get("dnsmos", {}).get("bak", 0)),
        }
    except Exception as e:
        if _dnsmos_available is None:
            logger.warning("DNSMOS not available (%s), skipping for all sentences", e)
            _dnsmos_available = False
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "naturalness"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.1: Automated Naturalness ===")

    sentences = get_naturalness_sentences()
    logger.info("Evaluating %d sentences", len(sentences))

    completed = get_completed_ids(jsonl_path)

    utmos_scores = []
    dnsmos_ovrl_scores = []
    dnsmos_sig_scores = []
    dnsmos_bak_scores = []

    for i, text in enumerate(tqdm(sentences, desc="Naturalness")):
        item_id = f"nat_{i}"
        if item_id in completed:
            continue
        wav_path = results_dir / f"synth_{i:04d}.wav"

        try:
            result = tts_client.save_synthesis(text, str(wav_path))

            utmos = _score_utmos(str(wav_path))
            dnsmos = _score_dnsmos(str(wav_path))

            record = {
                "id": f"nat_{i}",
                "text": text,
                "audio_duration_s": result["audio_duration"],
                "utmos": utmos,
                "dnsmos_ovrl": dnsmos["ovrl"] if dnsmos else None,
                "dnsmos_sig": dnsmos["sig"] if dnsmos else None,
                "dnsmos_bak": dnsmos["bak"] if dnsmos else None,
            }
            write_jsonl(jsonl_path, record)

            if utmos is not None:
                utmos_scores.append(utmos)
            if dnsmos:
                dnsmos_ovrl_scores.append(dnsmos["ovrl"])
                dnsmos_sig_scores.append(dnsmos["sig"])
                dnsmos_bak_scores.append(dnsmos["bak"])

        except Exception as e:
            logger.warning("Failed sentence %d: %s", i, e)

    summary = {
        "n_sentences": len(sentences),
        "utmos": compute_percentiles(utmos_scores) if utmos_scores else {},
        "dnsmos_ovrl": compute_percentiles(dnsmos_ovrl_scores) if dnsmos_ovrl_scores else {},
        "dnsmos_sig": compute_percentiles(dnsmos_sig_scores) if dnsmos_sig_scores else {},
        "dnsmos_bak": compute_percentiles(dnsmos_bak_scores) if dnsmos_bak_scores else {},
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if utmos_scores:
        logger.info("UTMOS: mean=%.2f std=%.2f", np.mean(utmos_scores), np.std(utmos_scores))
    if dnsmos_ovrl_scores:
        logger.info("DNSMOS OVRL: mean=%.2f", np.mean(dnsmos_ovrl_scores))

    return {"test": "2.1", "name": "naturalness", "results": summary}
