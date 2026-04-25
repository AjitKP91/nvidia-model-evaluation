"""Test 2.1 — Automated Naturalness (UTMOS / DNSMOS / NISQA)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_naturalness_sentences
from eval.tts.client import TTSClient
from eval.tts.dnsmos import score_dnsmos
from eval.tts.utmos import release_utmos, score_utmos
from eval.utils import compute_percentiles, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.naturalness")


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

            utmos = score_utmos(str(wav_path))
            dnsmos = score_dnsmos(str(wav_path))
            wav_path.unlink(missing_ok=True)

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

    release_utmos()

    return {"test": "2.1", "name": "naturalness", "results": summary}
