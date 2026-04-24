"""Test 2.4 — Audio Signal Quality (MCD / PESQ / STOI)."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.tts.client import TTSClient
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.signal_quality")

TARGET_SR = 16000


def compute_mcd(ref_path: str, syn_path: str, n_mfcc: int = 13) -> float:
    """Mel Cepstral Distortion via librosa + DTW."""
    from dtw import dtw

    ref, sr = librosa.load(ref_path, sr=22050)
    syn, _ = librosa.load(syn_path, sr=22050)
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).T
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc).T
    alignment = dtw(ref_mfcc, syn_mfcc, dist_method="euclidean")
    return (10.0 * np.sqrt(2) / np.log(10)) * alignment.distance / len(ref_mfcc)


def compute_pesq_score(ref_path: str, syn_path: str) -> float | None:
    try:
        from pesq import pesq
        ref, sr_ref = sf.read(ref_path)
        syn, sr_syn = sf.read(syn_path)
        # Resample to 16kHz
        if sr_ref != TARGET_SR:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=sr_ref, target_sr=TARGET_SR)
        if sr_syn != TARGET_SR:
            syn = librosa.resample(syn.astype(np.float32), orig_sr=sr_syn, target_sr=TARGET_SR)
        min_len = min(len(ref), len(syn))
        return float(pesq(TARGET_SR, ref[:min_len], syn[:min_len], "wb"))
    except Exception as e:
        logger.warning("PESQ failed: %s", e)
        return None


def compute_stoi_score(ref_path: str, syn_path: str) -> float | None:
    try:
        from pystoi import stoi
        ref, sr_ref = sf.read(ref_path)
        syn, sr_syn = sf.read(syn_path)
        if sr_ref != TARGET_SR:
            ref = librosa.resample(ref.astype(np.float32), orig_sr=sr_ref, target_sr=TARGET_SR)
        if sr_syn != TARGET_SR:
            syn = librosa.resample(syn.astype(np.float32), orig_sr=sr_syn, target_sr=TARGET_SR)
        min_len = min(len(ref), len(syn))
        return float(stoi(ref[:min_len], syn[:min_len], TARGET_SR, extended=True))
    except Exception as e:
        logger.warning("STOI failed: %s", e)
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "signal_quality"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.4: Audio Signal Quality ===")

    # Load LJSpeech for matched reference
    try:
        from datasets import load_dataset
        lj = load_dataset("keithito/lj_speech", split="train", trust_remote_code=True)
        lj_items = list(lj.select(range(min(50, len(lj)))))
    except Exception as e:
        logger.warning("LJSpeech not available: %s. Skipping matched-speaker metrics.", e)
        lj_items = []

    if not lj_items:
        logger.warning(
            "No matched-speaker reference available. "
            "PESQ/STOI/MCD require same-speaker reference. "
            "Skipping Test 2.4 — rely on Tests 2.1 and 2.2 for quality."
        )
        return {"test": "2.4", "name": "signal_quality", "results": {"skipped": True}}

    mcd_values = []
    pesq_values = []
    stoi_values = []

    for i, item in enumerate(tqdm(lj_items, desc="Signal quality")):
        text = item.get("normalized_text") or item.get("text", "")
        ref_audio = np.array(item["audio"]["array"], dtype=np.float32)
        ref_sr = item["audio"]["sampling_rate"]

        ref_path = results_dir / f"ref_{i:04d}.wav"
        syn_path = results_dir / f"syn_{i:04d}.wav"

        sf.write(str(ref_path), ref_audio, ref_sr)

        try:
            tts_client.save_synthesis(text, str(syn_path))

            mcd = compute_mcd(str(ref_path), str(syn_path))
            pesq_val = compute_pesq_score(str(ref_path), str(syn_path))
            stoi_val = compute_stoi_score(str(ref_path), str(syn_path))

            mcd_values.append(mcd)
            if pesq_val is not None:
                pesq_values.append(pesq_val)
            if stoi_val is not None:
                stoi_values.append(stoi_val)

            write_jsonl(jsonl_path, {
                "id": f"sigq_{i}",
                "text": text,
                "mcd": round(mcd, 2),
                "pesq": round(pesq_val, 3) if pesq_val else None,
                "stoi": round(stoi_val, 4) if stoi_val else None,
            })
        except Exception as e:
            logger.warning("Failed signal quality %d: %s", i, e)

    summary = {
        "speaker_mismatch_warning": (
            "LJSpeech speaker differs from Magpie Aria. "
            "PESQ/STOI/MCD scores reflect voice mismatch, not quality."
        ),
        "mcd_mean": round(np.mean(mcd_values), 2) if mcd_values else None,
        "pesq_mean": round(np.mean(pesq_values), 3) if pesq_values else None,
        "stoi_mean": round(np.mean(stoi_values), 4) if stoi_values else None,
        "n_samples": len(mcd_values),
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if mcd_values:
        logger.info("MCD: %.1f dB, PESQ: %.2f, STOI: %.3f",
            np.mean(mcd_values),
            np.mean(pesq_values) if pesq_values else 0,
            np.mean(stoi_values) if stoi_values else 0,
        )

    return {"test": "2.4", "name": "signal_quality", "results": summary}
