"""Test 1.5 — Noise Robustness."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import jiwer
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.noise_robustness")

NOISE_TYPES = ["gaussian", "babble", "traffic", "office"]
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
N_UTTERANCES = 100


def _inject_gaussian_noise(audio: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    from audiomentations import AddGaussianNoise
    aug = AddGaussianNoise(min_snr_in_db=snr_db, max_snr_in_db=snr_db, p=1.0)
    return aug(audio, sample_rate=sr)


def _inject_background_noise(
    audio: np.ndarray, sr: int, snr_db: float, noise_dir: str
) -> np.ndarray:
    from audiomentations import AddBackgroundNoise
    aug = AddBackgroundNoise(
        sounds_path=noise_dir,
        min_snr_in_db=snr_db,
        max_snr_in_db=snr_db,
        p=1.0,
    )
    return aug(audio, sample_rate=sr)


def _inject_noise(
    audio: np.ndarray, sr: int, snr_db: float, noise_type: str, noise_dir: str | None
) -> np.ndarray:
    if noise_type == "gaussian":
        return _inject_gaussian_noise(audio, sr, snr_db)
    if noise_dir and Path(noise_dir).exists():
        return _inject_background_noise(audio, sr, snr_db, noise_dir)
    logger.warning("Noise dir not found for %s, falling back to Gaussian", noise_type)
    return _inject_gaussian_noise(audio, sr, snr_db)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "noise_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.5: Noise Robustness ===")

    ds = load_dataset("librispeech_asr", "clean", split="test.clean", token=True)
    subset = list(ds.select(range(min(N_UTTERANCES, len(ds)))))

    noise_base = Path(config.evaluation.data_dir) / "noise"
    noise_dirs = {
        "babble": str(noise_base / "musan" / "noise"),
        "traffic": str(noise_base / "demand" / "STREET"),
        "office": str(noise_base / "demand" / "OFFICE"),
    }

    # Clean baseline
    logger.info("Computing clean baseline WER...")
    clean_refs, clean_hyps = [], []
    for ex in tqdm(subset[:50], desc="Clean baseline"):
        audio = np.array(ex["audio"]["array"], dtype=np.float32)
        sr = ex["audio"]["sampling_rate"]
        ref = ex.get("norm_text") or ex.get("text", "")
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, sr)
            clean_refs.append(ref)
            clean_hyps.append(result["transcript"])
        except Exception:
            pass

    clean_wer = jiwer.wer(clean_refs, clean_hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER) if clean_refs else 0.1

    # Noisy conditions
    summary_rows = [{"noise_type": "clean", "snr_db": "inf", "wer": round(clean_wer, 4), "delta_rel": 0.0}]
    completed = get_completed_ids(jsonl_path)

    for noise_type in NOISE_TYPES:
        snr_range = SNR_LEVELS if noise_type == "gaussian" else [s for s in SNR_LEVELS if s >= 0]

        for snr_db in snr_range:
            logger.info("Noise=%s SNR=%d dB", noise_type, snr_db)
            refs, hyps = [], []

            for i, ex in enumerate(tqdm(subset, desc=f"{noise_type}@{snr_db}dB")):
                item_id = f"{noise_type}_{snr_db}_{i}"
                if item_id in completed:
                    continue
                audio = np.array(ex["audio"]["array"], dtype=np.float32)
                sr = ex["audio"]["sampling_rate"]
                ref = ex.get("norm_text") or ex.get("text", "")

                try:
                    noisy = _inject_noise(audio, sr, snr_db, noise_type, noise_dirs.get(noise_type))
                    noisy_bytes = (noisy * 32768).astype(np.int16).tobytes()
                    result = stt_client.recognize_batch(noisy_bytes, sr)
                    refs.append(ref)
                    hyps.append(result["transcript"])

                    write_jsonl(jsonl_path, {
                        "id": f"{noise_type}_{snr_db}_{i}",
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "reference": ref,
                        "hypothesis": result["transcript"],
                    })
                except Exception as e:
                    logger.warning("Failed %s@%ddB item %d: %s", noise_type, snr_db, i, e)

            if refs:
                wer = jiwer.wer(refs, hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
                delta = (wer - clean_wer) / clean_wer * 100 if clean_wer > 0 else 0
                summary_rows.append({
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "wer": round(wer, 4),
                    "delta_rel": round(delta, 1),
                    "n_utterances": len(refs),
                })
                logger.info("  WER=%.2f%% (Δ=%.1f%%)", wer * 100, delta)

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.5", "name": "noise_robustness", "results": summary_rows}
