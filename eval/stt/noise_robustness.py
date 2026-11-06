"""Test 1.5 — Noise Robustness."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import jiwer
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes, resolve_languages
from eval.utils import NORMALIZE_FOR_WER, NORMALIZE_FOR_WER_AGG, get_completed_ids, load_dataset_tmp, save_summary_csv, score_pair, write_jsonl

logger = logging.getLogger("eval.stt.noise_robustness")

NOISE_TYPES = ["gaussian", "babble", "traffic", "office"]
SNR_LEVELS = [-5, 0, 5, 10, 15, 20]
N_UTTERANCES = 100
# Per-language FLEURS clip budget for the opt-in Tier-A multilingual passes.
# Smaller than the English N_UTTERANCES so adding several languages stays
# tractable against the gateway; languages are only measured when
# stt.multilingual_extra_languages is non-empty.
PER_LANG_NOISE_LIMIT = 30


def _inject_gaussian_noise(audio: np.ndarray, sr: int, snr_db: float) -> np.ndarray:
    from audiomentations import AddGaussianSNR
    aug = AddGaussianSNR(min_snr_db=snr_db, max_snr_db=snr_db, p=1.0)
    return aug(audio, sample_rate=sr)


def _inject_background_noise(
    audio: np.ndarray, sr: int, snr_db: float, noise_dir: str
) -> np.ndarray:
    from audiomentations import AddBackgroundNoise
    aug = AddBackgroundNoise(
        sounds_path=noise_dir,
        min_snr_db=snr_db,
        max_snr_db=snr_db,
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
    """Test 1.5 entry point.

    Always runs the unchanged English LibriSpeech pass. If
    ``stt.multilingual_extra_languages`` is non-empty, each of those languages is
    additionally measured on FLEURS with the same noise grid, scored with
    ``score_pair`` so CJK/Thai use CER. English behaviour (datasets, scoring,
    ``calls.jsonl`` path, summary.csv) is byte-for-byte identical when the list
    is empty.
    """
    results_dir = Path(config.evaluation.results_dir) / "stt" / "noise_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)

    logger.info("=== Test 1.5: Noise Robustness ===")

    summary_rows = _run_english(config, stt_client, results_dir)

    extra = resolve_languages(getattr(config.stt, "multilingual_extra_languages", []) or [])
    for lang in extra:
        try:
            summary_rows.extend(_run_language(lang, config, stt_client, results_dir))
        except Exception as e:  # pragma: no cover - defensive
            logger.error("Language %s failed in 1.5: %s", lang["key"], e)
            summary_rows.append({
                "language": lang["key"], "noise_type": "load_failed",
                "snr_db": "n/a", "wer": None, "delta_rel": None,
            })

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.5", "name": "noise_robustness", "results": summary_rows}


def _run_english(config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """The original English LibriSpeech noise-robustness pass (unchanged)."""
    jsonl_path = results_dir / "calls.jsonl"

    with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=N_UTTERANCES) as subset:
        pass

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

    clean_wer = jiwer.wer(clean_refs, clean_hyps, reference_transform=NORMALIZE_FOR_WER_AGG, hypothesis_transform=NORMALIZE_FOR_WER_AGG) if clean_refs else 0.1

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
                wer = jiwer.wer(refs, hyps, reference_transform=NORMALIZE_FOR_WER_AGG, hypothesis_transform=NORMALIZE_FOR_WER_AGG)
                delta = (wer - clean_wer) / clean_wer * 100 if clean_wer > 0 else 0
                summary_rows.append({
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "wer": round(wer, 4),
                    "delta_rel": round(delta, 1),
                    "n_utterances": len(refs),
                })
                logger.info("  WER=%.2f%% (Δ=%.1f%%)", wer * 100, delta)

    return summary_rows


def _run_language(lang: dict, config: Config, stt_client: STTClient, results_dir: Path) -> list[dict]:
    """Opt-in per-language noise-robustness pass on FLEURS.

    Same noise grid as English, but scored with ``score_pair`` so CJK/Thai get
    CER, and language-hinted recognition. Should any language return empty
    transcripts on this endpoint, it yields a graceful ``no_output`` row, not a
    fabricated score.
    """
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"
    logger.info("--- 1.5 language=%s (FLEURS %s) ---", key, lang["fleurs"])

    try:
        with fleurs_load(lang, limit=PER_LANG_NOISE_LIMIT) as examples:
            pass
    except Exception as e:
        logger.error("FLEURS load failed for %s: %s", key, e)
        return [{"language": key, "noise_type": "load_failed", "snr_db": "n/a", "wer": None, "delta_rel": None}]

    if not examples:
        return [{"language": key, "noise_type": "load_failed", "snr_db": "n/a", "wer": None, "delta_rel": None}]

    noise_base = Path(config.evaluation.data_dir) / "noise"
    noise_dirs = {
        "babble": str(noise_base / "musan" / "noise"),
        "traffic": str(noise_base / "demand" / "STREET"),
        "office": str(noise_base / "demand" / "OFFICE"),
    }

    completed = get_completed_ids(jsonl_path)

    # Clean baseline for this language.
    clean_refs, clean_hyps = [], []
    for ex in examples[:min(len(examples), 20)]:
        audio, sr, ref = get_audio_and_ref(ex)
        try:
            result = stt_client.recognize_batch(pcm16_bytes(audio), sr, language_code=lang["model_lang"])
            clean_refs.append(ref)
            clean_hyps.append(result["transcript"])
        except Exception:
            pass

    non_empty = sum(1 for h in clean_hyps if (h or "").strip())
    if clean_refs and non_empty == 0:
        # Model produces no output for this language on this endpoint — honest row.
        logger.warning("Language %s returns empty transcripts; recording no_output", key)
        return [{"language": key, "noise_type": "no_output", "snr_db": "n/a", "wer": None, "delta_rel": None}]

    clean_score = score_pair(clean_refs, clean_hyps, key) if clean_refs else None
    clean_metric = clean_score["primary"] if clean_score else "wer"
    clean_val = clean_score["primary_value"] if clean_score else None

    rows: list[dict] = [{
        "language": key, "noise_type": "clean", "snr_db": "inf",
        "wer": round(clean_val, 4) if clean_val is not None else None,
        "metric": clean_metric, "delta_rel": 0.0,
    }]

    for noise_type in NOISE_TYPES:
        snr_range = SNR_LEVELS if noise_type == "gaussian" else [s for s in SNR_LEVELS if s >= 0]
        for snr_db in snr_range:
            logger.info("  %s noise=%s SNR=%d dB", key, noise_type, snr_db)
            refs, hyps = [], []
            for i, ex in enumerate(examples):
                item_id = f"{key}_{noise_type}_{snr_db}_{i}"
                if item_id in completed:
                    continue
                audio, sr, ref = get_audio_and_ref(ex)
                try:
                    noisy = _inject_noise(audio, sr, snr_db, noise_type, noise_dirs.get(noise_type))
                    result = stt_client.recognize_batch(pcm16_bytes(noisy), sr, language_code=lang["model_lang"])
                    refs.append(ref)
                    hyps.append(result["transcript"])
                    write_jsonl(jsonl_path, {
                        "id": item_id,
                        "language": key,
                        "noise_type": noise_type,
                        "snr_db": snr_db,
                        "reference": ref,
                        "hypothesis": result["transcript"],
                    })
                except Exception as e:
                    logger.warning("Failed %s %s@%ddB item %d: %s", key, noise_type, snr_db, i, e)

            if refs:
                sc = score_pair(refs, hyps, key)
                val = sc["primary_value"]
                delta = (val - clean_val) / clean_val * 100 if (clean_val and clean_val > 0) else 0
                rows.append({
                    "language": key,
                    "noise_type": noise_type,
                    "snr_db": snr_db,
                    "wer": round(val, 4),
                    "metric": sc["primary"],
                    "delta_rel": round(delta, 1),
                    "n_utterances": len(refs),
                })
                logger.info("  %s %s@%ddB %s=%.2f%% (Δ=%.1f%%)", key, noise_type, snr_db, sc["primary"], val * 100, delta)

    return rows
