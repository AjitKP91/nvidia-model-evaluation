"""Test 2.3 — Prosody (F0 / Duration / Rate / nPVI)."""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from scipy.stats import pearsonr

try:
    import pyworld as pw
    _PYWORLD_AVAILABLE = True
except ImportError:
    pw = None
    _PYWORLD_AVAILABLE = False
from tqdm import tqdm

from eval.config import Config
from eval.tts.client import TTSClient
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.prosody")


def extract_f0(audio_path: str, sr: int = 16000) -> np.ndarray:
    if not _PYWORLD_AVAILABLE:
        return np.array([])
    wav, actual_sr = librosa.load(audio_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr)
    f0 = pw.stonemask(wav, f0, t, sr)
    voiced = f0 > 0
    return f0[voiced]


def speaking_rate_wpm(text: str, audio_path: str) -> float:
    duration = librosa.get_duration(path=audio_path)
    word_count = len(text.split())
    return (word_count / duration) * 60 if duration > 0 else 0


def npvi(durations: list[float]) -> float:
    if len(durations) < 2:
        return 0.0
    pairs = [(durations[i], durations[i + 1]) for i in range(len(durations) - 1)]
    return 100 * sum(abs(a - b) / ((a + b) / 2) for a, b in pairs if (a + b) > 0) / len(pairs)


def _get_ljspeech_matched_sentences() -> list[dict]:
    """Return LJSpeech sentences for prosody comparison."""
    try:
        from datasets import load_dataset
        lj = load_dataset("keithito/lj_speech", split="train")
        sentences = []
        for i, ex in enumerate(lj):
            if i >= 50:
                break
            sentences.append({
                "id": ex.get("id", f"LJ{i:04d}"),
                "text": ex.get("normalized_text") or ex.get("text", ""),
                "audio": ex["audio"],
            })
        return sentences
    except Exception as e:
        logger.warning("LJSpeech not available: %s", e)
        return []


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "prosody"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.3: Prosody ===")

    lj_sentences = _get_ljspeech_matched_sentences()
    if not lj_sentences:
        logger.warning("No LJSpeech data — running prosody without reference")

    f0_rmses = []
    f0_correlations = []
    speaking_rates = []
    all_records = []

    for i, item in enumerate(tqdm(lj_sentences or [], desc="Prosody")):
        text = item["text"]
        synth_path = results_dir / f"synth_{i:04d}.wav"

        try:
            result = tts_client.save_synthesis(text, str(synth_path))
            wpm = speaking_rate_wpm(text, str(synth_path))
            speaking_rates.append(wpm)

            # F0 from synthesized
            f0_syn = extract_f0(str(synth_path))

            record = {
                "id": f"prosody_{i}",
                "text": text,
                "wpm": round(wpm, 1),
                "syn_mean_f0": round(float(np.mean(f0_syn)), 1) if len(f0_syn) > 0 else None,
            }

            # F0 from reference (LJSpeech)
            if item.get("audio"):
                import tempfile, soundfile as sf
                ref_audio = np.array(item["audio"]["array"], dtype=np.float32)
                ref_sr = item["audio"]["sampling_rate"]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, ref_audio, ref_sr)
                    f0_ref = extract_f0(tmp.name)
                    Path(tmp.name).unlink(missing_ok=True)

                if len(f0_ref) > 0 and len(f0_syn) > 0:
                    min_len = min(len(f0_ref), len(f0_syn))
                    rmse = float(np.sqrt(np.mean((f0_ref[:min_len] - f0_syn[:min_len]) ** 2)))
                    corr, _ = pearsonr(f0_ref[:min_len], f0_syn[:min_len])

                    f0_rmses.append(rmse)
                    f0_correlations.append(float(corr))
                    record["f0_rmse"] = round(rmse, 2)
                    record["f0_pearson_r"] = round(float(corr), 4)
                    record["ref_mean_f0"] = round(float(np.mean(f0_ref)), 1)

            write_jsonl(jsonl_path, record)
            all_records.append(record)

        except Exception as e:
            logger.warning("Failed prosody %d: %s", i, e)

    # If no LJSpeech, synthesize standalone sentences for speaking rate
    if not lj_sentences:
        standalone_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        ] * 10
        for i, text in enumerate(tqdm(standalone_texts, desc="WPM only")):
            synth_path = results_dir / f"standalone_{i:04d}.wav"
            try:
                result = tts_client.save_synthesis(text, str(synth_path))
                wpm = speaking_rate_wpm(text, str(synth_path))
                speaking_rates.append(wpm)
            except Exception:
                pass

    summary = {
        "f0_rmse_mean": round(np.mean(f0_rmses), 2) if f0_rmses else None,
        "f0_pearson_r_mean": round(np.mean(f0_correlations), 4) if f0_correlations else None,
        "speaking_rate_wpm_mean": round(np.mean(speaking_rates), 1) if speaking_rates else None,
        "speaking_rate_wpm_std": round(np.std(speaking_rates), 1) if speaking_rates else None,
        "n_compared": len(f0_rmses),
        "n_wpm_samples": len(speaking_rates),
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if f0_rmses:
        logger.info("F0 RMSE: %.1f Hz, F0 Pearson r: %.3f", np.mean(f0_rmses), np.mean(f0_correlations))
    if speaking_rates:
        logger.info("Speaking rate: %.0f ± %.0f WPM", np.mean(speaking_rates), np.std(speaking_rates))

    return {"test": "2.3", "name": "prosody", "results": summary}
