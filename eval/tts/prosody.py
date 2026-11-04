"""Test 2.3 — Prosody Analysis (reference-free).

Prosody here is measured **reference-free**: intrusive metrics like F0 RMSE or
F0 correlation "vs. a reference" are impossible for a fixed synthetic voice (no
same-speaker ground truth exists — see 2.4). What *is* measurable and genuinely
useful are the within-utterance dynamics that distinguish natural from
flat/robotic speech:

  * **Intra-utterance F0 std & range** — pitch dynamism *inside* each sentence.
    (The previous version computed the std of per-sentence *mean* F0s, i.e. an
    inter-sentence spread — it did not detect a monotone voice at all. Fixed.)
  * **Acoustic nPVI** — durational variability over successive voiced intervals
    (a rhythm proxy). Tagged ``npvi_method: "acoustic_approx"`` because it is
    derived from the F0 contour, not a forced phonetic alignment.
  * **Pause ratio** — fraction of the utterance that is silence.
  * **WPM** — speaking rate.

These flag monotone pitch, unnatural rhythm, or odd pacing. They are language-
neutral, so when ``tts.multilingual_languages`` is set the same metrics run per
language on FLEURS text synthesised with that language's voice.
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

try:
    import pyworld as pw
    _PYWORLD_AVAILABLE = True
except ImportError:
    pw = None
    _PYWORLD_AVAILABLE = False

from eval.config import Config
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref
from eval.tts.client import TTSClient
from eval.tts.tts_languages import resolve_tts_languages
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.prosody")

_TEST_SENTENCES = [
    "The meeting has been rescheduled to Thursday afternoon at three o'clock.",
    "Please review the attached document before the end of the day.",
    "Our quarterly revenue exceeded expectations by twelve percent.",
    "Can you confirm receipt of the invoice we sent last week?",
    "The new policy takes effect on the first of next month.",
    "She asked whether the deadline could be extended by two days.",
    "Technical issues delayed the deployment by approximately six hours.",
    "We need to finalize the budget before the board meeting on Friday.",
    "All employees are required to complete the training by December.",
    "The project is currently on track and within budget.",
] * 5  # 50 sentences

_FRAME_PERIOD_MS = 5.0  # pyworld dio default


def extract_f0(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Voiced-only F0 values (Hz). Kept for backward compatibility."""
    if not _PYWORLD_AVAILABLE:
        return np.array([])
    wav, _ = librosa.load(audio_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=_FRAME_PERIOD_MS)
    f0 = pw.stonemask(wav, f0, t, sr)
    return f0[f0 > 0]


def _f0_contour(audio_path: str, sr: int = 16000) -> np.ndarray:
    """Full F0 contour (Hz) including unvoiced frames as 0.0."""
    if not _PYWORLD_AVAILABLE:
        return np.array([])
    wav, _ = librosa.load(audio_path, sr=sr)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=_FRAME_PERIOD_MS)
    return pw.stonemask(wav, f0, t, sr)


def _voiced_run_durations_ms(f0_contour: np.ndarray) -> list[float]:
    """Durations (ms) of contiguous voiced runs in the F0 contour.

    Used as an acoustic proxy for 'vocalic intervals' in the nPVI calculation.
    """
    voiced = f0_contour > 0
    durations: list[float] = []
    run = 0
    for v in voiced:
        if v:
            run += 1
        elif run:
            durations.append(run * _FRAME_PERIOD_MS)
            run = 0
    if run:
        durations.append(run * _FRAME_PERIOD_MS)
    return durations


def _npvi(durations: list[float]) -> float | None:
    """Normalized Pairwise Variability Index over successive interval durations.

    nPVI = 100/(m-1) * Σ |d_k − d_{k+1}| / ((d_k + d_{k+1})/2).
    Higher = more durational contrast (more speech-rhythm variability).
    """
    if len(durations) < 2:
        return None
    diffs = []
    for a, b in zip(durations[:-1], durations[1:]):
        denom = (a + b) / 2.0
        if denom > 0:
            diffs.append(abs(a - b) / denom)
    if not diffs:
        return None
    return 100.0 * float(np.mean(diffs))


def _pause_ratio(audio_path: str, sr: int = 16000) -> float | None:
    """Fraction of the utterance that is silence (non-speech)."""
    try:
        wav, _ = librosa.load(audio_path, sr=sr)
    except Exception:
        return None
    if wav.size == 0:
        return None
    intervals = librosa.effects.split(wav, top_db=30)
    speech = int(np.sum(intervals[:, 1] - intervals[:, 0])) if len(intervals) else 0
    total = len(wav)
    if total == 0:
        return None
    return max(0.0, 1.0 - speech / total)


def speaking_rate_wpm(text: str, audio_path: str) -> float:
    duration = librosa.get_duration(path=audio_path)
    word_count = len(text.split())
    return (word_count / duration) * 60 if duration > 0 else 0.0


def _measure_utterance(text: str, synth_path: str) -> dict:
    """Compute all reference-free prosody metrics for one synthesised clip."""
    contour = _f0_contour(synth_path)
    voiced = contour[contour > 0] if contour.size else np.array([])
    f0_mean = float(np.mean(voiced)) if voiced.size else None
    f0_std = float(np.std(voiced)) if voiced.size else None
    f0_range = (
        float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
        if voiced.size else None
    )
    npvi = _npvi(_voiced_run_durations_ms(contour)) if contour.size else None
    return {
        "wpm": round(speaking_rate_wpm(text, synth_path), 1),
        "f0_mean_hz": round(f0_mean, 1) if f0_mean is not None else None,
        "f0_std_intra_hz": round(f0_std, 1) if f0_std is not None else None,
        "f0_range_intra_hz": round(f0_range, 1) if f0_range is not None else None,
        "npvi": round(npvi, 1) if npvi is not None else None,
        "pause_ratio": (lambda p: round(p, 3) if p is not None else None)(_pause_ratio(synth_path)),
    }


def _aggregate(records: list[dict], n_sentences: int, language: str | None = None) -> dict:
    def _mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return round(float(np.mean(vals)), 1) if vals else None

    wpms = [r["wpm"] for r in records if r.get("wpm") is not None]
    summary = {
        "f0_mean_hz": _mean("f0_mean_hz"),
        "f0_std_intra_hz": _mean("f0_std_intra_hz"),
        "f0_range_intra_hz": _mean("f0_range_intra_hz"),
        "npvi": _mean("npvi"),
        "npvi_method": "acoustic_approx",
        "pause_ratio": (
            round(float(np.mean([r["pause_ratio"] for r in records if r.get("pause_ratio") is not None])), 3)
            if any(r.get("pause_ratio") is not None for r in records) else None
        ),
        "wpm_mean": round(float(np.mean(wpms)), 1) if wpms else None,
        "wpm_std": round(float(np.std(wpms)), 1) if wpms else None,
        "n_sentences": n_sentences,
    }
    if language is not None:
        summary = {"language": language, **summary}
    return summary


def _run_english(tts_client: TTSClient, results_dir: Path) -> dict:
    jsonl_path = results_dir / "calls.jsonl"
    records: list[dict] = []
    for i, text in enumerate(tqdm(_TEST_SENTENCES, desc="Prosody (en)")):
        synth_path = results_dir / f"synth_{i:04d}.wav"
        try:
            tts_client.save_synthesis(text, str(synth_path))
            metrics = _measure_utterance(text, str(synth_path))
            record = {"id": f"prosody_{i}", "text": text, **metrics}
            write_jsonl(jsonl_path, record)
            records.append(record)
        except Exception as e:
            logger.warning("Failed prosody %d: %s", i, e)
        finally:
            synth_path.unlink(missing_ok=True)

    summary = _aggregate(records, len(_TEST_SENTENCES))
    save_summary_csv(results_dir / "summary.csv", [summary])
    if summary["f0_mean_hz"] is not None:
        logger.info(
            "Prosody (en): F0 mean=%.1f Hz intra-std=%.1f Hz range=%.1f Hz nPVI=%.1f pause=%.2f",
            summary["f0_mean_hz"], summary["f0_std_intra_hz"] or 0,
            summary["f0_range_intra_hz"] or 0, summary["npvi"] or 0,
            summary["pause_ratio"] or 0,
        )
    return summary


def _run_language(tts_client: TTSClient, lang: dict, results_dir: Path, limit: int = 30) -> dict:
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"
    records: list[dict] = []
    try:
        with fleurs_load(lang, limit=limit) as examples:
            texts = [get_audio_and_ref(ex, field="raw_transcription")[2] for ex in examples]
    except Exception as e:
        logger.warning("Prosody %s: FLEURS load failed: %s", key, e)
        return {"language": key, "status": "load_failed", "reason": str(e)}

    texts = [t for t in texts if t.strip()][:limit]
    for i, text in enumerate(tqdm(texts, desc=f"Prosody ({key})")):
        synth_path = results_dir / f"synth_{key}_{i:04d}.wav"
        try:
            tts_client.save_synthesis(
                text, str(synth_path),
                language_code=lang["language_code"], voice_name=lang["voice_name"],
            )
            if synth_path.stat().st_size == 0:
                write_jsonl(jsonl_path, {"id": f"prosody_{key}_{i}", "language": key,
                                         "text": text, "status": "no_output"})
                continue
            metrics = _measure_utterance(text, str(synth_path))
            record = {"id": f"prosody_{key}_{i}", "language": key, "text": text, **metrics}
            write_jsonl(jsonl_path, record)
            records.append(record)
        except Exception as e:
            logger.warning("Failed prosody %s %d: %s", key, i, e)
            write_jsonl(jsonl_path, {"id": f"prosody_{key}_{i}", "language": key,
                                     "text": text, "status": "error", "error": str(e)})
        finally:
            synth_path.unlink(missing_ok=True)

    if not records:
        return {"language": key, "status": "no_output", "n_sentences": len(texts)}
    summary = _aggregate(records, len(texts), language=key)
    save_summary_csv(results_dir / f"summary_{key}.csv", [summary])
    return summary


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "prosody"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    logger.info("=== Test 2.3: Prosody (reference-free) ===")

    english = _run_english(tts_client, results_dir)

    per_language: list[dict] = []
    langs = resolve_tts_languages(
        getattr(config.tts, "multilingual_languages", []),
        getattr(config.tts, "voice_overrides", {}),
    )
    for lang in langs:
        if lang["key"] == "en":
            continue  # English already covered by the fixed-sentence pass
        per_language.append(_run_language(tts_client, lang, results_dir))

    results = {"test": "2.3", "name": "prosody", "results": english}
    if per_language:
        results["per_language"] = per_language
    return results
