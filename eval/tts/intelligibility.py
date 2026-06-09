"""Test 2.2 — Intelligibility: ASR Round-Trip WER."""
from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_intelligibility_sentences
from eval.tts.client import TTSClient
from eval.utils import NORMALIZE_FOR_WER, get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.intelligibility")


def _transcribe_whisper(audio_path: str, model=None):
    """Transcribe audio using Whisper large-v3."""
    import whisper
    if model is None:
        model = whisper.load_model("large-v3")
    result = model.transcribe(audio_path)
    return result["text"]


def _seed_categories_from_jsonl(jsonl_path: Path) -> dict[str, dict]:
    """Reconstruct per-category ref/hyp lists from existing JSONL.

    Resume-safe: when every sentence was completed on a prior run, the loop
    appends nothing and the summary would be empty otherwise.
    """
    seeded: dict[str, dict] = defaultdict(lambda: {"refs": [], "hyps": []})
    for r in read_jsonl(jsonl_path):
        cat = r.get("category")
        ref = r.get("original_text")
        hyp = r.get("whisper_transcript")
        if cat and ref is not None and hyp is not None:
            seeded[cat]["refs"].append(ref)
            seeded[cat]["hyps"].append(hyp)
    return seeded


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "intelligibility"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.2: Intelligibility (Round-Trip WER) ===")

    categories = get_intelligibility_sentences()
    logger.info("Categories: %s", list(categories.keys()))

    # Load Whisper once
    import whisper
    logger.info("Loading Whisper large-v3...")
    whisper_model = whisper.load_model("large-v3")

    completed = get_completed_ids(jsonl_path)
    seeded = _seed_categories_from_jsonl(jsonl_path)
    summary_rows = []

    for category, sentences in categories.items():
        logger.info("Category: %s (%d sentences)", category, len(sentences))
        # Seed with anything already on disk so resume-only runs still produce
        # a summary row.
        refs = list(seeded[category]["refs"])
        hyps = list(seeded[category]["hyps"])
        per_sent_wers = []

        for i, text in enumerate(tqdm(sentences, desc=category)):
            item_id = f"{category}_{i}"
            if item_id in completed:
                continue
            wav_path = results_dir / f"synth_{category}_{i:04d}.wav"

            try:
                result = tts_client.save_synthesis(text, str(wav_path))
                hyp = _transcribe_whisper(str(wav_path), whisper_model)
                wav_path.unlink(missing_ok=True)

                wer_val = jiwer.wer(
                    text, hyp,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )
                cer_val = jiwer.cer(
                    text, hyp,
                    reference_transform=NORMALIZE_FOR_WER,
                    hypothesis_transform=NORMALIZE_FOR_WER,
                )

                refs.append(text)
                hyps.append(hyp)
                per_sent_wers.append(wer_val)

                write_jsonl(jsonl_path, {
                    "id": f"{category}_{i}",
                    "category": category,
                    "original_text": text,
                    "whisper_transcript": hyp,
                    "wer": wer_val,
                    "cer": cer_val,
                })
            except Exception as e:
                wav_path.unlink(missing_ok=True)
                logger.warning("Failed %s_%d: %s", category, i, e)

        if refs:
            agg_wer = jiwer.wer(refs, hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
            agg_cer = jiwer.cer(refs, hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)

            row = {
                "category": category,
                "n_sentences": len(refs),
                "round_trip_wer": round(agg_wer, 4),
                "round_trip_cer": round(agg_cer, 4),
            }
            summary_rows.append(row)
            logger.info("  %s: WER=%.2f%% CER=%.2f%%", category, agg_wer * 100, agg_cer * 100)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    # Delete Whisper model cache — large (~3 GB) and not needed after this test.
    logger.info("Releasing Whisper model and cleaning cache...")
    del whisper_model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    whisper_cache = Path.home() / ".cache" / "whisper"
    for f in sorted(whisper_cache.glob("large-v3*")):
        try:
            f.unlink()
            logger.info("Deleted Whisper cache file: %s", f)
        except Exception:
            pass

    return {"test": "2.2", "name": "intelligibility", "results": summary_rows}
