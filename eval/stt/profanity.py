"""Test 1.17 — Profanity Filter (serving-layer feature honor check).

``RecognitionConfig.profanity_filter`` asks the recognizer to mask profanity in
the transcript (Riva replaces offending words with asterisks). The honor
question: does the gateway mask, or pass the flag through to a model that never
implemented it? We self-construct the clips — TTS-synthesize short utterances
containing mild profanity — then recognize each twice: ``profanity_filter=False``
vs ``True``. Masking that appears only in the filtered pass is the honored
signal; byte-identical transcripts mean ignored.

Audio is generated in-run from the live TTS endpoint into this test's own output
dir (no committed corpus). Utterances use only mild, common expletives for a
legitimate masking-behavior probe.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.tts.client import TTSClient, _call_with_deadline, _is_permanent_tts_error
from eval.utils import get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.profanity")

# Mild expletives in neutral carrier sentences — enough to trigger a masking
# filter without being gratuitous.
CASES = [
    ("damn",  "Oh damn, I missed the morning train again."),
    ("hell",  "What the hell happened to the server last night."),
    ("crap",  "This is a load of crap and it never works."),
    ("ass",   "He fell flat on his ass during the demo."),
    ("bastard", "That stubborn bastard of a bug took all day."),
]

MASK_RE = re.compile(r"\*+|#+|<profanity>|\[profanity\]", re.IGNORECASE)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "profanity"
    results_dir.mkdir(parents=True, exist_ok=True)
    stt_client = STTClient(config)
    tts_client = TTSClient(config)
    sr = tts_client.sample_rate
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.17: Profanity Filter ===")

    completed = get_completed_ids(jsonl_path)

    for word, sentence in tqdm(CASES, desc="profanity"):
        item_id = f"pf_{word}"
        if item_id in completed:
            continue
        try:
            synth = _call_with_deadline(
                lambda: tts_client.synthesize_batch(sentence),
                timeout_s=config.tts.request_timeout_s,
            )
            audio_bytes = synth["audio_bytes"]
        except Exception as e:
            if _is_permanent_tts_error(e):
                write_jsonl(jsonl_path, {"id": item_id, "word": word, "status": "tts_failed"})
            logger.warning("TTS failed for %s: %s", word, e)
            continue

        try:
            unfiltered = stt_client.recognize_batch(audio_bytes, sr, profanity_filter=False)["transcript"]
            filtered = stt_client.recognize_batch(audio_bytes, sr, profanity_filter=True)["transcript"]
        except Exception as e:
            logger.warning("STT failed for %s: %s", word, e)
            write_jsonl(jsonl_path, {"id": item_id, "word": word, "status": "stt_failed"})
            continue

        word_in_unfiltered = word.lower() in unfiltered.lower()
        word_in_filtered = word.lower() in filtered.lower()
        write_jsonl(jsonl_path, {
            "id": item_id,
            "word": word,
            "status": "ok",
            "word_in_unfiltered": word_in_unfiltered,
            "word_in_filtered": word_in_filtered,
            "mask_in_filtered": bool(MASK_RE.search(filtered)),
            "masked": word_in_unfiltered and (not word_in_filtered or bool(MASK_RE.search(filtered))),
            "changed": unfiltered != filtered,
            "unfiltered_transcript": unfiltered,
            "filtered_transcript": filtered,
        })

    recs = [r for r in read_jsonl(jsonl_path) if r.get("status") == "ok"]
    n = len(recs)
    if n == 0:
        summary = {"verdict": "no_data", "n_cases": 0}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.17", "name": "profanity", "results": summary}

    recognizable = sum(1 for r in recs if r.get("word_in_unfiltered"))
    masked = sum(1 for r in recs if r.get("masked"))
    changed = sum(1 for r in recs if r.get("changed"))

    if masked > 0:
        verdict = "honored"              # filter masked words the plain pass surfaced
    elif recognizable == 0:
        verdict = "inconclusive"         # model never transcribed the target words
    elif changed == 0:
        verdict = "ignored"              # identical output — flag had no effect
    else:
        verdict = "inconclusive"

    summary = {
        "verdict": verdict,
        "n_cases": n,
        "words_recognized_plain": recognizable,
        "words_masked_when_filtered": masked,
        "transcripts_changed": changed,
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("Profanity-filter verdict=%s recognized=%d masked=%d", verdict, recognizable, masked)
    return {"test": "1.17", "name": "profanity", "results": summary}
