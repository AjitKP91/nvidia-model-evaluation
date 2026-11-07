"""Test 1.16 — Word Boosting / speech_contexts (serving-layer feature honor check).

``RecognitionConfig.speech_contexts`` lets a caller bias the decoder toward a
set of phrases (boost weight). The honor question: does the gateway apply the
bias, or silently drop it? We self-construct the clips — TTS-synthesize carrier
sentences containing rare/technical proper nouns the base model is likely to
mis-transcribe, then recognize each clip twice: once plain, once with
``speech_contexts=[{phrases:[...], boost:20}]``. A per-phrase hit-rate *increase*
under boosting is the honored signal; identical hit rates mean ignored.

Audio is generated in-run from the live TTS endpoint into this test's own output
dir (no committed corpus).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.tts.client import TTSClient, _call_with_deadline, _is_permanent_tts_error
from eval.utils import get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.word_boosting")

BOOST = 20.0

# Rare / domain proper nouns embedded in natural carrier sentences. These are
# exactly the kind of low-frequency tokens word-boosting is meant to help with.
CASES = [
    ("Nemotron",     "The engineers deployed Nemotron for speech recognition."),
    ("Magpie",       "She configured the Magpie voice for the demo."),
    ("FastConformer", "Our pipeline runs on a FastConformer acoustic model."),
    ("Parakeet",     "They benchmarked Parakeet against the older system."),
    ("Riva",         "The Riva gateway handled every streaming request."),
    ("Joule",        "Customers access the model through Joule each morning."),
    ("Nvidia",       "The team evaluated the Nvidia inference stack."),
    ("Zarvox",       "The assistant was codenamed Zarvox during testing."),
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def _hit(phrase: str, transcript: str) -> bool:
    return _norm(phrase).strip() in _norm(transcript)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "word_boosting"
    results_dir.mkdir(parents=True, exist_ok=True)
    stt_client = STTClient(config)
    tts_client = TTSClient(config)
    sr = tts_client.sample_rate
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.16: Word Boosting (speech_contexts, boost=%.0f) ===", BOOST)

    completed = get_completed_ids(jsonl_path)

    for phrase, sentence in tqdm(CASES, desc="word-boosting"):
        item_id = f"wb_{phrase}"
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
                logger.warning("TTS permanent error for %s: %s", phrase, e)
                write_jsonl(jsonl_path, {"id": item_id, "phrase": phrase, "status": "tts_failed"})
            else:
                logger.warning("TTS failed for %s: %s", phrase, e)
            continue

        try:
            base = stt_client.recognize_batch(audio_bytes, sr)["transcript"]
            boosted = stt_client.recognize_batch(
                audio_bytes, sr,
                speech_contexts=[{"phrases": [phrase], "boost": BOOST}],
            )["transcript"]
        except Exception as e:
            logger.warning("STT failed for %s: %s", phrase, e)
            write_jsonl(jsonl_path, {"id": item_id, "phrase": phrase, "status": "stt_failed"})
            continue

        write_jsonl(jsonl_path, {
            "id": item_id,
            "phrase": phrase,
            "status": "ok",
            "baseline_hit": _hit(phrase, base),
            "boosted_hit": _hit(phrase, boosted),
            "baseline_transcript": base,
            "boosted_transcript": boosted,
        })

    recs = [r for r in read_jsonl(jsonl_path) if r.get("status") == "ok"]
    n = len(recs)
    if n == 0:
        summary = {"verdict": "no_data", "n_cases": 0}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.16", "name": "word_boosting", "results": summary}

    base_hits = sum(1 for r in recs if r.get("baseline_hit"))
    boost_hits = sum(1 for r in recs if r.get("boosted_hit"))
    gained = sum(1 for r in recs if r.get("boosted_hit") and not r.get("baseline_hit"))
    lost = sum(1 for r in recs if r.get("baseline_hit") and not r.get("boosted_hit"))
    changed = sum(1 for r in recs
                  if r.get("baseline_transcript") != r.get("boosted_transcript"))

    if gained > lost and boost_hits > base_hits:
        verdict = "honored"              # boosting lifted recall of target phrases
    elif changed == 0:
        verdict = "ignored"              # byte-identical output — bias dropped
    else:
        verdict = "inconclusive"         # output shifted but not toward the phrases

    summary = {
        "verdict": verdict,
        "n_cases": n,
        "boost": BOOST,
        "baseline_hits": base_hits,
        "boosted_hits": boost_hits,
        "gained": gained,
        "lost": lost,
        "transcripts_changed": changed,
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("Word-boosting verdict=%s baseline_hits=%d boosted_hits=%d gained=%d",
                verdict, base_hits, boost_hits, gained)
    return {"test": "1.16", "name": "word_boosting", "results": summary}
