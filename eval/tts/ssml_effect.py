"""Test 2.10 — SSML Functional Effect (serving-layer feature honor check).

Test 2.7 proves SSML is *accepted* (the call returns audio). This test proves
whether SSML has a *functional effect*: synthesize a plain baseline and an SSML
variant of the same utterance and measure the expected acoustic delta —

  * ``<break time="2s"/>``        → audio_duration noticeably LONGER
  * ``<prosody rate="50%">``      → audio_duration noticeably LONGER (50% is in
                                    Magpie's accepted 25%–250% range, so this is
                                    an *accepted-in-spec* tag, not a malformed one)
  * ``<say-as interpret-as="characters">`` → probes an UNSUPPORTED interpret-as
                                    value; Magpie hard-rejects it and names the
                                    classes it does support — a recorded finding.

Four terminal outcomes per tag, encoding the honor layer:

  * ``honored``  — accepted, measurable effect, no tag leakage.
  * ``literal``  — accepted but the tags are spoken verbatim (leaked into STT).
  * ``ignored``  — accepted, returns audio, but no acoustic effect (the tag is
                   parsed and discarded).
  * ``rejected`` — permanent SSML error: the tag or attribute value is not
                   supported at all (distinct from ``ignored``).

A longer/different output alone is ambiguous: an engine that ignores SSML but
*speaks the tags literally* also lengthens the audio. To disambiguate, each SSML
clip is transcribed and checked for tag-keyword leakage. All SSML is rooted at
``<speak>`` so the Magpie parser does not fail fast. Audio is generated in-run
into this test's own output dir.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.tts.client import TTSClient, _is_permanent_tts_error
from eval.utils import get_completed_ids, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.ssml_effect")

# Each case: tag, plain baseline, SSML variant, expected effect, leak markers.
CASES = [
    {
        "tag": "break",
        "expect": "longer",
        "plain": "<speak>The meeting starts right now.</speak>",
        "ssml": '<speak>The meeting starts<break time="2s"/> right now.</speak>',
        "min_ratio": 1.15,
        "leak_markers": ["break", "time", "two s", "second"],
    },
    {
        "tag": "prosody_rate",
        "expect": "longer",
        "plain": "<speak>The quick brown fox jumps over the lazy dog.</speak>",
        "ssml": '<speak><prosody rate="50%">The quick brown fox jumps over the lazy dog.</prosody></speak>',
        "min_ratio": 1.15,
        "leak_markers": ["prosody", "rate", "fifty", "percent"],
    },
    {
        "tag": "say_as_characters",
        "expect": "differs",
        "plain": "<speak>Reference code 2024.</speak>",
        "ssml": '<speak>Reference code <say-as interpret-as="characters">2024</say-as>.</speak>',
        "min_ratio": 1.10,
        "leak_markers": ["say as", "interpret", "characters", "say-as"],
    },
]


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower())


def _leaked(markers: list[str], plain_tx: str, ssml_tx: str) -> bool:
    p, s = _norm(plain_tx), _norm(ssml_tx)
    return any(m in s and m not in p for m in markers)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "ssml_effect"
    results_dir.mkdir(parents=True, exist_ok=True)
    tts_client = TTSClient(config)
    stt_client = STTClient(config)
    sr = tts_client.sample_rate
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 2.10: SSML Functional Effect ===")

    completed = get_completed_ids(jsonl_path)

    for case in tqdm(CASES, desc="ssml-effect"):
        item_id = f"ssml_{case['tag']}"
        if item_id in completed:
            continue
        try:
            plain = tts_client.synthesize_batch(case["plain"])
            variant = tts_client.synthesize_batch(case["ssml"])
        except TimeoutError as e:
            logger.warning("SSML %s timed out: %s", case["tag"], e)
            write_jsonl(jsonl_path, {"id": item_id, "tag": case["tag"], "status": "timeout"})
            continue
        except Exception as e:
            # A permanent SSML error means the tag / attribute value is not
            # supported at all — a distinct, recorded outcome ("rejected"),
            # not a transient failure.
            if _is_permanent_tts_error(e):
                detail = str(e)
                m = re.search(r"Error message: (.+?)(?:\"|debug_error_string|$)", detail)
                reason = (m.group(1) if m else detail).strip()[:300]
                logger.info("SSML %s rejected (unsupported): %s", case["tag"], reason)
                write_jsonl(jsonl_path, {
                    "id": item_id,
                    "tag": case["tag"],
                    "status": "ok",
                    "expect": case["expect"],
                    "outcome": "rejected",
                    "reject_reason": reason,
                })
            else:
                logger.warning("SSML %s fail: %s", case["tag"], e)
                write_jsonl(jsonl_path, {"id": item_id, "tag": case["tag"], "status": "fail"})
            continue

        plain_dur = plain["audio_duration"]
        ssml_dur = variant["audio_duration"]
        ratio = ssml_dur / plain_dur if plain_dur > 0 else None

        # Transcribe both to detect literal tag leakage.
        try:
            plain_tx = stt_client.recognize_batch(plain["audio_bytes"], sr)["transcript"]
            ssml_tx = stt_client.recognize_batch(variant["audio_bytes"], sr)["transcript"]
        except Exception as e:
            logger.info("Leak-check STT failed for %s: %s", case["tag"], e)
            plain_tx, ssml_tx = "", ""

        leaked = _leaked(case["leak_markers"], plain_tx, ssml_tx)
        if case["expect"] == "longer":
            effect = ratio is not None and ratio >= case["min_ratio"]
        else:  # differs — either direction, or audible byte-length change
            effect = (ratio is not None and (ratio >= case["min_ratio"] or ratio <= 1 / case["min_ratio"]))

        if leaked:
            outcome = "literal"
        elif effect:
            outcome = "honored"
        else:
            outcome = "ignored"

        write_jsonl(jsonl_path, {
            "id": item_id,
            "tag": case["tag"],
            "status": "ok",
            "expect": case["expect"],
            "plain_duration": round(plain_dur, 3),
            "ssml_duration": round(ssml_dur, 3),
            "duration_ratio": round(ratio, 3) if ratio else None,
            "leaked": leaked,
            "outcome": outcome,
            "plain_transcript": plain_tx,
            "ssml_transcript": ssml_tx,
        })

    recs = [r for r in read_jsonl(jsonl_path) if r.get("status") == "ok"]
    n = len(recs)
    if n == 0:
        summary = {"verdict": "no_data", "n_tags": 0}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "2.10", "name": "ssml_effect", "results": summary}

    kinds = ("honored", "literal", "ignored", "rejected")
    outcomes = {o: sum(1 for r in recs if r.get("outcome") == o) for o in kinds}
    # Overall verdict: honored if any tag had a real effect; else the dominant
    # non-honored outcome. "ignored" and "rejected" are both negative findings
    # (accepted-but-inert vs not-supported) — distinguished in the per-tag detail.
    if outcomes["honored"] >= (n + 1) // 2:
        verdict = "honored"
    elif outcomes["honored"] > 0:
        verdict = "mixed"
    elif outcomes["literal"] >= (n + 1) // 2:
        verdict = "literal"
    elif outcomes["rejected"] >= (n + 1) // 2:
        verdict = "rejected"
    elif outcomes["ignored"] >= (n + 1) // 2:
        verdict = "ignored"
    else:
        verdict = "mixed"

    summary = {
        "verdict": verdict,
        "n_tags": n,
        "honored": outcomes["honored"],
        "literal": outcomes["literal"],
        "ignored": outcomes["ignored"],
        "rejected": outcomes["rejected"],
        "per_tag": "; ".join(f"{r['tag']}:{r.get('outcome', '?')}" for r in recs),
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("SSML-effect verdict=%s honored=%d literal=%d ignored=%d rejected=%d",
                verdict, outcomes["honored"], outcomes["literal"],
                outcomes["ignored"], outcomes["rejected"])
    return {"test": "2.10", "name": "ssml_effect", "results": summary}
