"""Test 2.1 — Automated Naturalness (UTMOS / DNSMOS / NISQA).

**Multilingual caveat:** UTMOS was validated on English (with Chinese only as an
out-of-domain check), so it is **not reported as a naturalness score for any
non-English language** — for those we record ``utmos: null`` with a note. DNSMOS
is a signal-quality predictor that is language-agnostic, so it is reported for
every language. English behaviour is byte-for-byte unchanged (UTMOS + DNSMOS).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.data.tts_test_sets import get_naturalness_sentences
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref
from eval.tts.client import TTSClient
from eval.tts.dnsmos import score_dnsmos
from eval.tts.tts_languages import resolve_tts_languages
from eval.tts.utmos import release_utmos, score_utmos
from eval.utils import compute_percentiles, read_jsonl, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.tts.naturalness")

_UTMOS_NOTE = "UTMOS validated on English only (zh OOD); not a valid naturalness score for this language."


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "tts" / "naturalness"
    results_dir.mkdir(parents=True, exist_ok=True)

    tts_client = TTSClient(config)
    logger.info("=== Test 2.1: Automated Naturalness ===")

    english = _run_english(tts_client, results_dir)

    per_language: list[dict] = []
    langs = resolve_tts_languages(
        getattr(config.tts, "multilingual_languages", []),
        getattr(config.tts, "voice_overrides", {}),
    )
    for lang in langs:
        if lang["key"] == "en":
            continue  # English is the fixed-sentence pass above
        per_language.append(_run_language(tts_client, lang, results_dir))

    results = {"test": "2.1", "name": "naturalness", "results": english}
    if per_language:
        results["per_language"] = per_language
    return results


def _run_english(tts_client: TTSClient, results_dir: Path) -> dict:
    jsonl_path = results_dir / "calls.jsonl"

    sentences = get_naturalness_sentences()
    logger.info("Evaluating %d sentences", len(sentences))

    # Load existing records; re-process any that are missing UTMOS scores.
    existing_records: dict[str, dict] = {}
    for r in read_jsonl(jsonl_path):
        if "id" in r:
            existing_records[r["id"]] = r

    utmos_null_ids = {id_ for id_, r in existing_records.items() if r.get("utmos") is None}
    fully_done = set(existing_records.keys()) - utmos_null_ids

    if utmos_null_ids:
        logger.info(
            "%d records missing UTMOS — will re-synthesise to score them "
            "(existing DNSMOS cached)",
            len(utmos_null_ids),
        )

    for i, text in enumerate(tqdm(sentences, desc="Naturalness")):
        item_id = f"nat_{i}"
        if item_id in fully_done:
            continue

        wav_path = results_dir / f"synth_{i:04d}.wav"
        try:
            result = tts_client.save_synthesis(text, str(wav_path))

            utmos = score_utmos(str(wav_path))

            # Re-use cached DNSMOS if this was a UTMOS-only re-score pass.
            existing = existing_records.get(item_id)
            if existing and existing.get("dnsmos_ovrl") is not None:
                dnsmos = {
                    "ovrl": existing["dnsmos_ovrl"],
                    "sig": existing["dnsmos_sig"],
                    "bak": existing["dnsmos_bak"],
                }
            else:
                dnsmos = score_dnsmos(str(wav_path))

            wav_path.unlink(missing_ok=True)

            record = {
                "id": item_id,
                "text": text,
                "audio_duration_s": result["audio_duration"],
                "utmos": utmos,
                "dnsmos_ovrl": dnsmos["ovrl"] if dnsmos else None,
                "dnsmos_sig": dnsmos["sig"] if dnsmos else None,
                "dnsmos_bak": dnsmos["bak"] if dnsmos else None,
            }
            write_jsonl(jsonl_path, record)
            existing_records[item_id] = record

        except Exception as e:
            logger.warning("Failed sentence %d: %s", i, e)

    # Deduplicate JSONL (keep latest record per ID) if we did any re-scoring.
    if utmos_null_ids:
        with jsonl_path.open("w") as f:
            for r in existing_records.values():
                f.write(json.dumps(r, default=str) + "\n")

    # Collect scores from all records.
    utmos_scores, dnsmos_ovrl_scores, dnsmos_sig_scores, dnsmos_bak_scores = [], [], [], []
    for rec in existing_records.values():
        if rec.get("utmos") is not None:
            utmos_scores.append(rec["utmos"])
        if rec.get("dnsmos_ovrl") is not None:
            dnsmos_ovrl_scores.append(rec["dnsmos_ovrl"])
            dnsmos_sig_scores.append(rec["dnsmos_sig"])
            dnsmos_bak_scores.append(rec["dnsmos_bak"])

    summary = {
        "n_sentences": len(sentences),
        "utmos": compute_percentiles(utmos_scores) if utmos_scores else {},
        "dnsmos_ovrl": compute_percentiles(dnsmos_ovrl_scores) if dnsmos_ovrl_scores else {},
        "dnsmos_sig": compute_percentiles(dnsmos_sig_scores) if dnsmos_sig_scores else {},
        "dnsmos_bak": compute_percentiles(dnsmos_bak_scores) if dnsmos_bak_scores else {},
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    if utmos_scores:
        logger.info("UTMOS: mean=%.3f std=%.3f", np.mean(utmos_scores), np.std(utmos_scores))
    if dnsmos_ovrl_scores:
        logger.info("DNSMOS OVRL: mean=%.3f", np.mean(dnsmos_ovrl_scores))

    release_utmos()
    return summary


def _run_language(tts_client: TTSClient, lang: dict, results_dir: Path, limit: int = 50) -> dict:
    """Per-language naturalness: DNSMOS for every language (language-agnostic),
    UTMOS deliberately omitted (English-biased ⇒ ``utmos: null`` + note)."""
    key = lang["key"]
    jsonl_path = results_dir / f"{key}.jsonl"

    completed = {r["id"] for r in read_jsonl(jsonl_path) if "id" in r}

    try:
        with fleurs_load(lang, limit=limit) as examples:
            texts = [get_audio_and_ref(ex, field="raw_transcription")[2] for ex in examples]
    except Exception as e:
        logger.warning("Naturalness %s: FLEURS load failed: %s", key, e)
        return {"language": key, "status": "load_failed", "reason": str(e)}

    texts = [t for t in texts if t.strip()][:limit]
    records: list[dict] = [r for r in read_jsonl(jsonl_path) if r.get("dnsmos_ovrl") is not None]

    for i, text in enumerate(tqdm(texts, desc=f"naturalness ({key})")):
        item_id = f"nat_{key}_{i}"
        if item_id in completed:
            continue
        wav_path = results_dir / f"synth_{key}_{i:04d}.wav"
        try:
            result = tts_client.save_synthesis(
                text, str(wav_path),
                language_code=lang["language_code"], voice_name=lang["voice_name"],
            )
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                write_jsonl(jsonl_path, {"id": item_id, "language": key,
                                         "text": text, "status": "no_output"})
                continue
            dnsmos = score_dnsmos(str(wav_path))
            wav_path.unlink(missing_ok=True)
            record = {
                "id": item_id,
                "language": key,
                "text": text,
                "audio_duration_s": result["audio_duration"],
                "utmos": None,
                "utmos_note": _UTMOS_NOTE,
                "dnsmos_ovrl": dnsmos["ovrl"] if dnsmos else None,
                "dnsmos_sig": dnsmos["sig"] if dnsmos else None,
                "dnsmos_bak": dnsmos["bak"] if dnsmos else None,
            }
            write_jsonl(jsonl_path, record)
            if record["dnsmos_ovrl"] is not None:
                records.append(record)
        except Exception as e:
            wav_path.unlink(missing_ok=True)
            logger.warning("Failed naturalness %s_%d: %s", key, i, e)

    ovrl = [r["dnsmos_ovrl"] for r in records if r.get("dnsmos_ovrl") is not None]
    sig = [r["dnsmos_sig"] for r in records if r.get("dnsmos_sig") is not None]
    bak = [r["dnsmos_bak"] for r in records if r.get("dnsmos_bak") is not None]

    if not ovrl:
        summary = {"language": key, "status": "no_output", "n_sentences": len(texts)}
        save_summary_csv(results_dir / f"summary_{key}.csv", [summary])
        return summary

    summary = {
        "language": key,
        "n_sentences": len(records),
        "utmos": None,
        "utmos_note": _UTMOS_NOTE,
        "dnsmos_ovrl": compute_percentiles(ovrl),
        "dnsmos_sig": compute_percentiles(sig),
        "dnsmos_bak": compute_percentiles(bak),
    }
    save_summary_csv(results_dir / f"summary_{key}.csv", [summary])
    logger.info("  %s: DNSMOS OVRL mean=%.3f (UTMOS omitted — en-only)", key, np.mean(ovrl))
    return summary
