"""Test 1.12 — Multilingual Accuracy (FLEURS).

Measures transcription accuracy across the languages SAP Joule supports, using
FLEURS (google/fleurs, CC-BY-4.0) as one consistent read-speech corpus that
covers every one of them. The model under evaluation —
NVIDIA-Nemotron-3.5-Transcribe-30B-A4B — supports 40 languages and auto-detects
language, so the `model_lang` tag below is at most a hint; every Joule language
is expected to return output. Languages the endpoint does not accept are
recorded with status "no_output" rather than crashing the run.

Scoring is language-aware (see eval.utils):
  * space-delimited languages  -> WER, after Whisper BasicTextNormalizer
  * CJK / Thai (no word bounds) -> CER, after Whisper BasicTextNormalizer
English uses Whisper's EnglishTextNormalizer, matching Test 1.1 and the
Open ASR Leaderboard, so the en row is comparable cell-for-cell.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref, pcm16_bytes
from eval.stt.languages import JOULE_LANGUAGES
from eval.utils import (
    bootstrap_ci,
    get_completed_ids,
    is_cer_language,
    read_jsonl,
    save_summary_csv,
    score_pair,
    write_jsonl,
)

logger = logging.getLogger("eval.stt.multilingual")

# Scoped to the languages SAP Joule exposes (15 GA + 3 preview). The full
# 40-language registry lives in eval/stt/languages.py; language_acceptance.py
# (Phase 0) probes all 40 against the endpoint, while this test runs the full
# FLEURS WER/CER evaluation only on the Joule subset.
#
# Notes on corpus/product mismatches worth remembering when reading results:
#   * es -> FLEURS ships only es_419 (Latin-American Spanish); Joule targets
#     Spain Spanish. Close enough for a WER sanity check, not a localisation QA.
#   * ar -> FLEURS ar_eg is Egyptian Arabic; Joule/most ASR target MSA. Expect
#     a higher WER here than the model-card MSA number.
LANGUAGES = [
    {**lang, "tier": lang["joule"]} for lang in JOULE_LANGUAGES
]

# FLEURS test split is ~300-900 utterances/language. 200 gives a stable WER/CER
# at a reasonable per-language cost; raise for a final certification run.
PER_LANG_LIMIT = 200

# Expected-WER gates for space-delimited languages (the model claims ~4.6%
# multilingual WER on its own curated set; FLEURS read speech runs a little
# higher, so these bands leave headroom). CER languages get their own bands.
WER_THRESHOLDS = {"pass": 0.10, "good": 0.15, "acceptable": 0.25}
CER_THRESHOLDS = {"pass": 0.08, "good": 0.12, "acceptable": 0.20}


def _verdict(value: float | None, cer: bool) -> str:
    if value is None:
        return "N/A"
    th = CER_THRESHOLDS if cer else WER_THRESHOLDS
    if value <= th["pass"]:
        return "PASS"
    if value <= th["acceptable"]:
        return "FLAG"
    return "FAIL"


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "multilingual"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Optional allow-list from config (stt.multilingual_languages). Empty = all.
    only = set(getattr(config.stt, "multilingual_languages", []) or [])

    stt_client = STTClient(config)
    summary_rows: list[dict] = []

    for lang in LANGUAGES:
        if only and lang["key"] not in only:
            continue
        cer = is_cer_language(lang["key"])
        metric = "CER" if cer else "WER"
        logger.info(
            "=== Test 1.12: %s (FLEURS %s, %s, quality=%s) ===",
            lang["label"], lang["fleurs"], metric, lang["quality"],
        )

        jsonl_path = results_dir / f"{lang['key']}.jsonl"
        completed = get_completed_ids(jsonl_path)

        try:
            with fleurs_load(lang, limit=PER_LANG_LIMIT) as examples:
                pass
        except Exception as e:
            logger.error("Failed to load FLEURS %s: %s", lang["fleurs"], e)
            summary_rows.append({
                "language": lang["label"], "key": lang["key"], "fleurs": lang["fleurs"],
                "tier": lang["tier"], "quality": lang["quality"],
                "primary_metric": "cer" if cer else "wer", "status": "load_failed",
            })
            continue

        references: list[str] = []
        hypotheses: list[str] = []
        per_utt: list[float] = []
        n_err = 0

        for i, example in enumerate(tqdm(examples, desc=lang["key"])):
            item_id = f"{lang['key']}_{i}"
            if item_id in completed:
                continue
            try:
                audio_array, sr, ref_text = get_audio_and_ref(example)
                if not ref_text.strip():
                    continue
                audio_bytes = pcm16_bytes(audio_array)

                result = stt_client.recognize_batch(
                    audio_bytes, sr, language_code=lang["model_lang"],
                )
                hyp = result["transcript"]

                s = score_pair(ref_text, hyp, lang["key"])
                write_jsonl(jsonl_path, {
                    "id": item_id,
                    "language": lang["key"],
                    "reference": ref_text,
                    "hypothesis": hyp,
                    "wer": s["wer"],
                    "cer": s["cer"],
                    "primary": s["primary"],
                    "elapsed_s": result["elapsed_s"],
                })
                references.append(ref_text)
                hypotheses.append(hyp)
                if s["primary_value"] is not None:
                    per_utt.append(s["primary_value"])
            except Exception as e:
                n_err += 1
                if n_err <= 3:
                    logger.warning("Error on %s item %d: %s", lang["key"], i, e)

        # Resume path: everything was already done on a prior run.
        if not references and jsonl_path.exists():
            for rec in read_jsonl(jsonl_path):
                if rec.get("reference") and rec.get("hypothesis") is not None:
                    references.append(rec["reference"])
                    hypotheses.append(rec["hypothesis"])
                    pv = rec.get("cer") if cer else rec.get("wer")
                    if pv is not None:
                        per_utt.append(pv)

        if not references:
            logger.warning(
                "No usable output for %s (errors=%d) — endpoint may not accept "
                "this language.", lang["label"], n_err,
            )
            summary_rows.append({
                "language": lang["label"], "key": lang["key"], "fleurs": lang["fleurs"],
                "tier": lang["tier"], "quality": lang["quality"],
                "primary_metric": "cer" if cer else "wer",
                "status": "no_output", "n_errors": n_err,
            })
            continue

        agg = score_pair(references, hypotheses, lang["key"])
        primary_value = agg["primary_value"]

        if per_utt:
            _, ci_lo, ci_hi = bootstrap_ci(per_utt, n_bootstrap=config.evaluation.bootstrap_n)
        else:
            ci_lo = ci_hi = None

        row = {
            "language": lang["label"],
            "key": lang["key"],
            "fleurs": lang["fleurs"],
            "tier": lang["tier"],
            "quality": lang["quality"],
            "n_utterances": agg["n"],
            "primary_metric": agg["primary"],
            "wer": None if agg["wer"] is None else round(agg["wer"], 4),
            "cer": None if agg["cer"] is None else round(agg["cer"], 4),
            "primary_value": None if primary_value is None else round(primary_value, 4),
            "ci_lo": None if ci_lo is None else round(ci_lo, 4),
            "ci_hi": None if ci_hi is None else round(ci_hi, 4),
            "verdict": _verdict(primary_value, cer),
            "status": "ok",
        }
        summary_rows.append(row)
        logger.info(
            "%s: %s=%.2f%% (n=%d) → %s",
            lang["label"], agg["primary"].upper(),
            (primary_value or 0) * 100, agg["n"], row["verdict"],
        )

    # Macro-averages (unweighted across languages) so high-resource languages
    # don't dominate. WER macro over WER-primary langs; CER macro over CER langs.
    wer_vals = [r["wer"] for r in summary_rows
                if r.get("status") == "ok" and r.get("primary_metric") == "wer" and r.get("wer") is not None]
    cer_vals = [r["cer"] for r in summary_rows
                if r.get("status") == "ok" and r.get("primary_metric") == "cer" and r.get("cer") is not None]
    macro = {
        "language": "MACRO-AVG",
        "status": "summary",
        "n_languages_wer": len(wer_vals),
        "macro_wer": round(float(np.mean(wer_vals)), 4) if wer_vals else None,
        "n_languages_cer": len(cer_vals),
        "macro_cer": round(float(np.mean(cer_vals)), 4) if cer_vals else None,
    }
    summary_rows.append(macro)

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    logger.info(
        "Multilingual: macro WER=%s over %d langs; macro CER=%s over %d langs",
        macro["macro_wer"], macro["n_languages_wer"],
        macro["macro_cer"], macro["n_languages_cer"],
    )
    return {"test": "1.12", "name": "multilingual_accuracy", "results": summary_rows}
