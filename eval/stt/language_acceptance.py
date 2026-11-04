"""Phase-0 language-acceptance probe.

Before spending a full multilingual evaluation, confirm the live endpoint
actually transcribes each of the model's 40 advertised languages. For every
language we stream a handful of FLEURS clips (no full-split download), send them
through the real client, and record an acceptance verdict:

  * "ok"          - at least one clip returned a non-empty transcript
  * "empty"       - calls succeeded but every transcript was blank
  * "error"       - the endpoint raised (e.g. rejected the language hint)
  * "load_failed" - FLEURS could not be streamed for this language

It also keeps one sample (reference, hypothesis) and a quick WER/CER per
language so the operator can eyeball that the output is in the right script
(e.g. Japanese came back as kana/kanji, Thai as Thai, not romanised). This is a
smoke test, NOT the accuracy benchmark — that is Test 1.12 (multilingual.py).

The probe is idempotent: languages already recorded "ok" are skipped on re-run,
so wiring it into Phase 0 costs almost nothing after the first pass.
"""
from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path

import numpy as np

from eval.config import Config
from eval.stt.client import STTClient
from eval.stt.languages import NEMOTRON_LANGUAGES
from eval.utils import (
    is_cer_language,
    read_jsonl,
    save_summary_csv,
    score_pair,
    write_jsonl,
)

logger = logging.getLogger("eval.stt.language_acceptance")

N_PROBE = 3  # clips per language — enough to tell "works" from "rejected".


def _stream_fleurs(fleurs_cfg: str, n: int) -> list[dict]:
    """Pull the first `n` FLEURS test clips for a language without downloading
    the whole split (streaming=True fetches only the shards it needs)."""
    from datasets import load_dataset

    ds = load_dataset(
        "google/fleurs", fleurs_cfg, split="test",
        streaming=True, trust_remote_code=True,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return list(itertools.islice(ds, n))


def probe(
    stt_client: STTClient,
    out_dir: Path,
    n_clips: int = N_PROBE,
    only: set[str] | None = None,
    resume: bool = True,
) -> dict:
    """Run the acceptance probe. Returns {'rows': [...], 'summary': {...}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "language_acceptance.jsonl"

    done_ok: set[str] = set()
    prior: dict[str, dict] = {}
    if resume:
        for rec in read_jsonl(jsonl_path):
            prior[rec["key"]] = rec
            if rec.get("status") == "ok":
                done_ok.add(rec["key"])

    rows: list[dict] = []
    for lang in NEMOTRON_LANGUAGES:
        key = lang["key"]
        if only and key not in only:
            continue
        if key in done_ok:
            rows.append(prior[key])  # already confirmed
            continue

        cer = is_cer_language(key)
        logger.info("Probe %-22s (FLEURS %s, hint %s)", lang["label"], lang["fleurs"], lang["model_lang"])

        try:
            clips = _stream_fleurs(lang["fleurs"], n_clips)
        except Exception as e:
            logger.warning("  load_failed %s: %s", key, e)
            row = {"key": key, "label": lang["label"], "joule": lang["joule"],
                   "quality": lang["quality"], "metric": "cer" if cer else "wer",
                   "status": "load_failed", "detail": str(e)[:200]}
            write_jsonl(jsonl_path, row)
            rows.append(row)
            continue

        n_ok = 0
        n_err = 0
        last_err = ""
        sample_ref = sample_hyp = ""
        primary_vals: list[float] = []
        for ex in clips:
            try:
                arr = np.array(ex["audio"]["array"], dtype=np.float32)
                sr = ex["audio"]["sampling_rate"]
                ref = ex.get("transcription") or ex.get("raw_transcription") or ""
                audio_bytes = (arr * 32768).astype(np.int16).tobytes()
                result = stt_client.recognize_batch(audio_bytes, sr, language_code=lang["model_lang"])
                hyp = result["transcript"]
                if hyp.strip():
                    n_ok += 1
                    if not sample_hyp:
                        sample_ref, sample_hyp = ref, hyp
                s = score_pair(ref, hyp, key)
                if s["primary_value"] is not None:
                    primary_vals.append(s["primary_value"])
            except Exception as e:
                n_err += 1
                last_err = str(e)[:200]

        if n_ok > 0:
            status = "ok"
        elif n_err > 0:
            status = "error"
        else:
            status = "empty"

        row = {
            "key": key,
            "label": lang["label"],
            "joule": lang["joule"],
            "quality": lang["quality"],
            "metric": "cer" if cer else "wer",
            "status": status,
            "clips": len(clips),
            "n_transcribed": n_ok,
            "n_errors": n_err,
            "sample_primary": round(float(np.mean(primary_vals)), 4) if primary_vals else None,
            "sample_ref": sample_ref[:120],
            "sample_hyp": sample_hyp[:120],
            "detail": last_err if status == "error" else "",
        }
        write_jsonl(jsonl_path, row)
        rows.append(row)
        logger.info("  -> %s (%d/%d transcribed, sample %s=%s)",
                    status, n_ok, len(clips), row["metric"],
                    row["sample_primary"] if row["sample_primary"] is not None else "n/a")

    # Summary counts
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    joule_rows = [r for r in rows if r.get("joule")]
    joule_ok = sum(1 for r in joule_rows if r["status"] == "ok")
    summary = {
        "total": len(rows),
        "by_status": by_status,
        "joule_total": len(joule_rows),
        "joule_ok": joule_ok,
    }

    save_summary_csv(out_dir / "language_acceptance.csv", rows)
    logger.info(
        "Language acceptance: %s | Joule languages transcribing: %d/%d",
        by_status, joule_ok, len(joule_rows),
    )
    return {"rows": rows, "summary": summary}


def run(config: Config) -> dict:
    """Standalone entry point (Test 1.13 / `--test language_acceptance`)."""
    out_dir = Path(config.evaluation.results_dir) / "stt" / "language_acceptance"
    only = set(getattr(config.stt, "multilingual_languages", []) or [])
    stt_client = STTClient(config)
    result = probe(stt_client, out_dir, only=only or None)
    return {"test": "1.13", "name": "language_acceptance", **result}
