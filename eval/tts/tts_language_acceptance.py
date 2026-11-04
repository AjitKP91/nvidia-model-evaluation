"""Test 2.9 — TTS Language Acceptance probe.

The TTS analogue of STT 1.13. Magpie TTS Multilingual advertises three
languages (en-US, fr-FR, es-US), each with its own voice name — but whether
*this* SAP AI Core deployment actually exposes the fr/es voices is unknown
until probed. Before scoring per-language naturalness/intelligibility/prosody,
confirm the endpoint really synthesises each configured language.

For each language we synthesise 2–3 FLEURS sentences with that language's
voice+locale, transcribe the output with Whisper (auto-detect), and record:

  * "ok"             - audio produced AND Whisper detected the target language
  * "wrong_language" - audio produced but Whisper detected a different language
                       (endpoint ignored the locale / fell back to English)
  * "empty"          - the call returned no audio
  * "error"          - the endpoint raised (e.g. rejected the voice name)
  * "load_failed"    - FLEURS text could not be loaded for this language

Round-trip WER (synth text vs Whisper transcript, language-aware via
``score_pair``) is kept per language as a sanity signal. Idempotent: languages
already "ok" are skipped on re-run. This is the honest gate — per-language
2.1/2.2/2.3 only score languages that pass here. If the deployment turns out to
be English-only, that is itself the finding (a serving gap, like STT th/id/ar),
reported rather than papered over.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

from eval.config import Config
from eval.stt.fleurs_common import fleurs_load, get_audio_and_ref
from eval.tts.client import TTSClient
from eval.tts.tts_languages import TTS_LANGUAGES, resolve_tts_languages
from eval.utils import read_jsonl, save_summary_csv, score_pair, write_jsonl

logger = logging.getLogger("eval.tts.tts_language_acceptance")

N_PROBE = 3  # sentences per language


def _load_whisper():
    import whisper
    logger.info("Loading Whisper large-v3 for round-trip transcription...")
    return whisper.load_model("large-v3")


def _release_whisper(model) -> None:
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    cache = Path.home() / ".cache" / "whisper"
    for f in sorted(cache.glob("large-v3*")):
        try:
            f.unlink()
        except Exception:
            pass


def probe(
    tts_client: TTSClient,
    whisper_model,
    out_dir: Path,
    languages: list[dict],
    n_clips: int = N_PROBE,
    resume: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "calls.jsonl"

    done_ok: set[str] = set()
    prior: dict[str, dict] = {}
    if resume:
        for rec in read_jsonl(jsonl_path):
            prior[rec["key"]] = rec
            if rec.get("status") == "ok":
                done_ok.add(rec["key"])

    rows: list[dict] = []
    for lang in languages:
        key = lang["key"]
        if key in done_ok:
            rows.append(prior[key])
            continue

        logger.info("Probe %-14s (voice %s, locale %s)", lang["label"], lang["voice_name"], lang["language_code"])

        try:
            with fleurs_load(lang, limit=n_clips * 2) as examples:
                texts = [get_audio_and_ref(ex, field="raw_transcription")[2] for ex in examples]
        except Exception as e:
            logger.warning("  load_failed %s: %s", key, e)
            row = {"key": key, "label": lang["label"], "voice": lang["voice_name"],
                   "language_code": lang["language_code"], "status": "load_failed", "detail": str(e)[:200]}
            write_jsonl(jsonl_path, row)
            rows.append(row)
            continue

        texts = [t for t in texts if t.strip()][:n_clips]
        n_audio = 0
        n_err = 0
        last_err = ""
        detected_langs: list[str] = []
        refs: list[str] = []
        hyps: list[str] = []
        sample_ref = sample_hyp = ""

        for i, text in enumerate(texts):
            syn_path = out_dir / f"probe_{key}_{i}.wav"
            try:
                tts_client.save_synthesis(
                    text, str(syn_path),
                    language_code=lang["language_code"], voice_name=lang["voice_name"],
                )
                if not syn_path.exists() or syn_path.stat().st_size == 0:
                    continue
                n_audio += 1
                result = whisper_model.transcribe(str(syn_path))
                hyp = result.get("text", "")
                detected = result.get("language", "")
                if detected:
                    detected_langs.append(detected)
                refs.append(text)
                hyps.append(hyp)
                if not sample_hyp and hyp.strip():
                    sample_ref, sample_hyp = text, hyp
            except Exception as e:
                n_err += 1
                last_err = str(e)[:200]
            finally:
                syn_path.unlink(missing_ok=True)

        # Majority detected language.
        detected_lang = ""
        if detected_langs:
            detected_lang = max(set(detected_langs), key=detected_langs.count)
        matches_target = detected_lang == key

        if n_audio == 0:
            status = "error" if n_err > 0 else "empty"
        elif not matches_target:
            status = "wrong_language"
        else:
            status = "ok"

        s = score_pair(refs, hyps, key) if refs else {"primary_value": None}
        row = {
            "key": key,
            "label": lang["label"],
            "voice": lang["voice_name"],
            "language_code": lang["language_code"],
            "status": status,
            "clips": len(texts),
            "n_audio": n_audio,
            "n_errors": n_err,
            "target_language": key,
            "detected_language": detected_lang,
            "matches_target": matches_target,
            "round_trip_wer": (
                round(s["primary_value"], 4) if s.get("primary_value") is not None else None
            ),
            "sample_ref": sample_ref[:120],
            "sample_hyp": sample_hyp[:120],
            "detail": last_err if status in ("error", "empty") else "",
        }
        write_jsonl(jsonl_path, row)
        rows.append(row)
        logger.info("  -> %s (%d/%d audio, detected=%s, WER=%s)",
                    status, n_audio, len(texts), detected_lang or "n/a",
                    row["round_trip_wer"] if row["round_trip_wer"] is not None else "n/a")

    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    summary = {
        "total": len(rows),
        "by_status": by_status,
        "accepted": [r["key"] for r in rows if r["status"] == "ok"],
    }
    save_summary_csv(out_dir / "summary.csv", rows)
    logger.info("TTS language acceptance: %s | accepted: %s", by_status, summary["accepted"])
    return {"rows": rows, "summary": summary}


def run(config: Config) -> dict:
    out_dir = Path(config.evaluation.results_dir) / "tts" / "language_acceptance"

    keys = getattr(config.tts, "multilingual_languages", []) or [lang["key"] for lang in TTS_LANGUAGES]
    languages = resolve_tts_languages(keys, getattr(config.tts, "voice_overrides", {}))
    if not languages:
        languages = list(TTS_LANGUAGES)

    tts_client = TTSClient(config)
    whisper_model = _load_whisper()
    try:
        result = probe(tts_client, whisper_model, out_dir, languages)
    finally:
        _release_whisper(whisper_model)

    return {"test": "2.9", "name": "tts_language_acceptance", **result}
