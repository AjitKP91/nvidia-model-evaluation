"""Supported-language registry for the multilingual TTS tests.

Unlike the STT model (NVIDIA-Nemotron-3.5-Transcribe, 40 languages, auto-detect),
the TTS model under test — **Magpie TTS Multilingual** — supports only THREE
languages per NVIDIA's documentation, and each language has its OWN voice name:

    English (en-US), French (fr-FR), Spanish (es-US).

So driving non-English synthesis means setting BOTH a per-language ``voice_name``
and ``language_code`` on the request — not just a language hint. All three are
Latin-script, space-delimited languages, so intelligibility is scored with WER
throughout (no CER path is needed, unlike STT).

Whether the specific SAP AI Core deployment actually exposes the fr-FR / es-US
voices is unknown until probed — Test 2.9 (tts_language_acceptance) discovers
which languages this endpoint really synthesises, and records a graceful
status for the ones it does not, rather than a fabricated score.

The ``voice_name`` values below are the documented Magpie voices. They can be
overridden per deployment via ``tts.voice_overrides`` in config.yaml (map a
language key to the exact voice string the endpoint serves).
"""
from __future__ import annotations

# key:           short language key (matches eval/stt/languages.py keys where they overlap).
# language_code: BCP-47 locale sent to the endpoint (required — selects the language).
# voice_name:    documented Magpie Multilingual voice for that language.
# fleurs:        google/fleurs config name — source of the synthesis TEXT (and ASR scoring).
# label:         human-readable name for reports.
TTS_LANGUAGES: list[dict] = [
    {"key": "en", "language_code": "en-US", "voice_name": "Magpie-Multilingual.EN-US.Aria",               "fleurs": "en_us", "label": "English (US)"},
    {"key": "fr", "language_code": "fr-FR", "voice_name": "Magpie-Multilingual.FR-FR.Female.Female-1",     "fleurs": "fr_fr", "label": "French"},
    {"key": "es", "language_code": "es-US", "voice_name": "Magpie-Multilingual.ES-US.Female.Female-1",     "fleurs": "es_419", "label": "Spanish (LatAm)"},
]

BY_KEY: dict[str, dict] = {lang["key"]: lang for lang in TTS_LANGUAGES}


def resolve_tts_languages(keys: list[str] | None, voice_overrides: dict | None = None) -> list[dict]:
    """Map config language keys to TTS language dicts, applying any voice overrides.

    Unknown keys are skipped with no error (callers treat an empty result as
    English-only). ``voice_overrides`` maps a language key to the exact voice
    string this deployment serves, replacing the documented default.
    """
    import logging

    log = logging.getLogger("eval.tts.tts_languages")
    overrides = voice_overrides or {}
    out: list[dict] = []
    for key in keys or []:
        lang = BY_KEY.get(key)
        if lang is None:
            log.warning("Unknown TTS language key %r (supported: %s); skipping", key, list(BY_KEY))
            continue
        lang = dict(lang)  # copy so overrides don't mutate the registry
        if key in overrides:
            lang["voice_name"] = overrides[key]
        out.append(lang)
    return out
