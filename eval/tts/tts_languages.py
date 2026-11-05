"""Supported-language registry for the multilingual TTS tests.

The TTS model under test — **Magpie TTS Multilingual** — serves **12 languages**
on this SAP AI Core deployment, CONFIRMED live via one ``GetRivaSynthesisConfig``
call (2026-11-05): the deployment advertises
``en-US,es-US,fr-FR,de-DE,zh-CN,vi-VN,it-IT,hi-IN,ja-JP,ko-KR,ar-XA,pt-BR`` with
139 subvoices, and a short synthesis succeeded for 11/12 immediately (the 12th,
ar-XA, only rejected *Latin* probe text — its G2P front-end needs Arabic script;
it synthesises fine from FLEURS Arabic). This replaces the earlier (wrong)
"3 languages" assumption and the fabricated ``*.Female.Female-1`` voice strings,
which did not exist on the server and caused every non-English call to fail.

Each language has its OWN voice; driving non-English synthesis means setting BOTH
a per-language ``voice_name`` and ``language_code`` on the request. The actual
voice-name format the server accepts is ``Magpie-Multilingual.<LOCALE>.<Name>``
(e.g. ``Magpie-Multilingual.FR-FR.Aria``) — the known-good English config, not a
``...Female.Female-1`` string. The ``subvoices`` list reports each as
``LOCALE.Name[.Emotion]:index``; strip the ``:index`` and prefix the model's
``voice_name`` (``Magpie-Multilingual``) to build a request voice.

Voice selection below uses a stable persona chain (Aria → Isabela → Sofia) over
the voices the server actually lists per locale, so the same persona is reused
across languages wherever it exists. Override per deployment via
``tts.voice_overrides`` in config.yaml (map a language key to an exact voice
string). Chinese and Japanese are scored by CER (no word boundaries); the rest —
Korean included, which is space-delimited at the eojeol level — by WER. See
eval/utils.py, which already routes by language.

Re-probe any time with::

    .venv-probe/bin/python scripts/probe_tts_voices.py --synth
"""
from __future__ import annotations

# key:           short language key (matches eval/stt/languages.py keys where they overlap).
# language_code: BCP-47 locale sent to the endpoint (required — selects the language).
# voice_name:    Magpie Multilingual voice for that language, VERIFIED served on this
#                deployment (format Magpie-Multilingual.<LOCALE>.<Name>).
# fleurs:        google/fleurs config name — source of the synthesis TEXT (and ASR scoring).
# label:         human-readable name for reports.
TTS_LANGUAGES: list[dict] = [
    {"key": "en", "language_code": "en-US", "voice_name": "Magpie-Multilingual.EN-US.Aria",    "fleurs": "en_us",       "label": "English (US)"},
    {"key": "es", "language_code": "es-US", "voice_name": "Magpie-Multilingual.ES-US.Aria",    "fleurs": "es_419",      "label": "Spanish (LatAm)"},
    {"key": "fr", "language_code": "fr-FR", "voice_name": "Magpie-Multilingual.FR-FR.Aria",    "fleurs": "fr_fr",       "label": "French"},
    {"key": "de", "language_code": "de-DE", "voice_name": "Magpie-Multilingual.DE-DE.Aria",    "fleurs": "de_de",       "label": "German"},
    {"key": "it", "language_code": "it-IT", "voice_name": "Magpie-Multilingual.IT-IT.Isabela", "fleurs": "it_it",       "label": "Italian"},
    {"key": "pt", "language_code": "pt-BR", "voice_name": "Magpie-Multilingual.PT-BR.Isabela", "fleurs": "pt_br",       "label": "Portuguese (BR)"},
    {"key": "hi", "language_code": "hi-IN", "voice_name": "Magpie-Multilingual.HI-IN.Aria",    "fleurs": "hi_in",       "label": "Hindi"},
    {"key": "ko", "language_code": "ko-KR", "voice_name": "Magpie-Multilingual.KO-KR.Aria",    "fleurs": "ko_kr",       "label": "Korean"},
    {"key": "vi", "language_code": "vi-VN", "voice_name": "Magpie-Multilingual.VI-VN.Isabela", "fleurs": "vi_vn",       "label": "Vietnamese"},
    {"key": "zh", "language_code": "zh-CN", "voice_name": "Magpie-Multilingual.ZH-CN.Isabela", "fleurs": "cmn_hans_cn", "label": "Chinese (Mandarin)"},
    {"key": "ja", "language_code": "ja-JP", "voice_name": "Magpie-Multilingual.JA-JP.Isabela", "fleurs": "ja_jp",       "label": "Japanese"},
    {"key": "ar", "language_code": "ar-XA", "voice_name": "Magpie-Multilingual.AR-XA.Sofia",   "fleurs": "ar_eg",       "label": "Arabic"},
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
