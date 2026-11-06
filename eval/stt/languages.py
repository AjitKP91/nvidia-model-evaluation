"""Single source of truth for the languages the deployed STT model —
nvidia/nemotron-3.5-asr-streaming-0.6b — is evaluated on, mapped to FLEURS
configs.

Model card: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
The card lists 40 language-locales in three tiers:
  * transcription-ready — highest-accuracy ASR, ready out of the box.
  * broad-coverage      — production ASR across additional locales.
  * adaptation-ready    — recognised by the tokenizer; need in-domain
                          fine-tuning to unlock full transcription.
Each row records its card tier (`quality`) and whether the language is in SAP
Joule's product scope: "ga" (generally available), "preview", or None (the
model supports it but Joule does not currently expose it). The model
auto-detects language (`target_lang=auto`), so the per-request `model_lang`
tag is a hint only.

This list tracks only languages the card actually supports. Languages absent
from the card (Indonesian, Bengali, Marathi, Tagalog, Tamil, Telugu) and the
fine-tune-only Thai have been removed — they were probed in earlier runs under
a mistaken assumption (the 30B "Transcribe" card) and are not claimed by the
streaming model actually deployed.

Consumers:
  * language_acceptance.py — probes ALL of these against the live endpoint.
  * multilingual.py        — runs the full FLEURS WER/CER eval on the Joule
                             subset (JOULE_LANGUAGES).

FLEURS (google/fleurs, CC-BY-4.0) covers every one of these in one consistent
read-speech corpus. Scoring metric (WER vs CER) is derived per language via
eval.utils.is_cer_language(), so CJK is handled automatically.
"""
from __future__ import annotations

# key: ISO-639 root used for metric routing (is_cer_language) and file names.
# fleurs: google/fleurs config name.
# model_lang: BCP-47 hint sent to the endpoint (model auto-detects regardless).
# quality: card tier — "transcription-ready" | "broad-coverage" | "adaptation-ready".
# joule: "ga" | "preview" | None  (SAP Joule product scope).
# label: human-readable name.
NEMOTRON_LANGUAGES: list[dict] = [
    # --- SAP Joule GA (13) ---
    {"key": "en", "fleurs": "en_us",       "model_lang": "en-US", "quality": "transcription-ready", "joule": "ga",      "label": "English (US)"},
    {"key": "de", "fleurs": "de_de",       "model_lang": "de-DE", "quality": "transcription-ready", "joule": "ga",      "label": "German"},
    {"key": "fr", "fleurs": "fr_fr",       "model_lang": "fr-FR", "quality": "transcription-ready", "joule": "ga",      "label": "French"},
    {"key": "es", "fleurs": "es_419",      "model_lang": "es-ES", "quality": "transcription-ready", "joule": "ga",      "label": "Spanish"},
    {"key": "pt", "fleurs": "pt_br",       "model_lang": "pt-BR", "quality": "transcription-ready", "joule": "ga",      "label": "Portuguese (BR)"},
    {"key": "nl", "fleurs": "nl_nl",       "model_lang": "nl-NL", "quality": "transcription-ready", "joule": "ga",      "label": "Dutch"},
    {"key": "pl", "fleurs": "pl_pl",       "model_lang": "pl-PL", "quality": "broad-coverage",      "joule": "ga",      "label": "Polish"},
    {"key": "ko", "fleurs": "ko_kr",       "model_lang": "ko-KR", "quality": "transcription-ready", "joule": "ga",      "label": "Korean"},
    {"key": "el", "fleurs": "el_gr",       "model_lang": "el-GR", "quality": "adaptation-ready",    "joule": "ga",      "label": "Greek"},
    {"key": "ja", "fleurs": "ja_jp",       "model_lang": "ja-JP", "quality": "transcription-ready", "joule": "ga",      "label": "Japanese"},
    {"key": "zh", "fleurs": "cmn_hans_cn", "model_lang": "zh-CN", "quality": "broad-coverage",      "joule": "ga",      "label": "Chinese (Mandarin)"},
    {"key": "vi", "fleurs": "vi_vn",       "model_lang": "vi-VN", "quality": "transcription-ready", "joule": "ga",      "label": "Vietnamese"},
    {"key": "ar", "fleurs": "ar_eg",       "model_lang": "ar-AR", "quality": "transcription-ready", "joule": "ga",      "label": "Arabic"},
    # --- SAP Joule preview (3) ---
    {"key": "cs", "fleurs": "cs_cz",       "model_lang": "cs-CZ", "quality": "broad-coverage",      "joule": "preview", "label": "Czech"},
    {"key": "it", "fleurs": "it_it",       "model_lang": "it-IT", "quality": "transcription-ready", "joule": "preview", "label": "Italian"},
    {"key": "tr", "fleurs": "tr_tr",       "model_lang": "tr-TR", "quality": "transcription-ready", "joule": "preview", "label": "Turkish"},
    # --- Model-supported, not in Joule scope (18) ---
    {"key": "hi", "fleurs": "hi_in",       "model_lang": "hi-IN", "quality": "transcription-ready", "joule": None,      "label": "Hindi"},
    {"key": "ru", "fleurs": "ru_ru",       "model_lang": "ru-RU", "quality": "transcription-ready", "joule": None,      "label": "Russian"},
    {"key": "uk", "fleurs": "uk_ua",       "model_lang": "uk-UA", "quality": "transcription-ready", "joule": None,      "label": "Ukrainian"},
    {"key": "sk", "fleurs": "sk_sk",       "model_lang": "sk-SK", "quality": "broad-coverage",      "joule": None,      "label": "Slovak"},
    {"key": "bg", "fleurs": "bg_bg",       "model_lang": "bg-BG", "quality": "broad-coverage",      "joule": None,      "label": "Bulgarian"},
    {"key": "hr", "fleurs": "hr_hr",       "model_lang": "hr-HR", "quality": "broad-coverage",      "joule": None,      "label": "Croatian"},
    {"key": "da", "fleurs": "da_dk",       "model_lang": "da-DK", "quality": "broad-coverage",      "joule": None,      "label": "Danish"},
    {"key": "et", "fleurs": "et_ee",       "model_lang": "et-EE", "quality": "broad-coverage",      "joule": None,      "label": "Estonian"},
    {"key": "fi", "fleurs": "fi_fi",       "model_lang": "fi-FI", "quality": "broad-coverage",      "joule": None,      "label": "Finnish"},
    {"key": "hu", "fleurs": "hu_hu",       "model_lang": "hu-HU", "quality": "broad-coverage",      "joule": None,      "label": "Hungarian"},
    {"key": "ro", "fleurs": "ro_ro",       "model_lang": "ro-RO", "quality": "broad-coverage",      "joule": None,      "label": "Romanian"},
    {"key": "sv", "fleurs": "sv_se",       "model_lang": "sv-SE", "quality": "broad-coverage",      "joule": None,      "label": "Swedish"},
    {"key": "nb", "fleurs": "nb_no",       "model_lang": "nb-NO", "quality": "broad-coverage",      "joule": None,      "label": "Norwegian (Bokmål)"},
    {"key": "lv", "fleurs": "lv_lv",       "model_lang": "lv-LV", "quality": "adaptation-ready",    "joule": None,      "label": "Latvian"},
    {"key": "lt", "fleurs": "lt_lt",       "model_lang": "lt-LT", "quality": "adaptation-ready",    "joule": None,      "label": "Lithuanian"},
    {"key": "sl", "fleurs": "sl_si",       "model_lang": "sl-SI", "quality": "adaptation-ready",    "joule": None,      "label": "Slovenian"},
    {"key": "he", "fleurs": "he_il",       "model_lang": "he-IL", "quality": "adaptation-ready",    "joule": None,      "label": "Hebrew"},
    {"key": "mt", "fleurs": "mt_mt",       "model_lang": "mt-MT", "quality": "adaptation-ready",    "joule": None,      "label": "Maltese"},
]

# The subset SAP Joule exposes as a product (13 GA + 3 preview = 16).
JOULE_LANGUAGES: list[dict] = [l for l in NEMOTRON_LANGUAGES if l["joule"] is not None]

BY_KEY: dict[str, dict] = {l["key"]: l for l in NEMOTRON_LANGUAGES}
