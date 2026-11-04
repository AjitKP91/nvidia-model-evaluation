"""Single source of truth for the languages NVIDIA-Nemotron-3.5-Transcribe
supports, mapped to FLEURS configs for evaluation.

The model card lists 40 transcription languages in three quality tiers
(leading / production / supported) and auto-detects language (the per-request
`model_lang` tag is a hint only). Each row also records whether the language is
part of SAP Joule's product scope: "ga" (generally available), "preview", or
None (model supports it but Joule does not currently expose it).

Consumers:
  * language_acceptance.py — probes ALL 40 against the live endpoint.
  * multilingual.py        — runs the full FLEURS WER/CER eval on the Joule
                             subset (JOULE_LANGUAGES).

FLEURS (google/fleurs, CC-BY-4.0) covers every one of these in one consistent
read-speech corpus. Scoring metric (WER vs CER) is derived per language via
eval.utils.is_cer_language(), so CJK/Thai are handled automatically.
"""
from __future__ import annotations

# key: ISO-639 root used for metric routing (is_cer_language) and file names.
# fleurs: google/fleurs config name.
# model_lang: BCP-47 hint sent to the endpoint (model auto-detects regardless).
# quality: Nemotron-3.5 model-card tier.
# joule: "ga" | "preview" | None  (SAP Joule product scope).
# label: human-readable name.
NEMOTRON_LANGUAGES: list[dict] = [
    # --- SAP Joule GA (15) ---
    {"key": "en", "fleurs": "en_us",       "model_lang": "en-US", "quality": "leading",    "joule": "ga",      "label": "English (US)"},
    {"key": "de", "fleurs": "de_de",       "model_lang": "de-DE", "quality": "leading",    "joule": "ga",      "label": "German"},
    {"key": "fr", "fleurs": "fr_fr",       "model_lang": "fr-FR", "quality": "leading",    "joule": "ga",      "label": "French"},
    {"key": "es", "fleurs": "es_419",      "model_lang": "es-ES", "quality": "leading",    "joule": "ga",      "label": "Spanish"},
    {"key": "pt", "fleurs": "pt_br",       "model_lang": "pt-BR", "quality": "leading",    "joule": "ga",      "label": "Portuguese (BR)"},
    {"key": "nl", "fleurs": "nl_nl",       "model_lang": "nl-NL", "quality": "leading",    "joule": "ga",      "label": "Dutch"},
    {"key": "pl", "fleurs": "pl_pl",       "model_lang": "pl-PL", "quality": "leading",    "joule": "ga",      "label": "Polish"},
    {"key": "ko", "fleurs": "ko_kr",       "model_lang": "ko-KR", "quality": "leading",    "joule": "ga",      "label": "Korean"},
    {"key": "el", "fleurs": "el_gr",       "model_lang": "el-GR", "quality": "supported",  "joule": "ga",      "label": "Greek"},
    {"key": "ja", "fleurs": "ja_jp",       "model_lang": "ja-JP", "quality": "production", "joule": "ga",      "label": "Japanese"},
    {"key": "zh", "fleurs": "cmn_hans_cn", "model_lang": "zh-CN", "quality": "supported",  "joule": "ga",      "label": "Chinese (Mandarin)"},
    {"key": "vi", "fleurs": "vi_vn",       "model_lang": "vi-VN", "quality": "supported",  "joule": "ga",      "label": "Vietnamese"},
    {"key": "th", "fleurs": "th_th",       "model_lang": "th-TH", "quality": "production", "joule": "ga",      "label": "Thai"},
    {"key": "id", "fleurs": "id_id",       "model_lang": "id-ID", "quality": "supported",  "joule": "ga",      "label": "Indonesian"},
    {"key": "ar", "fleurs": "ar_eg",       "model_lang": "ar-SA", "quality": "supported",  "joule": "ga",      "label": "Arabic"},
    # --- SAP Joule preview (3) ---
    {"key": "cs", "fleurs": "cs_cz",       "model_lang": "cs-CZ", "quality": "leading",    "joule": "preview", "label": "Czech"},
    {"key": "it", "fleurs": "it_it",       "model_lang": "it-IT", "quality": "leading",    "joule": "preview", "label": "Italian"},
    {"key": "tr", "fleurs": "tr_tr",       "model_lang": "tr-TR", "quality": "production", "joule": "preview", "label": "Turkish"},
    # --- Model-supported, not in Joule scope (22) ---
    {"key": "hi", "fleurs": "hi_in",       "model_lang": "hi-IN", "quality": "leading",    "joule": None,      "label": "Hindi"},
    {"key": "ru", "fleurs": "ru_ru",       "model_lang": "ru-RU", "quality": "leading",    "joule": None,      "label": "Russian"},
    {"key": "sk", "fleurs": "sk_sk",       "model_lang": "sk-SK", "quality": "leading",    "joule": None,      "label": "Slovak"},
    {"key": "bg", "fleurs": "bg_bg",       "model_lang": "bg-BG", "quality": "production", "joule": None,      "label": "Bulgarian"},
    {"key": "hr", "fleurs": "hr_hr",       "model_lang": "hr-HR", "quality": "production", "joule": None,      "label": "Croatian"},
    {"key": "da", "fleurs": "da_dk",       "model_lang": "da-DK", "quality": "production", "joule": None,      "label": "Danish"},
    {"key": "et", "fleurs": "et_ee",       "model_lang": "et-EE", "quality": "production", "joule": None,      "label": "Estonian"},
    {"key": "fi", "fleurs": "fi_fi",       "model_lang": "fi-FI", "quality": "production", "joule": None,      "label": "Finnish"},
    {"key": "hu", "fleurs": "hu_hu",       "model_lang": "hu-HU", "quality": "production", "joule": None,      "label": "Hungarian"},
    {"key": "lv", "fleurs": "lv_lv",       "model_lang": "lv-LV", "quality": "production", "joule": None,      "label": "Latvian"},
    {"key": "lt", "fleurs": "lt_lt",       "model_lang": "lt-LT", "quality": "production", "joule": None,      "label": "Lithuanian"},
    {"key": "ro", "fleurs": "ro_ro",       "model_lang": "ro-RO", "quality": "production", "joule": None,      "label": "Romanian"},
    {"key": "sl", "fleurs": "sl_si",       "model_lang": "sl-SI", "quality": "production", "joule": None,      "label": "Slovenian"},
    {"key": "sv", "fleurs": "sv_se",       "model_lang": "sv-SE", "quality": "production", "joule": None,      "label": "Swedish"},
    {"key": "uk", "fleurs": "uk_ua",       "model_lang": "uk-UA", "quality": "production", "joule": None,      "label": "Ukrainian"},
    {"key": "bn", "fleurs": "bn_in",       "model_lang": "bn-IN", "quality": "supported",  "joule": None,      "label": "Bengali"},
    {"key": "he", "fleurs": "he_il",       "model_lang": "he-IL", "quality": "supported",  "joule": None,      "label": "Hebrew"},
    {"key": "mt", "fleurs": "mt_mt",       "model_lang": "mt-MT", "quality": "supported",  "joule": None,      "label": "Maltese"},
    {"key": "mr", "fleurs": "mr_in",       "model_lang": "mr-IN", "quality": "supported",  "joule": None,      "label": "Marathi"},
    {"key": "fil","fleurs": "fil_ph",      "model_lang": "fil-PH","quality": "supported",  "joule": None,      "label": "Tagalog (Filipino)"},
    {"key": "ta", "fleurs": "ta_in",       "model_lang": "ta-IN", "quality": "supported",  "joule": None,      "label": "Tamil"},
    {"key": "te", "fleurs": "te_in",       "model_lang": "te-IN", "quality": "supported",  "joule": None,      "label": "Telugu"},
]

# The subset SAP Joule exposes as a product (15 GA + 3 preview = 18).
JOULE_LANGUAGES: list[dict] = [l for l in NEMOTRON_LANGUAGES if l["joule"] is not None]

BY_KEY: dict[str, dict] = {l["key"]: l for l in NEMOTRON_LANGUAGES}
