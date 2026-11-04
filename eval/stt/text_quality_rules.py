"""Per-language text-formatting rules for the output-quality test (1.9).

Test 1.9 scores punctuation, capitalisation, and number formatting. Those
conventions are language-specific, so scoring English rules against Japanese or
Arabic output produces meaningless numbers. This table keys the rules by
language root (the part of the FLEURS/key code before any region suffix) so the
1.9 scorers can:

  * count the RIGHT punctuation marks (full-width 。、？！ for ja/zh, ؟، for ar,
    ¿¡ for es, ·; for el, on top of the Latin . , ? !);
  * return ``None`` (= N/A, never a false 0) for capitalisation on unicameral
    scripts (ja, zh, th, ar, ko) that have no case distinction;
  * skip number-formatting scoring when a script normally uses non-ASCII digits,
    rather than reporting a wrong count.

English rules reproduce the module's original behaviour exactly, so the default
English pass is unchanged.
"""
from __future__ import annotations

# Latin sentence punctuation — the original PUNCT_MARKS for 1.9.
_LATIN_PUNCT = [".", ",", "?", "!"]

# rules keyed by language root. Fields:
#   punct_marks: marks to score for this language
#   has_case:    True if the script distinguishes upper/lower case
#   ascii_digits: True if numbers are normally written with ASCII 0-9 (so the
#                 existing number-accuracy regex is meaningful)
LANG_RULES: dict[str, dict] = {
    # --- bicameral Latin/Cyrillic/Greek scripts (case + ASCII digits) ---
    "en": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "de": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "fr": {"punct_marks": _LATIN_PUNCT + ["?", "!", ";", ":"], "has_case": True, "ascii_digits": True},
    "nl": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "pl": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "pt": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "it": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "cs": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "tr": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "vi": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    "id": {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True},
    # Spanish adds inverted opening marks.
    "es": {"punct_marks": _LATIN_PUNCT + ["¿", "¡"], "has_case": True, "ascii_digits": True},
    # Greek: bicameral script; uses middle dot and Greek question mark (;).
    "el": {"punct_marks": [".", ",", "!", "·", ";"], "has_case": True, "ascii_digits": True},

    # --- unicameral scripts: capitalisation is N/A ---
    # CJK — full-width punctuation; CER-scored elsewhere.
    "ja": {"punct_marks": ["。", "、", "？", "！", "，"], "has_case": False, "ascii_digits": True},
    "zh": {"punct_marks": ["。", "、", "？", "！", "，"], "has_case": False, "ascii_digits": True},
    # Korean uses Latin punctuation but Hangul has no case.
    "ko": {"punct_marks": _LATIN_PUNCT, "has_case": False, "ascii_digits": True},
    # Thai: no case, no sentence-final punctuation convention, native digits common.
    "th": {"punct_marks": [], "has_case": False, "ascii_digits": False},
    # Arabic: RTL, no case; Arabic comma/question mark; native digits common.
    "ar": {"punct_marks": ["؟", "،", ".", "!"], "has_case": False, "ascii_digits": False},
}

# Default used for any language not explicitly listed: treat as Latin/bicameral.
_DEFAULT_RULES = {"punct_marks": _LATIN_PUNCT, "has_case": True, "ascii_digits": True}


def rules_for(lang_key: str) -> dict:
    """Return the formatting rules for a language key (e.g. "es", "ja", "en").

    The key's region suffix is ignored (``pt_br`` -> ``pt``). Unknown languages
    fall back to Latin/bicameral defaults.
    """
    root = (lang_key or "en").split("_")[0].split("-")[0].lower()
    return LANG_RULES.get(root, _DEFAULT_RULES)
