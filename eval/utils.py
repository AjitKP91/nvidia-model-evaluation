from __future__ import annotations

import functools
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import soundfile as sf

logger = logging.getLogger("eval")

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

NORMALIZE_FOR_WER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

# Same as above but without RemoveEmptyStrings — use for aggregate (list) calls
# so that empty refs/hyps stay in the list and lengths never diverge.
NORMALIZE_FOR_WER_AGG = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("?", "")
        .replace("!", "")
        .strip()
    )


# ---------------------------------------------------------------------------
# Language-aware normalisation for WER / CER
# ---------------------------------------------------------------------------
# The HF Open ASR Leaderboard and every NVIDIA model card score transcription
# accuracy by applying ONE shared normaliser to both hypothesis and reference
# before computing WER. For English that normaliser is Whisper's
# EnglishTextNormalizer (digit<->word, currency, contractions, unit spelling);
# the English rules are wrong for other languages, so Whisper's
# BasicTextNormalizer is used instead (lowercase, strip punctuation/symbols,
# preserve CJK characters). Matching the industry normaliser is what makes our
# WER numbers directly comparable to published leaderboard / model-card scores.

# Scripts with no (or inconsistent) word boundaries: WER is not meaningful, so
# these are scored primarily by Character Error Rate (CER).
CER_LANGUAGES = {"ja", "zh", "cmn", "yue", "wuu", "th", "lo", "km", "my", "bo"}


def _lang_root(lang_code: str | None) -> str:
    """Reduce a locale tag to its base language.

    'de-DE' -> 'de', 'cmn_hans_cn' -> 'cmn', 'ja_jp' -> 'ja', '' -> 'en'.
    """
    if not lang_code:
        return "en"
    return lang_code.replace("_", "-").split("-")[0].lower()


@functools.lru_cache(maxsize=1)
def _english_normalizer():
    try:
        from whisper.normalizers import EnglishTextNormalizer
        return EnglishTextNormalizer()
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Whisper EnglishTextNormalizer unavailable (%s); falling back to "
            "jiwer-only normalisation. English WER on numbers/contractions will "
            "be inflated and NOT comparable to leaderboard numbers.",
            exc,
        )
        return None


@functools.lru_cache(maxsize=1)
def _basic_normalizer():
    try:
        from whisper.normalizers import BasicTextNormalizer
        # Keep diacritics: they are phonemic in French/Vietnamese/Arabic/etc.,
        # and stripping them would reward wrong output. split_letters=False so
        # word-based WER is preserved for space-delimited languages; CER handles
        # the character-level CJK/Thai case separately.
        return BasicTextNormalizer(remove_diacritics=False, split_letters=False)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning(
            "Whisper BasicTextNormalizer unavailable (%s); falling back to "
            "jiwer-only normalisation for multilingual scoring.",
            exc,
        )
        return None


def is_cer_language(lang_code: str | None) -> bool:
    """True when the language should be scored by CER rather than WER."""
    return _lang_root(lang_code) in CER_LANGUAGES


def _strip_all_ws(text: str) -> str:
    """Remove every whitespace character (spaces, tabs, full-width spaces).

    Used only for CER-primary (CJK/Thai) scoring, where word spacing carries no
    linguistic meaning and ASR spacing is an artifact rather than signal.
    """
    return re.sub(r"\s+", "", text)


def normalize_for_language(text: str | None, lang_code: str = "en") -> str:
    """Apply the correct Whisper normaliser for the language.

    English -> EnglishTextNormalizer, everything else -> BasicTextNormalizer.
    Falls back to jiwer-style lowercase/strip-punctuation when Whisper is not
    installed, so scoring never crashes (only loses number/contraction mapping).
    """
    if not text:
        return ""
    norm = _english_normalizer() if _lang_root(lang_code) == "en" else _basic_normalizer()
    if norm is None:
        return normalize_text(text)
    try:
        return norm(text)
    except Exception:  # pragma: no cover - defensive
        return normalize_text(text)


def score_pair(references, hypotheses, lang_code: str = "en") -> dict:
    """Aggregate WER/CER for one language after language-aware normalisation.

    Accepts a single (str, str) or equal-length lists. Returns a dict with
    ``wer`` (None for CER-primary languages, where word WER is meaningless),
    ``cer``, ``primary`` ('cer' for CJK/Thai else 'wer'), ``primary_value`` and
    ``n`` (number of scored pairs). Pairs whose reference is empty after
    normalisation are dropped so jiwer never divides by zero.
    """
    if isinstance(references, str):
        references = [references]
        hypotheses = [hypotheses]
    refs_n = [normalize_for_language(r, lang_code) for r in references]
    hyps_n = [normalize_for_language(h, lang_code) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs_n, hyps_n) if r.strip()]
    cer_primary = is_cer_language(lang_code)
    if not pairs:
        return {"wer": None, "cer": None, "primary": "cer" if cer_primary else "wer",
                "primary_value": None, "n": 0}
    refs_f = [p[0] for p in pairs]
    hyps_f = [p[1] for p in pairs]
    if cer_primary:
        # Word WER is nonsense without word boundaries — report CER only.
        # CJK/Thai have no inter-word spaces, and ASR engines are inconsistent
        # about inserting them (this endpoint emits one space PER CHARACTER,
        # e.g. "イ ン タ ー ネ ッ ト"). Counting those spaces as characters
        # inflates CER toward 1.0, so — matching Open ASR Leaderboard practice
        # for Chinese/Japanese — strip ALL whitespace before scoring CER.
        refs_c = [_strip_all_ws(r) for r in refs_f]
        hyps_c = [_strip_all_ws(h) for h in hyps_f]
        refs_c, hyps_c = zip(*[(r, h) for r, h in zip(refs_c, hyps_c) if r]) if any(refs_c) else ([], [])
        cer = float(jiwer.cer(list(refs_c), list(hyps_c))) if refs_c else None
        return {"wer": None, "cer": cer, "primary": "cer",
                "primary_value": cer, "n": len(refs_c)}
    cer = float(jiwer.cer(refs_f, hyps_f))
    wer = float(jiwer.wer(
        refs_f, hyps_f,
        reference_transform=NORMALIZE_FOR_WER_AGG,
        hypothesis_transform=NORMALIZE_FOR_WER_AGG,
    ))
    return {"wer": wer, "cer": cer, "primary": "wer",
            "primary_value": wer, "n": len(pairs)}


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    return audio, sr


def save_audio(path: str | Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr)


def audio_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return info.duration


# ---------------------------------------------------------------------------
# JSONL persistence (idempotent resume support)
# ---------------------------------------------------------------------------

def write_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    # Guard against a prior partial line (e.g. an interrupted or disk-full write)
    # merging with this record into one unparseable line: if the file doesn't end
    # in a newline, start a fresh line first. Cheap — one seek per append.
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, 2)  # 2 == os.SEEK_END
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a") as f:
        f.write(prefix + line + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file, tolerating malformed lines.

    A partial line left by an interrupted or disk-full write (the classic
    ``json.decoder.JSONDecodeError: Unterminated string`` on resume) is skipped
    with a warning instead of crashing the whole test. The affected item simply
    re-runs, since it was never recorded as completed.
    """
    path = Path(path)
    if not path.exists():
        return []
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        logger.warning(
            "read_jsonl: skipped %d malformed line(s) in %s (likely a partial "
            "write from an interrupted/disk-full run; those items will re-run).",
            skipped, path,
        )
    return records


def get_completed_ids(path: str | Path, id_field: str = "id") -> set:
    return {r[id_field] for r in read_jsonl(path) if id_field in r}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_percentiles(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def bootstrap_ci(
    values: list[float], n_bootstrap: int = 1000, ci: float = 0.95
) -> tuple[float, float, float]:
    arr = np.array(values)
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return float(np.mean(arr)), lower, upper


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_with_backoff(
    max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0,
    reraise_if=None,
):
    """Exponential-backoff retry decorator.

    reraise_if: optional callable(exc) -> bool; if it returns True the
    exception is re-raised immediately without retrying (permanent failures).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if reraise_if is not None and reraise_if(e):
                        raise
                    if attempt == max_retries:
                        raise
                    delay = min(base_delay * (2**attempt), max_delay)
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence verbose HTTP request logs from httpx/httpcore; keep datasets progress bars
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Result summary helpers
# ---------------------------------------------------------------------------

def save_summary_csv(path: str | Path, rows: list[dict]) -> None:
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Summary saved to %s", path)


# ---------------------------------------------------------------------------
# Dataset loader — download, load into memory, delete cache
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def load_dataset_tmp(path: str, split: str, name=None, limit: int | None = None,
                     streaming: bool = True, **kwargs):
    """Stream a HF dataset into an in-memory list WITHOUT caching it to disk.

    Uses ``streaming=True`` by default: HuggingFace fetches shards lazily over the
    network and nothing is written to the dataset cache. This keeps disk usage
    FLAT regardless of corpus size — essential because several script-based
    builders (``facebook/voxpopuli``, ``openslr/librispeech_asr``) otherwise
    download the ENTIRE config (all splits, 100+ GB) just to serve one test
    split, which repeatedly filled the VM disk and corrupted result files.

    Only the items actually read cross the network: with ``limit`` set exactly
    ``limit`` examples are pulled; without it the full split is streamed once into
    a list. Re-runs re-stream (cheap when the VM is in-region) rather than relying
    on a multi-hundred-GB on-disk cache, so there is nothing to pre-download and
    nothing to clean up.

    Pass ``streaming=False`` only for a caller that genuinely needs the full
    map-style dataset materialised to the HF cache (random access to the raw
    dataset object). STT accuracy/perf/etc. just iterate, so streaming is correct.

    Usage:
        with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean") as examples:
            for ex in examples: ...
    """
    import itertools
    import os

    from datasets import load_dataset

    kw: dict = {
        "split": split,
        "token": os.environ.get("HF_TOKEN") or True,
        "streaming": streaming,
        **kwargs,
    }
    if name is not None:
        kw["name"] = name
    ds = load_dataset(path, **kw)
    data = list(itertools.islice(ds, limit)) if limit is not None else list(ds)
    del ds
    yield data
