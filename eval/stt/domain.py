"""Test 1.8 — Domain-Specific Vocabulary."""
from __future__ import annotations

import logging
from pathlib import Path

import jiwer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import NORMALIZE_FOR_WER, save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.domain")


def _extract_entities(text: str) -> list[str]:
    """Extract named entities using spaCy."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


def _compute_krr(ref_entities: list[str], hypothesis: str) -> float:
    """Keyword Recall Rate: fraction of reference entities found in hypothesis."""
    if not ref_entities:
        return 1.0
    hyp_lower = hypothesis.lower()
    found = sum(1 for e in ref_entities if e.lower() in hyp_lower)
    return found / len(ref_entities)


def _compute_ne_wer(reference: str, hypothesis: str, entities: list[str]) -> float | None:
    """WER computed only on named-entity tokens."""
    if not entities:
        return None
    entity_tokens = set()
    for e in entities:
        entity_tokens.update(e.lower().split())

    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    ref_filtered = " ".join(w for w in ref_words if w in entity_tokens)
    hyp_filtered = " ".join(w for w in hyp_words if w in entity_tokens)

    if not ref_filtered:
        return None
    return jiwer.wer(ref_filtered, hyp_filtered)


def _compute_oov_rate(reference: str, vocab_path: str | None = None) -> float:
    """Fraction of reference words not in a standard vocabulary."""
    COMMON_VOCAB_SIZE = 200_000
    words = reference.lower().split()
    if not words:
        return 0.0
    # Simple heuristic: words with digits, special chars, or very rare patterns
    oov = sum(1 for w in words if any(c.isdigit() for c in w) or len(w) > 20)
    return oov / len(words)


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "domain"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.8: Domain-Specific Vocabulary ===")

    datasets_to_test = [
        {"name": "spgispeech", "hf": ("esb/datasets", "spgispeech"), "label": "SPGISpeech (financial)"},
        {"name": "earnings22", "hf": ("esb/datasets", "earnings22"), "label": "Earnings-22 (business)"},
    ]

    summary_rows = []

    for ds_info in datasets_to_test:
        logger.info("Processing %s", ds_info["label"])
        try:
            path, name = ds_info["hf"]
            kwargs = {"path": path, "split": "test", "trust_remote_code": True}
            if name:
                kwargs["name"] = name
            ds = load_dataset(**kwargs)
        except Exception as e:
            logger.error("Failed to load %s: %s", ds_info["name"], e)
            continue

        subset = list(ds.select(range(min(500, len(ds)))))
        refs, hyps = [], []
        krr_values = []
        ne_wer_values = []
        misrecognised_terms: dict[str, int] = {}

        for i, ex in enumerate(tqdm(subset, desc=ds_info["name"])):
            audio = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            ref = ex.get("norm_text") or ex.get("text", "")
            audio_bytes = (audio * 32768).astype(np.int16).tobytes()

            try:
                result = stt_client.recognize_batch(audio_bytes, sr)
                hyp = result["transcript"]
                refs.append(ref)
                hyps.append(hyp)

                entities = _extract_entities(ref)
                krr = _compute_krr(entities, hyp)
                ne_wer = _compute_ne_wer(ref, hyp, entities)
                krr_values.append(krr)
                if ne_wer is not None:
                    ne_wer_values.append(ne_wer)

                for ent in entities:
                    if ent.lower() not in hyp.lower():
                        misrecognised_terms[ent] = misrecognised_terms.get(ent, 0) + 1

                write_jsonl(jsonl_path, {
                    "id": f"{ds_info['name']}_{i}",
                    "dataset": ds_info["name"],
                    "reference": ref,
                    "hypothesis": hyp,
                    "entities": entities,
                    "krr": krr,
                    "ne_wer": ne_wer,
                })
            except Exception as e:
                logger.warning("Failed %s item %d: %s", ds_info["name"], i, e)

        if refs:
            domain_wer = jiwer.wer(refs, hyps, truth_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER)
            row = {
                "dataset": ds_info["label"],
                "n_utterances": len(refs),
                "domain_wer": round(domain_wer, 4),
                "krr_mean": round(np.mean(krr_values), 4) if krr_values else None,
                "ne_wer_mean": round(np.mean(ne_wer_values), 4) if ne_wer_values else None,
                "top_misrecognised": sorted(misrecognised_terms.items(), key=lambda x: -x[1])[:20],
            }
            summary_rows.append(row)
            logger.info("%s: WER=%.2f%% KRR=%.2f%% NE-WER=%.2f%%",
                ds_info["label"], domain_wer * 100,
                (np.mean(krr_values) * 100) if krr_values else 0,
                (np.mean(ne_wer_values) * 100) if ne_wer_values else 0,
            )

    save_summary_csv(results_dir / "summary.csv", summary_rows)
    return {"test": "1.8", "name": "domain_vocabulary", "results": summary_rows}
