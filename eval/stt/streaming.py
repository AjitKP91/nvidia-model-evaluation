"""Test 1.3 — Streaming Performance (TTFW / Partial Latency / Finalization / Stability)."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import (
    NORMALIZE_FOR_WER,
    bootstrap_ci,
    compute_percentiles,
    get_completed_ids,
    save_summary_csv,
    write_jsonl,
)

logger = logging.getLogger("eval.stt.streaming")


def _compute_stability_rate(partials: list[dict]) -> float:
    """Fraction of partial-result words that are not revised in later updates."""
    if len(partials) < 2:
        return 1.0

    total_words = 0
    unchanged_words = 0

    for i in range(len(partials) - 1):
        words_current = partials[i]["transcript"].split()
        words_next = partials[i + 1]["transcript"].split()
        total_words += len(words_current)
        for j, w in enumerate(words_current):
            if j < len(words_next) and words_next[j] == w:
                unchanged_words += 1

    return unchanged_words / total_words if total_words > 0 else 1.0


def _compute_partial_latencies(chunk_send_times: list[float], partials: list[dict]) -> list[float]:
    """For each partial, find the most recent chunk_send_time and compute delta."""
    latencies = []
    for p in partials:
        recv_t = p["recv_t"]
        preceding = [t for t in chunk_send_times if t <= recv_t]
        if preceding:
            latencies.append(recv_t - preceding[-1])
    return latencies


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "streaming"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"
    completed = get_completed_ids(jsonl_path)

    logger.info("=== Test 1.3: Streaming Performance ===")

    # Load datasets
    ls_ds = load_dataset("esb/datasets", "librispeech", split="test", token=True)
    ls_subset = list(ls_ds.select(range(min(200, len(ls_ds)))))

    try:
        ted_ds = load_dataset("esb/datasets", "tedlium", split="test", token=True)
        ted_subset = list(ted_ds.select(range(min(100, len(ted_ds)))))
    except Exception:
        ted_subset = []
        logger.warning("TED-LIUM not available; using LibriSpeech only")

    all_items = [(f"ls_{i}", ex) for i, ex in enumerate(ls_subset)]
    all_items += [(f"ted_{i}", ex) for i, ex in enumerate(ted_subset)]

    ttwf_values = []
    final_lat_values = []
    streaming_rtf_values = []
    stability_values = []
    partial_lat_values = []

    for item_id, example in tqdm(all_items, desc="Streaming"):
        if item_id in completed:
            continue

        audio = example["audio"]
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        ref_text = example.get("norm_text") or example.get("text", "")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sr)
            tmp_path = tmp.name

        try:
            result = stt_client.stream_recognize(tmp_path)

            stability = _compute_stability_rate(result["partials"])
            p_lats = _compute_partial_latencies(
                result["chunk_send_times"], result["partials"]
            )

            record = {
                "id": item_id,
                "reference": ref_text,
                "hypothesis": result["transcript"],
                "ttfw": result["ttfw"],
                "finalization_latency": result["finalization_latency"],
                "streaming_rtf": result["streaming_rtf"],
                "stability_rate": stability,
                "n_partials": len(result["partials"]),
                "mean_partial_latency": float(np.mean(p_lats)) if p_lats else None,
                "audio_duration_s": result["audio_duration"],
            }
            write_jsonl(jsonl_path, record)

            if result["ttfw"] is not None:
                ttwf_values.append(result["ttfw"])
            if result["finalization_latency"] is not None:
                final_lat_values.append(result["finalization_latency"])
            streaming_rtf_values.append(result["streaming_rtf"])
            stability_values.append(stability)
            partial_lat_values.extend(p_lats)

        except Exception as e:
            logger.warning("Streaming failed for %s: %s", item_id, e)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ---- Streaming vs Batch WER comparison ----
    logger.info("Running streaming vs batch WER comparison...")
    batch_refs, batch_hyps = [], []
    stream_refs, stream_hyps = [], []
    compare_subset = ls_subset[:500]

    for i, example in enumerate(tqdm(compare_subset, desc="WER comparison")):
        audio = example["audio"]
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
        ref = example.get("norm_text") or example.get("text", "")
        audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()

        try:
            batch_result = stt_client.recognize_batch(audio_bytes, sr)
            batch_refs.append(ref)
            batch_hyps.append(batch_result["transcript"])
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sr)
            try:
                stream_result = stt_client.stream_recognize(tmp.name)
                stream_refs.append(ref)
                stream_hyps.append(stream_result["transcript"])
            except Exception:
                pass
            finally:
                Path(tmp.name).unlink(missing_ok=True)

    batch_wer = jiwer.wer(batch_refs, batch_hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER) if batch_refs else None
    stream_wer = jiwer.wer(stream_refs, stream_hyps, reference_transform=NORMALIZE_FOR_WER, hypothesis_transform=NORMALIZE_FOR_WER) if stream_refs else None

    # ---- Summary ----
    summary = {
        "ttfw": compute_percentiles(ttwf_values) if ttwf_values else {},
        "finalization_latency": compute_percentiles(final_lat_values) if final_lat_values else {},
        "streaming_rtf": compute_percentiles(streaming_rtf_values) if streaming_rtf_values else {},
        "stability_rate_mean": float(np.mean(stability_values)) if stability_values else None,
        "partial_latency": compute_percentiles(partial_lat_values) if partial_lat_values else {},
        "batch_wer": batch_wer,
        "stream_wer": stream_wer,
        "wer_delta": (stream_wer - batch_wer) if batch_wer and stream_wer else None,
        "n_items": len(all_items),
    }

    save_summary_csv(results_dir / "summary.csv", [summary])

    logger.info("TTFW P50=%.0fms, Finalization P50=%.0fms, RTF P50=%.3f, Stability=%.1f%%",
        summary["ttfw"].get("p50", 0) * 1000,
        summary["finalization_latency"].get("p50", 0) * 1000,
        summary["streaming_rtf"].get("p50", 0),
        (summary["stability_rate_mean"] or 0) * 100,
    )

    return {"test": "1.3", "name": "streaming_performance", "results": summary}
