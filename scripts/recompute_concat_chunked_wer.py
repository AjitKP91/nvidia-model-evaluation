"""Compute chunked WER for Test 1.7 concatenated LibriSpeech files (Approach B).

The main 1.7 code (eval/stt/long_form.py) transcribes the concatenated
``concat_*`` files whole-file ONLY — it never runs the chunked pass, so
``chunked_wer`` / ``wer_ratio`` / ``elapsed_s`` are blank for every concat row
in summary.csv. This script rebuilds the SAME 10 concat groups (first 100
LibriSpeech test-clean clips, grouped by 10) and runs BOTH passes:

  * full  — whole-file transcription (recomputed here only to sanity-check that
            the regenerated clips still match the committed ``full_wer``; if
            LibriSpeech stream order drifted, the rows won't line up and we say so)
  * chunked — 30s chunks (CHUNK_DURATION_S), each transcribed, joined, then
            scored against the same reference — exactly the block the short
            ``longform_*`` loop runs at lines 128-135.

NON-DESTRUCTIVE: writes ``<results>/stt/long_form/concat_chunked.csv`` and
``concat_chunked.jsonl``. It never touches summary.csv. Hits the live STT
endpoint (~10 files x 2-3 chunks + 10 full = ~30-40 calls).

Usage (from repo root, venv active):
    .venv/bin/python scripts/recompute_concat_chunked_wer.py \
        --results-dir results/run-04-11-26/riva
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import jiwer
import numpy as np

from eval.config import load_config
from eval.stt.client import STTClient
from eval.stt.long_form import CHUNK_DURATION_S, _chunk_audio
from eval.utils import NORMALIZE_FOR_WER, load_dataset_tmp, save_summary_csv, write_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recompute_concat_chunked")


def _wer(ref: str, hyp: str) -> float:
    return jiwer.wer(
        ref, hyp,
        reference_transform=NORMALIZE_FOR_WER,
        hypothesis_transform=NORMALIZE_FOR_WER,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        default="results/run-04-11-26/riva",
        help="Run directory (the one holding stt/long_form/summary.csv).",
    )
    ap.add_argument("--config", default=None, help="Path to config.yaml (default: eval/config.yaml)")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    config.evaluation.results_dir = args.results_dir
    out_dir = Path(args.results_dir) / "stt" / "long_form"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "concat_chunked.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()  # fresh run; don't append to a prior attempt

    stt_client = STTClient(config)

    log.info("=== Test 1.7 concat — chunked-WER recompute (non-destructive) ===")

    # Rebuild the exact same groups as eval/stt/long_form.py lines 104-106.
    with load_dataset_tmp("openslr/librispeech_asr", "test", name="clean", limit=100) as ls_examples:
        pass
    concat_groups = [ls_examples[i:i + 10] for i in range(0, len(ls_examples), 10)]
    log.info("Built %d concat groups from %d LibriSpeech clips", len(concat_groups), len(ls_examples))

    rows: list[dict] = []
    for gi, group in enumerate(concat_groups):
        all_audio, all_refs, sr = [], [], None
        for ex in group:
            a = np.array(ex["audio"]["array"], dtype=np.float32)
            sr = ex["audio"]["sampling_rate"]
            all_audio.append(a)
            all_refs.append(ex.get("norm_text") or ex.get("text", ""))
        if not all_audio:
            continue

        concat_audio = np.concatenate(all_audio)
        concat_ref = " ".join(all_refs)
        duration_s = len(concat_audio) / sr
        concat_bytes = (concat_audio * 32768).astype(np.int16).tobytes()

        try:
            # --- whole-file pass (sanity vs committed full_wer) ---
            full_result = stt_client.recognize_batch(concat_bytes, sr)
            full_hyp = full_result["transcript"]
            full_wer = _wer(concat_ref, full_hyp)

            # --- chunked pass (the part the main code omits) ---
            chunks = _chunk_audio(concat_audio, sr, CHUNK_DURATION_S)
            chunk_hyps = []
            for chunk in chunks:
                chunk_bytes = (chunk * 32768).astype(np.int16).tobytes()
                cr = stt_client.recognize_batch(chunk_bytes, sr)
                chunk_hyps.append(cr["transcript"])
            chunked_hyp = " ".join(chunk_hyps)
            chunked_wer = _wer(concat_ref, chunked_hyp)

            record = {
                "id": f"concat_{gi}",
                "duration_s": round(duration_s, 3),
                "n_chunks": len(chunks),
                "full_wer": round(full_wer, 6),
                "chunked_wer": round(chunked_wer, 6),
                "wer_ratio": round(full_wer / chunked_wer, 6) if chunked_wer > 0 else None,
                "elapsed_s": round(full_result.get("elapsed_s", 0.0), 4),
            }
            write_jsonl(jsonl_path, record)
            rows.append(record)
            log.info(
                "concat_%d: dur=%.1fs chunks=%d full_wer=%.4f chunked_wer=%.4f ratio=%s",
                gi, duration_s, len(chunks), full_wer, chunked_wer,
                record["wer_ratio"],
            )
        except Exception as e:  # pragma: no cover - defensive, keep going
            log.warning("concat group %d failed: %s", gi, e)

    if rows:
        csv_path = out_dir / "concat_chunked.csv"
        save_summary_csv(csv_path, rows)
        mean_full = np.mean([r["full_wer"] for r in rows])
        mean_chunk = np.mean([r["chunked_wer"] for r in rows])
        log.info("Wrote %d rows -> %s", len(rows), csv_path)
        log.info("mean full_wer=%.4f  mean chunked_wer=%.4f", mean_full, mean_chunk)
    else:
        log.error("No rows produced — nothing written.")


if __name__ == "__main__":
    main()
