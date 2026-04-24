"""Test 1.11 — Audio Format & Input Robustness."""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm

from eval.config import Config
from eval.stt.client import STTClient
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.format_robustness")

SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]
BIT_DEPTHS = ["pcm_16", "pcm_32", "float32"]
FORMATS = ["wav", "flac", "mp3", "ogg"]
DURATION_EXTREMES = [0.5, 5, 60, 600]


def _create_test_audio(sr: int = 16000, duration_s: float = 5.0) -> np.ndarray:
    """Generate a sine wave as test audio."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


def _convert_format(audio: np.ndarray, sr: int, fmt: str, bit_depth: str = "pcm_16") -> tuple[bytes, int]:
    """Convert audio to specified format and return bytes + sample rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        wav_path = tmp.name

    seg = AudioSegment.from_wav(wav_path)
    Path(wav_path).unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as out_tmp:
        if fmt == "wav":
            seg.export(out_tmp.name, format="wav")
        elif fmt == "flac":
            seg.export(out_tmp.name, format="flac")
        elif fmt == "mp3":
            seg.export(out_tmp.name, format="mp3")
        elif fmt == "ogg":
            seg.export(out_tmp.name, format="ogg", codec="libvorbis")
        else:
            seg.export(out_tmp.name, format=fmt)

        with open(out_tmp.name, "rb") as f:
            data = f.read()
        Path(out_tmp.name).unlink(missing_ok=True)

    return data, sr


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "format_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)

    stt_client = STTClient(config)
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.11: Audio Format & Input Robustness ===")

    base_audio = _create_test_audio(sr=16000, duration_s=5.0)
    summary_rows = []

    # ---- Sample rate test ----
    for sr in tqdm(SAMPLE_RATES, desc="Sample rates"):
        audio = _create_test_audio(sr=sr, duration_s=5.0)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, sr)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "sample_rate", "value": sr, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Format test ----
    for fmt in tqdm(FORMATS, desc="Formats"):
        try:
            audio_data, sr = _convert_format(base_audio, 16000, fmt)
            # For non-WAV formats, try REST endpoint which may accept different content types
            result = stt_client.recognize_batch_rest(audio_data, 16000, content_type=f"audio/{fmt}")
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "format", "value": fmt, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Channels test ----
    for channels in [1, 2]:
        try:
            if channels == 2:
                stereo = np.column_stack([base_audio, base_audio])
                audio_bytes = (stereo * 32768).astype(np.int16).tobytes()
            else:
                audio_bytes = (base_audio * 32768).astype(np.int16).tobytes()
            result = stt_client.recognize_batch(audio_bytes, 16000, channels=channels)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "channels", "value": channels, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Duration extremes ----
    for dur in tqdm(DURATION_EXTREMES, desc="Duration extremes"):
        audio = _create_test_audio(sr=16000, duration_s=dur)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, 16000)
            status = "pass"
            error = None
        except Exception as e:
            status = "fail"
            error = str(e)

        record = {"test": "duration", "value": dur, "status": status, "error": error}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    # ---- Edge cases ----
    edge_cases = [
        ("silence", np.zeros(16000 * 5, dtype=np.float32)),
        ("white_noise", np.random.randn(16000 * 5).astype(np.float32) * 0.5),
    ]
    for name, audio in edge_cases:
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        try:
            result = stt_client.recognize_batch(audio_bytes, 16000)
            status = "pass"
            transcript = result["transcript"]
        except Exception as e:
            status = "fail"
            transcript = str(e)

        record = {"test": "edge_case", "value": name, "status": status, "transcript": transcript}
        write_jsonl(jsonl_path, record)
        summary_rows.append(record)

    save_summary_csv(results_dir / "summary.csv", summary_rows)

    passed = sum(1 for r in summary_rows if r["status"] == "pass")
    total = len(summary_rows)
    logger.info("Format robustness: %d/%d passed", passed, total)

    return {"test": "1.11", "name": "format_robustness", "results": summary_rows}
