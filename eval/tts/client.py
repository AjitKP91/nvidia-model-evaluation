from __future__ import annotations

import io
import logging
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf

import riva.client

from eval.config import Config
from eval.utils import retry_with_backoff

logger = logging.getLogger("eval.tts.client")


def _is_permanent_tts_error(exc: Exception) -> bool:
    """Return True for errors that will never succeed on retry."""
    # Check both str() and gRPC .details() — __str__ format varies by grpcio version.
    candidates = [str(exc)]
    try:
        candidates.append(exc.details())  # type: ignore[attr-defined]
    except AttributeError:
        pass
    msg = " ".join(candidates)
    return "longer than maximum sequence length" in msg or "Input sentence is longer" in msg


class TTSClient:
    def __init__(self, config: Config):
        self.config = config
        self.tts_cfg = config.tts
        self.riva_cfg = config.riva

        self.auth = riva.client.Auth(
            uri=self.tts_cfg.grpc_uri,
            use_ssl=self.riva_cfg.use_ssl,
            metadata_args=[
                ["authorization", f"Bearer {self.riva_cfg.auth_token}"]
            ],
        )
        self.tts_service = riva.client.SpeechSynthesisService(self.auth)
        self.sample_rate = self.tts_cfg.sample_rate

    # ------------------------------------------------------------------
    # gRPC batch
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_tts_error)
    def synthesize_batch(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        response = self.tts_service.synthesize(
            text,
            voice_name=self.tts_cfg.voice_name,
            language_code=self.tts_cfg.language_code,
            sample_rate_hz=self.sample_rate,
        )
        elapsed = time.perf_counter() - start

        audio_bytes = response.audio
        audio_duration = len(audio_bytes) / (self.sample_rate * 2)

        return {
            "audio_bytes": audio_bytes,
            "audio_duration": audio_duration,
            "elapsed_s": elapsed,
            "rtf": elapsed / audio_duration if audio_duration > 0 else None,
            "interface": "grpc",
            "mode": "batch",
        }

    # ------------------------------------------------------------------
    # REST batch
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_tts_error)
    def synthesize_batch_rest(self, text: str) -> dict[str, Any]:
        headers = {
            self.tts_cfg.auth_header: f"Bearer {self.riva_cfg.auth_token}",
        }
        data = {
            "text": text,
            "voice": self.tts_cfg.voice_name,
            "language": self.tts_cfg.language_code,
            "sample_rate_hz": str(self.sample_rate),
        }

        start = time.perf_counter()
        resp = requests.post(
            self.tts_cfg.rest_endpoint,
            data=data,
            headers=headers,
            timeout=self.tts_cfg.request_timeout_s,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()

        audio_bytes = resp.content
        audio_duration = len(audio_bytes) / (self.sample_rate * 2)

        return {
            "audio_bytes": audio_bytes,
            "audio_duration": audio_duration,
            "elapsed_s": elapsed,
            "rtf": elapsed / audio_duration if audio_duration > 0 else None,
            "interface": "rest",
            "mode": "batch",
            "http_status": resp.status_code,
        }

    # ------------------------------------------------------------------
    # gRPC streaming
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_tts_error)
    def synthesize_stream(self, text: str) -> dict[str, Any]:
        start = time.perf_counter()
        first_chunk_time = None
        all_chunks: list[bytes] = []
        chunk_times: list[float] = []

        responses = self.tts_service.synthesize_online(
            text,
            voice_name=self.tts_cfg.voice_name,
            language_code=self.tts_cfg.language_code,
            sample_rate_hz=self.sample_rate,
        )

        for resp in responses:
            recv_t = time.perf_counter()
            if first_chunk_time is None:
                first_chunk_time = recv_t
            all_chunks.append(resp.audio)
            chunk_times.append(recv_t)

        total_elapsed = time.perf_counter() - start
        audio_bytes = b"".join(all_chunks)
        audio_duration = len(audio_bytes) / (self.sample_rate * 2)

        return {
            "audio_bytes": audio_bytes,
            "audio_duration": audio_duration,
            "elapsed_s": total_elapsed,
            "ttfb": (
                first_chunk_time - start if first_chunk_time else None
            ),
            "rtf": (
                total_elapsed / audio_duration if audio_duration > 0 else None
            ),
            "n_chunks": len(all_chunks),
            "chunk_times": chunk_times,
            "interface": "grpc",
            "mode": "streaming",
        }

    # ------------------------------------------------------------------
    # REST streaming
    # ------------------------------------------------------------------

    @retry_with_backoff(reraise_if=_is_permanent_tts_error)
    def synthesize_stream_rest(self, text: str) -> dict[str, Any]:
        headers = {
            self.tts_cfg.auth_header: f"Bearer {self.riva_cfg.auth_token}",
        }
        data = {
            "text": text,
            "voice": self.tts_cfg.voice_name,
            "language": self.tts_cfg.language_code,
            "sample_rate_hz": str(self.sample_rate),
        }

        start = time.perf_counter()
        resp = requests.post(
            self.tts_cfg.rest_endpoint,
            data=data,
            headers=headers,
            timeout=self.tts_cfg.request_timeout_s,
            stream=True,
        )

        first_chunk_time = None
        all_chunks: list[bytes] = []

        for chunk in resp.iter_content(chunk_size=4096):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            all_chunks.append(chunk)

        total_elapsed = time.perf_counter() - start
        resp.raise_for_status()

        audio_bytes = b"".join(all_chunks)
        audio_duration = len(audio_bytes) / (self.sample_rate * 2)

        return {
            "audio_bytes": audio_bytes,
            "audio_duration": audio_duration,
            "elapsed_s": total_elapsed,
            "ttfb": (
                first_chunk_time - start if first_chunk_time else None
            ),
            "rtf": (
                total_elapsed / audio_duration if audio_duration > 0 else None
            ),
            "interface": "rest",
            "mode": "streaming",
            "http_status": resp.status_code,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def save_synthesis(
        self, text: str, output_path: str | Path, interface: str = "grpc"
    ) -> dict[str, Any]:
        if interface == "rest":
            result = self.synthesize_batch_rest(text)
        else:
            result = self.synthesize_batch(text)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio_array = np.frombuffer(result["audio_bytes"], dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        sf.write(str(output_path), audio_float, self.sample_rate)

        result["output_path"] = str(output_path)
        return result

    def bytes_to_wav(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0
        return audio_float, self.sample_rate
