from __future__ import annotations

import io
import logging
import time
import wave
from typing import Any

import numpy as np
import requests
import soundfile as sf

import riva.client

from eval.config import Config
from eval.utils import retry_with_backoff

logger = logging.getLogger("eval.stt.client")


class STTClient:
    def __init__(self, config: Config):
        self.config = config
        self.stt_cfg = config.stt
        self.riva_cfg = config.riva

        self.auth = riva.client.Auth(
            uri=self.stt_cfg.grpc_uri,
            use_ssl=self.riva_cfg.use_ssl,
            metadata_args=[
                ["authorization", f"Bearer {self.riva_cfg.auth_token}"]
            ],
        )
        self.asr_service = riva.client.ASRService(self.auth)

    # ------------------------------------------------------------------
    # gRPC batch
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def recognize_batch(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        channels: int = 1,
        enable_punctuation: bool = True,
        enable_word_times: bool = True,
    ) -> dict[str, Any]:
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=self.stt_cfg.language_code,
            max_alternatives=1,
            enable_automatic_punctuation=enable_punctuation,
            enable_word_time_offsets=enable_word_times,
            sample_rate_hertz=sample_rate,
            audio_channel_count=channels,
        )

        start = time.perf_counter()
        response = self.asr_service.offline_recognize(audio_bytes, config)
        elapsed = time.perf_counter() - start

        transcript = ""
        confidence = 0.0
        words = []
        if response.results:
            alt = response.results[0].alternatives[0]
            transcript = alt.transcript
            confidence = alt.confidence
            words = [
                {
                    "word": w.word,
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": getattr(w, "confidence", None),
                }
                for w in getattr(alt, "words", [])
            ]

        return {
            "transcript": transcript,
            "confidence": confidence,
            "words": words,
            "elapsed_s": elapsed,
            "interface": "grpc",
            "mode": "batch",
            "raw_response": response,
        }

    # ------------------------------------------------------------------
    # REST batch
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def recognize_batch_rest(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        import io
        import wave

        # audio_bytes is raw PCM int16 — wrap it in a WAV container
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        wav_buf.seek(0)

        headers = {
            self.stt_cfg.auth_header: f"Bearer {self.riva_cfg.auth_token}",
        }
        files = {
            "file": ("audio.wav", wav_buf, "audio/wav"),
        }
        data = {
            "model": self.stt_cfg.model_name,
            "language": self.stt_cfg.language_code,
        }

        start = time.perf_counter()
        resp = requests.post(
            self.stt_cfg.rest_endpoint,
            files=files,
            data=data,
            headers=headers,
            timeout=self.stt_cfg.request_timeout_s,
        )
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        result = resp.json()

        transcript = result.get("text", "")

        return {
            "transcript": transcript,
            "confidence": 0.0,
            "words": [],
            "elapsed_s": elapsed,
            "interface": "rest",
            "mode": "batch",
            "http_status": resp.status_code,
            "raw_response": result,
        }

    # ------------------------------------------------------------------
    # gRPC streaming
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def stream_recognize(
        self,
        audio_path: str,
        chunk_duration_s: float = 0.1,
    ) -> dict[str, Any]:
        with wave.open(audio_path) as wf:
            sr = wf.getframerate()
            chunk_frames = int(sr * chunk_duration_s)
            total_frames = wf.getnframes()
            audio_duration = total_frames / sr

        streaming_config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=self.stt_cfg.language_code,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=sr,
            ),
            interim_results=True,
        )

        partial_results: list[dict] = []
        final_results: list[dict] = []
        first_word_time = None
        chunk_send_times: list[float] = []
        last_chunk_send_time = None

        def audio_chunks():
            nonlocal last_chunk_send_time
            with wave.open(audio_path) as wf:
                while True:
                    data = wf.readframes(chunk_frames)
                    if not data:
                        break
                    send_t = time.perf_counter()
                    chunk_send_times.append(send_t)
                    last_chunk_send_time = send_t
                    yield data
                    time.sleep(chunk_duration_s)

        stream_start = time.perf_counter()

        responses = self.asr_service.streaming_response_generator(
            audio_chunks=audio_chunks(),
            streaming_config=streaming_config,
        )
        for response in responses:
            recv_t = time.perf_counter()
            for result in response.results:
                transcript = result.alternatives[0].transcript.strip()
                if not result.is_final:
                    if first_word_time is None and transcript:
                        first_word_time = recv_t
                    partial_results.append({
                        "recv_t": recv_t,
                        "transcript": transcript,
                        "stability": getattr(result, "stability", None),
                        "elapsed_from_start": recv_t - stream_start,
                    })
                else:
                    final_results.append({
                        "recv_t": recv_t,
                        "transcript": transcript,
                    })

        stream_end = time.perf_counter()

        final_result = final_results[-1] if final_results else None
        final_transcript = final_result["transcript"] if final_result else ""

        ttfw = (
            first_word_time - chunk_send_times[0]
            if first_word_time and chunk_send_times
            else None
        )
        finalization_latency = (
            final_result["recv_t"] - last_chunk_send_time
            if final_result and last_chunk_send_time
            else None
        )

        return {
            "transcript": final_transcript,
            "ttfw": ttfw,
            "finalization_latency": finalization_latency,
            "streaming_rtf": (stream_end - stream_start) / audio_duration,
            "audio_duration": audio_duration,
            "total_elapsed": stream_end - stream_start,
            "chunk_send_times": chunk_send_times,
            "partials": partial_results,
            "finals": final_results,
            "interface": "grpc",
            "mode": "streaming",
        }

    # ------------------------------------------------------------------
    # REST streaming (chunked transfer / SSE — endpoint-dependent)
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def stream_recognize_rest(
        self,
        audio_path: str,
    ) -> dict[str, Any]:
        import io as _io

        audio, sr = sf.read(audio_path, dtype="int16")
        buf = _io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        audio_bytes = buf.getvalue()

        headers = {
            self.stt_cfg.auth_header: f"Bearer {self.riva_cfg.auth_token}",
        }
        files = {
            "file": ("audio.wav", _io.BytesIO(audio_bytes), "audio/wav"),
        }
        form_data = {
            "model": self.stt_cfg.model_name,
            "language": self.stt_cfg.language_code,
        }

        start = time.perf_counter()
        resp = requests.post(
            self.stt_cfg.rest_endpoint,
            files=files,
            data=form_data,
            headers=headers,
            timeout=self.stt_cfg.request_timeout_s,
            stream=True,
        )
        first_byte_time = None
        chunks = []
        for chunk in resp.iter_content(chunk_size=4096):
            if first_byte_time is None:
                first_byte_time = time.perf_counter()
            chunks.append(chunk)

        elapsed = time.perf_counter() - start
        resp.raise_for_status()

        import json as _json
        body = b"".join(chunks)
        data = {}
        try:
            data = _json.loads(body)
        except Exception:
            pass

        transcript = data.get("text", "")

        return {
            "transcript": transcript,
            "elapsed_s": elapsed,
            "ttfb": (first_byte_time - start if first_byte_time else None),
            "interface": "rest",
            "mode": "streaming",
            "http_status": resp.status_code,
            "raw_response": data,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def audio_to_bytes(self, path: str, target_sr: int | None = None) -> tuple[bytes, int]:
        audio, sr = sf.read(path, dtype="int16")
        if target_sr and sr != target_sr:
            import librosa
            audio_f = audio.astype(np.float32) / 32768.0
            audio_f = librosa.resample(audio_f, orig_sr=sr, target_sr=target_sr)
            audio = (audio_f * 32768).astype(np.int16)
            sr = target_sr
        return audio.tobytes(), sr
