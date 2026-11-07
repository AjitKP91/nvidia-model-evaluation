"""Test 1.18 — Speaker Diarization (serving-layer feature honor check).

``RecognitionConfig.diarization_config.enable_speaker_diarization`` asks the
recognizer to tag each word with a ``speaker_tag``. The honor question: does the
gateway run diarization, or accept the flag against a streaming model that never
emits speaker tags? We self-construct a two-speaker clip, recognize it with
``diarization=True``, and check whether ≥2 distinct non-null ``speaker_tag``s
come back.

Two-speaker construction (the deployment verifies only one EN-US voice,
``EN-US.Aria``): speaker A is Aria; speaker B is a second distinct EN-US persona
if the deployment serves one (probed at run time), otherwise a pitch-shifted copy
of Aria. The ``construction`` field records which path was used so the verdict is
never fabricated. Audio is generated in-run into this test's own output dir.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from eval.config import Config
from eval.stt.client import STTClient
from eval.tts.client import TTSClient, _call_with_deadline, _is_permanent_tts_error
from eval.utils import save_summary_csv, write_jsonl

logger = logging.getLogger("eval.stt.diarization")

PRIMARY_VOICE = "Magpie-Multilingual.EN-US.Aria"
# Candidate second EN-US personas (the deployment's persona chain is
# Aria → Isabela → Sofia); tried in order, first that synthesises wins.
CANDIDATE_SECOND_VOICES = [
    "Magpie-Multilingual.EN-US.Isabela",
    "Magpie-Multilingual.EN-US.Sofia",
]

TURN_A = "Good morning, thank you all for joining the call today."
TURN_B = "Absolutely, let us begin by reviewing last quarter's results."


def _to_float(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _synth(tts_client: TTSClient, config: Config, text: str, voice: str | None) -> bytes | None:
    try:
        res = _call_with_deadline(
            lambda: tts_client.synthesize_batch(text, voice_name=voice),
            timeout_s=config.tts.request_timeout_s,
        )
        return res["audio_bytes"]
    except Exception as e:
        logger.info("Synth failed for voice=%s: %s", voice, e)
        return None


def run(config: Config) -> dict:
    results_dir = Path(config.evaluation.results_dir) / "stt" / "diarization"
    results_dir.mkdir(parents=True, exist_ok=True)
    stt_client = STTClient(config)
    tts_client = TTSClient(config)
    sr = tts_client.sample_rate
    jsonl_path = results_dir / "calls.jsonl"

    logger.info("=== Test 1.18: Speaker Diarization ===")

    # Speaker A — the one verified EN-US voice.
    audio_a = _synth(tts_client, config, TURN_A, PRIMARY_VOICE)
    if audio_a is None:
        summary = {"verdict": "no_data", "construction": "primary_synth_failed"}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.18", "name": "diarization", "results": summary}

    # Speaker B — try a second distinct persona, else pitch-shift speaker A.
    audio_b = None
    construction = None
    second_voice = None
    for cand in CANDIDATE_SECOND_VOICES:
        audio_b = _synth(tts_client, config, TURN_B, cand)
        if audio_b is not None:
            construction = "two_voices"
            second_voice = cand
            break

    if audio_b is None:
        # Fall back: synthesise turn B with Aria, then pitch-shift to make a
        # distinct-sounding second speaker. Honest about being a surrogate.
        audio_b_same = _synth(tts_client, config, TURN_B, PRIMARY_VOICE)
        if audio_b_same is None:
            summary = {"verdict": "no_data", "construction": "secondary_synth_failed"}
            save_summary_csv(results_dir / "summary.csv", [summary])
            return {"test": "1.18", "name": "diarization", "results": summary}
        try:
            import librosa
            yb = _to_float(audio_b_same)
            yb_shift = librosa.effects.pitch_shift(yb, sr=sr, n_steps=-4.0)
            audio_b = (np.clip(yb_shift, -1.0, 1.0) * 32768).astype(np.int16).tobytes()
            construction = "pitch_shift"
        except Exception as e:
            logger.warning("Pitch-shift fallback failed: %s", e)
            audio_b = audio_b_same
            construction = "same_voice"

    # Concatenate A + 0.3s silence + B into one two-speaker clip.
    gap = np.zeros(int(sr * 0.3), dtype=np.float32)
    clip = np.concatenate([_to_float(audio_a), gap, _to_float(audio_b)])
    clip_bytes = (np.clip(clip, -1.0, 1.0) * 32768).astype(np.int16).tobytes()

    try:
        result = stt_client.recognize_batch(
            clip_bytes, sr, diarization=True, max_speaker_count=2,
        )
    except Exception as e:
        logger.warning("Diarized recognize failed: %s", e)
        summary = {"verdict": "error", "construction": construction, "error": str(e)[:200]}
        save_summary_csv(results_dir / "summary.csv", [summary])
        return {"test": "1.18", "name": "diarization", "results": summary}

    words = result.get("words", [])
    tags = [w.get("speaker_tag") for w in words]
    non_null = [t for t in tags if t is not None]
    distinct = sorted(set(non_null))

    write_jsonl(jsonl_path, {
        "id": "diar_0",
        "construction": construction,
        "second_voice": second_voice,
        "n_words": len(words),
        "n_words_tagged": len(non_null),
        "distinct_tags": [int(t) for t in distinct],
        "transcript": result.get("transcript", ""),
    })

    if len(distinct) >= 2:
        verdict = "honored"              # multiple speakers distinguished
    elif non_null:
        verdict = "partial"              # tags present but only one speaker id
    else:
        verdict = "ignored"              # flag accepted, no speaker_tag emitted

    summary = {
        "verdict": verdict,
        "construction": construction,
        "second_voice": second_voice or "",
        "n_words": len(words),
        "n_words_tagged": len(non_null),
        "distinct_speaker_tags": len(distinct),
    }
    save_summary_csv(results_dir / "summary.csv", [summary])
    logger.info("Diarization verdict=%s construction=%s distinct_tags=%d",
                verdict, construction, len(distinct))
    return {"test": "1.18", "name": "diarization", "results": summary}
