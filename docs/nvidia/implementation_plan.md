# Implementation Plan: NVIDIA Model Evaluation Harness

## Context

We have a detailed evaluation plan (`docs/nvidia/evaluation_plan.md`) for testing NVIDIA STT (Parakeet) and TTS (Magpie Aria) models deployed on SAP AI Core. The project currently has only documentation — no code. We need to implement the entire evaluation harness: Phase 0 discovery, all 11 STT tests, all 9 TTS tests, data download scripts, and report generation. Both REST and gRPC interfaces are required. The harness will run on an Azure NC24ads A100 VM in Frankfurt (same region as AI Core).

## Architecture

```
eval/
├── __init__.py
├── config.py              # Config loader (YAML → dataclass)
├── config.yaml            # Endpoints, auth, model names
├── utils.py               # Shared: text normalization, audio I/O, stats, logging, retry
├── run.py                 # CLI entrypoint: run individual or all tests
├── stt/
│   ├── __init__.py
│   ├── client.py          # STTClient: gRPC (riva) + REST wrappers
│   ├── accuracy.py        # Test 1.1
│   ├── performance.py     # Test 1.2
│   ├── streaming.py       # Test 1.3
│   ├── rest_vs_grpc.py    # Test 1.4
│   ├── noise_robustness.py # Test 1.5
│   ├── accent.py          # Test 1.6
│   ├── long_form.py       # Test 1.7
│   ├── domain.py          # Test 1.8
│   ├── output_quality.py  # Test 1.9
│   ├── confidence.py      # Test 1.10
│   └── format_robustness.py # Test 1.11
├── tts/
│   ├── __init__.py
│   ├── client.py          # TTSClient: gRPC (riva) + REST wrappers
│   ├── naturalness.py     # Test 2.1
│   ├── intelligibility.py # Test 2.2
│   ├── prosody.py         # Test 2.3
│   ├── signal_quality.py  # Test 2.4
│   ├── latency.py         # Test 2.5
│   ├── concurrency.py     # Test 2.6
│   ├── edge_cases.py      # Test 2.7
│   └── long_form.py       # Test 2.8
├── phase0/
│   ├── __init__.py
│   └── discovery.py       # Phase 0: connectivity, schema, smoke tests, cold start
├── data/
│   ├── __init__.py
│   ├── download_datasets.py  # HuggingFace/MUSAN/DEMAND download
│   ├── harvard_sentences.txt # 200 Harvard Sentences (static)
│   └── tts_test_sets.py      # TTS test case definitions (edge cases, categories)
└── report/
    ├── __init__.py
    └── generate_report.py    # HTML summary from results/ JSONL
```

Plus at project root:
- `requirements.txt` — all dependencies
- `results/stt/` and `results/tts/` — output directories

## Implementation Order (28 files)

### Layer 0: Infrastructure (7 files)

1. **`requirements.txt`** — all pip dependencies
2. **`eval/config.yaml`** — endpoints, auth, evaluation params
3. **`eval/config.py`** — load YAML into typed dataclasses (`RivaConfig`, `STTConfig`, `TTSConfig`, `EvalConfig`)
4. **`eval/utils.py`** — shared utilities:
   - `normalize_text()` — jiwer Compose pipeline
   - `load_audio()` / `save_audio()` — soundfile wrappers
   - `write_jsonl()` / `read_jsonl()` — result persistence
   - `compute_percentiles()` — P50/P90/P95/P99
   - `bootstrap_ci()` — 95% CI via bootstrap resampling
   - `retry_with_backoff()` — exponential backoff decorator for API calls
   - `setup_logging()` — structured logging
5. **`eval/stt/client.py`** — `STTClient` class:
   - `__init__(config)` — create riva.client.Auth + ASRService, store REST endpoint
   - `recognize_batch(audio_bytes, sample_rate) → dict` — gRPC batch
   - `recognize_batch_rest(audio_bytes, sample_rate) → dict` — REST batch
   - `stream_recognize(audio_path, chunk_duration_s=0.1) → dict` — gRPC streaming with full timing instrumentation
   - `stream_recognize_rest(audio_path) → dict` — REST streaming
   - Each method returns a standardized dict with transcript, confidence, timing, word_info
6. **`eval/tts/client.py`** — `TTSClient` class:
   - `__init__(config)` — create riva SpeechSynthesisService, store REST endpoint
   - `synthesize_batch(text) → dict` — gRPC batch, returns audio bytes + timing
   - `synthesize_batch_rest(text) → dict` — REST batch
   - `synthesize_stream(text) → dict` — gRPC streaming with TTFB + per-chunk timing
   - `synthesize_stream_rest(text) → dict` — REST streaming
   - `save_synthesis(text, output_path) → dict` — synthesize and save to WAV
7. **`eval/data/download_datasets.py`** — download all datasets:
   - LibriSpeech (test-clean, test-other) via ESB
   - TED-LIUM v3 test
   - GigaSpeech test (2000 utterance subset)
   - SPGISpeech test (2000 utterance subset)
   - Earnings-22 test
   - AMI IHM test
   - Common Voice EN test
   - L2-ARCTIC
   - MUSAN noise files
   - DEMAND noise files
   - LJSpeech (for TTS reference)
   - VCTK test prompts
   - LibriTTS test transcripts

### Layer 1: Phase 0 (1 file)

8. **`eval/phase0/discovery.py`** — implements all Phase 0 checks:
   - `check_connectivity()` — ping gRPC + REST endpoints
   - `discover_stt_schema()` — send known audio, log response fields
   - `discover_stt_streaming_schema()` — open stream, log partial/final fields
   - `discover_tts_schema()` — send text, log audio format/encoding
   - `explore_parameters()` — test sample rates, max durations, SSML
   - `smoke_tests()` — 4 baseline checks from plan
   - `cold_start_test()` — t_cold / t_warm ratio
   - `run_all()` — execute full Phase 0, write `results/phase0/discovery.json`

### Layer 2: STT Tests (11 files)

Each test file follows the same pattern:
- A `run(stt_client, config) → dict` function that executes the full test
- Results written to `results/stt/<test_name>/` as JSONL + summary CSV
- Returns summary dict for the report

9. **`eval/stt/accuracy.py`** — Test 1.1: Clean Speech Accuracy
   - Load 7 datasets via ESB
   - Batch recognize all utterances with idempotent resume
   - Compute WER/CER/MER/WIL/SER per dataset using jiwer
   - Bootstrap CI (N=1000)
   - Write per-utterance JSONL + summary CSV

10. **`eval/stt/performance.py`** — Test 1.2: Batch Performance
    - 5 duration buckets × 20 utterances = 100 calls
    - Measure RTF/RTFx per call
    - Concurrency test: ThreadPoolExecutor at N=1,5,10,20
    - Compute P50/P90/P95/P99 per bucket and concurrency

11. **`eval/stt/streaming.py`** — Test 1.3: Streaming Performance
    - 300 utterances (200 LibriSpeech + 100 TED-LIUM)
    - Use client.stream_recognize() with full instrumentation
    - Compute TTFW, Partial Result Latency, Finalization Latency, Streaming RTF
    - Stability Rate from partial diffs
    - Streaming vs Batch WER comparison (500 utterance subset)

12. **`eval/stt/rest_vs_grpc.py`** — Test 1.4: REST vs gRPC
    - 50 × batch REST + 50 × batch gRPC on same 30s audio
    - 50 × streaming REST + 50 × streaming gRPC
    - Compare P50/P90/P99 between interfaces

13. **`eval/stt/noise_robustness.py`** — Test 1.5: Noise Robustness
    - 100 LibriSpeech utterances × 4 noise types × 5-6 SNR levels
    - audiomentations for Gaussian; MUSAN/DEMAND for real noise
    - WER per condition, degradation curves

14. **`eval/stt/accent.py`** — Test 1.6: Accent Robustness
    - Common Voice EN filtered by accent (100+ per group)
    - L2-ARCTIC (6 groups)
    - Per-group WER + fairness delta

15. **`eval/stt/long_form.py`** — Test 1.7: Long-Form Audio
    - Earnings-21, TED-LIUM full talks, concatenated LibriSpeech
    - Full file vs 30s chunked WER comparison
    - Hallucination detection (5-gram sequences absent from reference)
    - Repetition detection (sliding 5-gram overlap)

16. **`eval/stt/domain.py`** — Test 1.8: Domain Vocabulary
    - SPGISpeech + Earnings-22
    - KRR via spaCy NER
    - NE-WER (entity-filtered WER)
    - OOV rate analysis

17. **`eval/stt/output_quality.py`** — Test 1.9: Output Quality
    - GigaSpeech, SPGISpeech, Earnings-22 (punctuated refs)
    - Punctuation F1 per mark (. , ? !)
    - Capitalisation accuracy
    - Number formatting accuracy

18. **`eval/stt/confidence.py`** — Test 1.10: Confidence Calibration
    - 1000+ utterances with confidence scores
    - ECE (bin-weighted), AUROC, Brier Score
    - Reliability diagram
    - Deletion exclusion policy, insertion inclusion

19. **`eval/stt/format_robustness.py`** — Test 1.11: Format Robustness
    - Re-encode one utterance to each format × sample rate × bit depth
    - pydub for transcoding
    - Record pass/fail/degraded per combination

### Layer 3: TTS Tests (8 files)

20. **`eval/tts/naturalness.py`** — Test 2.1: Automated Naturalness
    - Synthesize 200 sentences (Harvard + VCTK + LibriTTS)
    - UTMOS, DNSMOS (OVRL/SIG/BAK), NISQA scoring
    - Distribution stats

21. **`eval/tts/intelligibility.py`** — Test 2.2: Round-Trip WER
    - 350 sentences across 4 categories
    - Synthesize → Whisper large-v3 → jiwer WER/CER
    - Per-category breakdown

22. **`eval/tts/prosody.py`** — Test 2.3: Prosody
    - 50 LJSpeech-matched sentences
    - F0 extraction (pyworld), F0 RMSE, F0 Pearson r
    - Speaking rate (WPM)
    - Duration RMSE (MFA alignment)
    - nPVI computation

23. **`eval/tts/signal_quality.py`** — Test 2.4: Audio Signal Quality
    - PESQ, STOI, MCD against matched-speaker reference (if available)
    - Skip/caveat if no matched-speaker reference
    - MCD via librosa MFCC + DTW

24. **`eval/tts/latency.py`** — Test 2.5: Streaming TTFB & RTF
    - 5 text length buckets × 50 requests × 2 interfaces = 500 calls
    - TTFB, RTF measurement
    - REST vs gRPC comparison

25. **`eval/tts/concurrency.py`** — Test 2.6: Throughput
    - asyncio + aiohttp (REST) and async gRPC
    - N = 1, 5, 10, 20, 50
    - 100 requests per level
    - RPS, P50/P99, error rate, UTMOS quality check

26. **`eval/tts/edge_cases.py`** — Test 2.7: Edge Cases
    - ~130 test cases across 17 categories
    - Per-case: pass/fail, status code, UTMOS, round-trip WER

27. **`eval/tts/long_form.py`** — Test 2.8: Long-Form Consistency
    - 5 passages × 5-7 paragraphs
    - ECAPA-TDNN speaker embeddings per paragraph
    - Pairwise cosine similarity, F0 drift, speaking rate drift

### Layer 4: Data + Report + Runner (3 files)

28. **`eval/data/tts_test_sets.py`** — TTS test case definitions
    - Harvard sentences list
    - Edge case definitions (all 17 categories from plan)
    - Text length bucket samples
    - Long-form passages

29. **`eval/report/generate_report.py`** — HTML report generator
    - Read all results/ JSONL files
    - Apply pass/fail thresholds from plan
    - Generate summary tables, charts (matplotlib), overall verdict
    - Output: `results/report.html`

30. **`eval/run.py`** — CLI runner
    - `python -m eval.run phase0` — run Phase 0 only
    - `python -m eval.run stt` — run all STT tests
    - `python -m eval.run tts` — run all TTS tests
    - `python -m eval.run stt.accuracy` — run single test
    - `python -m eval.run all` — full evaluation
    - `python -m eval.run report` — generate report
    - Logging, progress tracking, error handling

Plus: `eval/data/harvard_sentences.txt` (static data file)

## Key Design Decisions

- **Idempotent resume**: Every API call result is written to JSONL immediately. On re-run, check if item ID already exists in output and skip.
- **Standardized client return**: Both STT and TTS clients return dicts with the same keys regardless of REST/gRPC, so test code is interface-agnostic.
- **Retry with backoff**: Decorator for all API calls — handles 429 / RESOURCE_EXHAUSTED.
- **Test 2.9 (MUSHRA)** is excluded from code — it's a human eval that needs WebMUSHRA deployed separately. The data prep (synthesizing test sentences + generating anchor audio) will be in `eval/tts/edge_cases.py` or a separate script if needed.
- **Test 2.4 signal quality**: Implemented but will self-skip and log a warning if no matched-speaker reference is found.

## Verification

After implementation:
1. `python -m eval.run phase0` should connect to AI Core, discover schemas, run smoke tests
2. `python -m eval.run stt.accuracy --dry-run` should load datasets and show what would be run
3. All test files should be importable: `python -c "from eval.stt import accuracy"`
4. `pip install -r requirements.txt` should install without errors
5. Config validation: `python -c "from eval.config import load_config; load_config()"` should parse config.yaml
