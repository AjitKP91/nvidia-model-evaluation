# NVIDIA AICore Model Evaluation Plan

**Models under test:**
- STT: `parakeet-1.1b-en-US-asr-offline`
- TTS: `Magpie-Multilingual.EN-US.Aria`

**Interface:** REST API and gRPC on SAP AI Core  
**Scope:** Standalone quality and performance characterization  
**Languages:** English only  
**Ground truth:** Public benchmark datasets (no proprietary reference data)  
**Human evaluation:** Automated metrics throughout; optional MUSHRA for TTS final sign-off

---

## Table of Contents

1. [Constraints & Architectural Notes](#1-constraints--architectural-notes)
2. [Tooling Stack](#2-tooling-stack)
3. [Phase 0 — Discovery & Infrastructure](#3-phase-0--discovery--infrastructure)
4. [Phase 1A — STT Evaluation](#4-phase-1a--stt-evaluation-parakeet-1.1b-en-us-asr-offline)
   - [Test 1.1 — Clean Speech Accuracy](#test-11--clean-speech-accuracy)
   - [Test 1.2 — Batch Processing Performance (RTF / RTFx)](#test-12--batch-processing-performance-rtf--rtfx)
   - [Test 1.3 — Streaming Performance (TTFW / Partial Latency / Finalization / Stability)](#test-13--streaming-performance)
   - [Test 1.4 — REST vs gRPC Comparison](#test-14--rest-vs-grpc-comparison)
   - [Test 1.5 — Noise Robustness](#test-15--noise-robustness)
   - [Test 1.6 — Accent & Dialect Robustness](#test-16--accent--dialect-robustness)
   - [Test 1.7 — Long-Form Audio](#test-17--long-form-audio)
   - [Test 1.8 — Domain-Specific Vocabulary](#test-18--domain-specific-vocabulary)
   - [Test 1.9 — Output Quality (Punctuation / Capitalisation / Numbers)](#test-19--output-quality)
   - [Test 1.10 — Confidence Score Calibration](#test-110--confidence-score-calibration)
   - [Test 1.11 — Audio Format & Input Robustness](#test-111--audio-format--input-robustness)
5. [Phase 1B — TTS Evaluation](#5-phase-1b--tts-evaluation-magpie-multilingualenusaria)
   - [Test 2.1 — Automated Naturalness (UTMOS / DNSMOS)](#test-21--automated-naturalness)
   - [Test 2.2 — Intelligibility: ASR Round-Trip WER](#test-22--intelligibility-asr-round-trip-wer)
   - [Test 2.3 — Prosody (F0 / Duration / Rate / nPVI)](#test-23--prosody)
   - [Test 2.4 — Audio Signal Quality (MCD / PESQ / STOI)](#test-24--audio-signal-quality)
   - [Test 2.5 — Streaming TTFB & RTF (REST vs gRPC)](#test-25--streaming-ttfb--rtf)
   - [Test 2.6 — Throughput & Concurrency](#test-26--throughput--concurrency)
   - [Test 2.7 — Edge Cases & Input Robustness](#test-27--edge-cases--input-robustness)
   - [Test 2.8 — Long-Form Voice Consistency](#test-28--long-form-voice-consistency)
   - [Test 2.9 — Human Evaluation (MUSHRA)](#test-29--human-evaluation-mushra)
6. [Phase 2 — Voice Agent Evaluation (Outline)](#6-phase-2--voice-agent-evaluation-outline)
7. [Evaluation Harness Design](#7-evaluation-harness-design)
8. [Pass / Fail Thresholds Summary](#8-pass--fail-thresholds-summary)
9. [Deliverables & Suggested Timeline](#9-deliverables--suggested-timeline)

---

## 1. Constraints & Architectural Notes

### STT model

| Property | Detail |
|:---|:---|
| Model name | `parakeet-1.1b-en-US-asr-offline` |
| Architecture | Parakeet-TDT 1.1 B parameters, CTC/TDT-family |
| Mode | **Batch (offline) + Streaming** — both endpoints confirmed available |
| Primary language | en-US |
| Expected HF baseline | WER ~7% mean, RTFx ~2,391 on A100 |

Both batch and streaming modes will be evaluated. Streaming adds per-chunk partial-result latency metrics and a streaming-vs-batch WER comparison. Despite the "offline" suffix in the model name, the streaming endpoint is available on AICore.

### TTS model

| Property | Detail |
|:---|:---|
| Model name | `Magpie-Multilingual.EN-US.Aria` |
| Supported languages | English (evaluation scope) |
| Synthesis mode | Batch (REST) and streaming audio (gRPC) both available |
| Interface | REST returns complete audio; gRPC streams audio chunks |

### Measurement context

All metrics are measured **client-side** against the AICore endpoints. No access to Triton Prometheus metrics, server GPU counters, or the Riva C++ perf clients is assumed. Every timing measurement must be instrumented in the evaluation harness.

---

## 2. Tooling Stack

### Model API layer

| Package | Install | Role |
|:---|:---|:---|
| **`nvidia-riva-client`** | `pip install nvidia-riva-client` | Primary SDK for gRPC calls — wraps Riva ASR and TTS protos; provides `ASRService`, `SpeechSynthesisService`, pre-compiled stubs, auth/TLS helpers |
| `requests` | `pip install requests` | REST calls (batch STT and TTS); fallback if Riva client can't reach AICore |
| `aiohttp` | `pip install aiohttp` | Async REST for concurrency/throughput tests (Test 1.2, 2.6) |

> **`nvidia-riva-client` caveat:** Only works if AICore's gRPC endpoint implements the standard Riva proto. Verify in Phase 0.1. If the proto differs, fall back to raw `grpcio` with hand-generated stubs from AICore's proto definitions.

#### Key `nvidia-riva-client` classes

```python
import riva.client

# Connect to AICore gRPC endpoint
auth = riva.client.Auth(
    uri="<aicore-grpc-host>:<port>",
    use_ssl=True,
    metadata_args=[["authorization", "Bearer <token>"]],
)

asr_service = riva.client.ASRService(auth)
tts_service = riva.client.SpeechSynthesisService(auth)
```

| Use case | Method | Returns |
|:---|:---|:---|
| Batch STT | `asr_service.offline_recognize(audio_bytes, config)` | `RecognizeResponse` with `alternatives[].transcript`, `confidence`, `word_time_offsets` |
| Streaming STT | `asr_service.streaming_response_generator(audio_chunks, streaming_config)` | Iterator of `StreamingRecognizeResponse` with `results[].is_final`, `.stability`, `.alternatives` |
| Batch TTS | `tts_service.synthesize(text, voice_name=..., language_code=..., sample_rate_hz=...)` | `SynthesizeSpeechResponse` with `.audio` bytes |
| Streaming TTS | `tts_service.synthesize_online(text, voice_name=..., language_code=..., sample_rate_hz=...)` | Iterator of responses, each with `.audio` chunk |

#### Riva response fields used for KPIs

These fields come from the Riva SDK response objects (see `voice_agent_evaluation.md` §1.13 for full reference):

| Riva Field | Used In | KPI |
|:---|:---|:---|
| `alternative.confidence` | Test 1.10 | Confidence calibration (ECE, AUROC) |
| `WordInfo.start_time` / `end_time` | Test 1.9 | Word-level timing for punctuation/formatting analysis |
| `StreamingRecognitionResult.stability` | Test 1.3 | Stability Rate (fraction of interim words not revised) |
| `StreamingRecognitionResult.audio_processed` | Test 1.3 | Streaming RTF (compare `audio_processed` to wall-clock) |
| `StreamingRecognitionResult.is_final` | Test 1.3 | Distinguish partial vs final results for Finalization Latency |

### STT evaluation libraries

| Package | Install | KPIs Measured | Tests |
|:---|:---|:---|:---|
| **`jiwer`** | `pip install jiwer>=3.0` | WER, CER, MER, WIL, word alignment, SER (via `process_words`) | 1.1, 1.3, 1.5–1.8 |
| `evaluate` | `pip install evaluate` | HF wrapper for WER/CER (`evaluate.load("wer")`) | 1.1 (alternative) |
| **`audiomentations`** | `pip install audiomentations` | Noise injection: `AddGaussianNoise`, `AddBackgroundNoise` | 1.5 |
| **`spacy`** | `pip install spacy && python -m spacy download en_core_web_trf` | NER extraction for KRR and NE-WER | 1.8 |
| **`scikit-learn`** | `pip install scikit-learn` | ECE (`calibration_curve`), AUROC (`roc_auc_score`), Brier (`brier_score_loss`), Punctuation F1 (`precision_recall_fscore_support`) | 1.9, 1.10 |
| `datasets` | `pip install datasets` | HuggingFace dataset loading (LibriSpeech, Common Voice, ESB, etc.) | 1.1, 1.5–1.8 |
| `soundfile` | `pip install soundfile` | Audio I/O (WAV, FLAC read/write) | All STT tests |
| **`pydub`** | `pip install pydub` | Audio format conversion (MP3, OGG, sample rate resampling) for format robustness | 1.11 |

### TTS evaluation libraries

| Package | Install | KPIs Measured | Tests |
|:---|:---|:---|:---|
| **`utmos`** | `pip install utmos` | UTMOS neural MOS prediction (1–5 scale) | 2.1 |
| **`nisqa`** | `pip install nisqa` | NISQA quality prediction (CNN+attention, range 1–5) | 2.1 |
| **`pesq`** | `pip install pesq` | PESQ ITU-T P.862.2 (requires matched-speaker reference) | 2.4 (conditional) |
| **`pystoi`** | `pip install pystoi` | STOI / ESTOI intelligibility (requires matched-speaker reference) | 2.4 (conditional) |
| **`pyworld`** | `pip install pyworld` | F0 extraction (DIO + StoneMask) for F0 RMSE, F0 correlation | 2.3, 2.8 |
| **`dtw-python`** | `pip install dtw-python` | Dynamic Time Warping alignment for MCD computation | 2.4 (conditional) |
| `librosa` | `pip install librosa` | MFCC extraction (MCD), mel spectrograms, audio duration, speaking rate | 2.3, 2.4 |
| **`speechbrain`** | `pip install speechbrain` | ECAPA-TDNN speaker embeddings for SpeakerSim (cosine similarity) | 2.8 |
| `resemblyzer` | `pip install resemblyzer` | Fast speaker embeddings (lighter than SpeechBrain; use as fallback) | 2.8 (alternative) |
| **`openai-whisper`** | `pip install openai-whisper` | Whisper large-v3 for TTS→ASR round-trip WER | 2.2 |
| **`montreal-forced-aligner`** | `pip install montreal-forced-aligner` | Phoneme-level forced alignment for Duration RMSE and nPVI | 2.3 |

### DNSMOS setup

DNSMOS is not a simple pip install — Microsoft distributes it as an ONNX model with a Python wrapper.

```bash
# Option 1: Microsoft's official DNSMOS P.835 (requires ONNX runtime)
pip install onnxruntime
# Download DNSMOS model from https://github.com/microsoft/DNS-Challenge
# Use dns_challenge/DNSMOS/dnsmos_local.py

# Option 2: speechmetrics (bundles MOSNet, PESQ, STOI)
pip install speechmetrics
```

### Statistics, reporting, infrastructure

| Package | Install | Role | Tests |
|:---|:---|:---|:---|
| `numpy` | `pip install numpy` | Array operations, bootstrap resampling, percentile computation | All |
| `scipy` | `pip install scipy` | Wilcoxon signed-rank test (MUSHRA), Pearson/Spearman correlation (F0) | 2.3, 2.9 |
| `pandas` | `pip install pandas` | Results aggregation, CSV/HTML report tables | All |
| `matplotlib` | `pip install matplotlib` | Reliability diagrams, WER-vs-SNR curves, box plots, heatmaps | All |
| `tqdm` | `pip install tqdm` | Progress bars for long evaluation runs | All |

### Complete KPI → Tool mapping

Every KPI in the plan is assigned a specific measurement tool:

#### STT KPIs

| KPI | Primary Tool | Riva Field Used | Notes |
|:---|:---|:---|:---|
| WER / CER / MER / WIL | `jiwer` | transcript from `alternatives[0].transcript` | Apply text normalisation first |
| SER | `jiwer.process_words()` | transcript | Count utterances with ≥ 1 error |
| RTF / RTFx (batch) | `time.perf_counter()` + Riva `offline_recognize` | — | Manual client-side timing |
| TTFW (streaming) | `time.perf_counter()` + Riva `streaming_response_generator` | `is_final=False` first non-empty transcript | See Test 1.3 code |
| Partial Result Latency | `time.perf_counter()` per chunk | each `StreamingRecognizeResponse` | Attribute partials to most recent chunk send time |
| Finalization Latency | `time.perf_counter()` | `is_final=True` response | `t_final − t_last_chunk_sent` |
| TTFS | `time.perf_counter()` | `is_final=True` response | `t_final − t_audio_end` (end of audio stream) |
| Stability Rate | `nvidia-riva-client` | `StreamingRecognitionResult.stability` | If `stability` field available; else compute from partial diffs |
| Streaming RTF | `time.perf_counter()` | `audio_processed` | `t_elapsed / t_audio_duration` |
| Noise robustness WER | `audiomentations` + `jiwer` | — | Inject noise → transcribe → WER |
| Per-accent WER | `datasets` (Common Voice accent filter) + `jiwer` | — | Report fairness delta |
| Hallucination rate | Custom n-gram analysis | transcript | Flag 5-gram sequences absent from reference |
| Repetition rate | Custom sliding-window 5-gram overlap | transcript | Count repeated phrases |
| KRR (Keyword Recall) | `spacy` NER + set comparison | transcript | `correctly_transcribed_entities / total_entities` |
| NE-WER | `spacy` NER + `jiwer` | transcript | WER on entity-filtered pairs only |
| OOV Rate | Vocabulary file + set diff | reference text | `oov_words / total_words` |
| Punctuation F1 | `sklearn.metrics.precision_recall_fscore_support` | transcript (with punctuation) | Per-mark: `.` `,` `?` `!` |
| Capitalisation accuracy | Custom string comparison | transcript (with casing) | `correct_caps / total_cap_decisions` |
| Number formatting | Custom pattern matching | transcript | Compare numeric patterns against reference |
| ECE | `sklearn.calibration.calibration_curve` | `alternative.confidence` | Bin-weighted formula |
| AUROC | `sklearn.metrics.roc_auc_score` | `alternative.confidence` | Confidence as binary error predictor |
| Brier Score | `sklearn.metrics.brier_score_loss` | `alternative.confidence` | Overall calibration |
| Audio format support | `pydub` / `soundfile` + re-encoding | — | Transcode to each format, send, check pass/fail |

#### TTS KPIs

| KPI | Primary Tool | Notes |
|:---|:---|:---|
| UTMOS | `utmos.UTMOSScore` | Reference-free; wav2vec 2.0 backbone |
| DNSMOS OVRL / SIG / BAK | `dnsmos_local` (Microsoft ONNX) | Reference-free; calibrated against ITU-T P.835 |
| NISQA | `nisqa` | Reference-free; CNN + attention |
| Round-trip WER | `openai-whisper` (large-v3) + `jiwer` | Synthesise → transcribe → compare |
| Round-trip CER | `openai-whisper` + `jiwer.cer()` | Same pipeline |
| F0 RMSE | `pyworld` (DIO + StoneMask) + `numpy` | Voiced frames only; use correlation if speaker mismatch |
| F0 Pearson r | `pyworld` + `scipy.stats.pearsonr` | Contour shape; valid even with speaker mismatch |
| Duration RMSE | `montreal-forced-aligner` + `numpy` | Phoneme-level alignment |
| Speaking rate (WPM) | `librosa.get_duration()` + word count | `(n_words / duration) × 60` |
| nPVI | MFA phoneme durations + custom code | Normalized Pairwise Variability Index |
| MCD | `librosa` (MFCC) + `dtw-python` + `numpy` | Only valid with matched-speaker reference |
| PESQ | `pesq` (ITU-T P.862.2) | Only valid with matched-speaker reference |
| STOI / ESTOI | `pystoi` | Only valid with matched-speaker reference |
| Speaker similarity (cosine) | `speechbrain` ECAPA-TDNN | Pairwise cosine distance between embeddings |
| TTFB | `time.perf_counter()` + Riva `synthesize_online` | `t_first_chunk − t_request_sent` |
| TTS RTF | `time.perf_counter()` | `t_total_synthesis / t_audio_duration` |
| Concurrency throughput | `asyncio` + `aiohttp` (REST) or async gRPC | RPS and P99 at N concurrent |
| Long-form SpeakerSim variance | `speechbrain` ECAPA-TDNN + `numpy` | Pairwise cosine across paragraphs |
| F0 drift | `pyworld` + `numpy` | Std dev of per-paragraph mean F0 |
| MUSHRA | WebMUSHRA + Prolific Academic | Human evaluation; 0–100 scale |

### Reference models

| Role | Model | Package | Notes |
|:---|:---|:---|:---|
| TTS intelligibility ASR | Whisper large-v3 | `openai-whisper` | Run locally; standard across papers |
| Speaker similarity | ECAPA-TDNN | `speechbrain` (`spkrec-ecapa-voxceleb`) | Cosine similarity; Resemblyzer as fast fallback |
| MOS prediction | UTMOS | `utmos` | System-level Pearson r ≈ 0.95 on VoiceMOS-2022 |
| Background quality | DNSMOS P.835 | Microsoft ONNX model | Better than UTMOS at codec/artifact detection |
| NER tagger | en_core_web_trf | `spacy` | For KRR and NE-WER extraction |
| Forced alignment | Montreal Forced Aligner | `montreal-forced-aligner` | Phoneme-level duration for Duration RMSE, nPVI |

### Human evaluation (MUSHRA — Phase 1B Test 2.9)

- **WebMUSHRA** (`github.com/audiolabs/webMUSHRA`) — browser-based MUSHRA
- Rater pool: Prolific Academic (native English speakers, > 200 approved prior studies)
- Minimum 20 raters; target 30

---

## 3. Phase 0 — Discovery & Infrastructure

**Goal:** Verify connectivity, confirm response schema, determine supported audio formats, and establish baseline numbers for all subsequent tests.

### 0.1 Connectivity & Auth

- [ ] Confirm both REST endpoint URL and gRPC host:port for each model
- [ ] Validate authentication (API key / service binding credentials)
- [ ] Measure round-trip network latency to AICore (ping equivalent via a no-op or health call)
- [ ] Confirm TLS/mTLS configuration

### 0.2 STT — Schema Discovery

**Batch mode:** Send a known short audio clip (10 s, 16 kHz, 16-bit mono WAV, LibriSpeech utterance) and inspect the raw JSON response:

- [ ] Identify transcript field path
- [ ] Confirm whether word-level timestamps are returned (and field names)
- [ ] Confirm whether per-word or per-utterance confidence scores are returned
- [ ] Confirm whether punctuation and capitalisation appear in the raw transcript
- [ ] Note any metadata fields (model version, processing time if server-reported)
- [ ] Measure single-call latency as a baseline

**Streaming mode:** Open a gRPC streaming session with the same audio clip:

- [ ] Identify the gRPC service and method names (e.g., `StreamingRecognize`)
- [ ] Confirm that partial (non-final) results are returned during audio streaming
- [ ] Confirm the field path for `is_final`, `stability`, and `alternatives[].transcript`
- [ ] Verify that the stream terminates cleanly after the audio ends
- [ ] Measure TTFW and total stream duration as a baseline

### 0.3 TTS — Schema Discovery

Send a short text string ("Hello, this is a test.") and inspect the response:

- [ ] Identify audio encoding in REST response (WAV bytes, PCM, mp3?)
- [ ] Identify sample rate and bit depth from response headers or file header
- [ ] Confirm gRPC streaming chunk boundaries and audio encoding
- [ ] Measure time-to-complete-audio for short text (REST)
- [ ] Measure time-to-first-chunk for short text (gRPC)

### 0.4 Parameter Exploration

For STT:
- [ ] Test whether sample rate tolerance is documented (8 kHz / 16 kHz / 44.1 kHz)
- [ ] Test whether max audio duration is documented or empirically discoverable
- [ ] Test whether language hint or model override parameters exist

For TTS:
- [ ] Test whether speaking rate, pitch, or volume parameters are exposed
- [ ] Test whether voice selection or style parameters are available
- [ ] Test whether SSML markup is accepted

### 0.5 Baseline Smoke Tests

| Test | Pass Criteria |
|:---|:---|
| STT — 10 s LibriSpeech clean | WER < 15%, response received in < 30 s |
| STT — 60 s audio | Response received without timeout |
| TTS — 20-word sentence (REST) | Audio returned, > 1 s duration, no artifacts |
| TTS — 20-word sentence (gRPC) | First chunk received in < 5 s |

### 0.6 Cold-Start vs Warm Latency

The first request after deployment (or after an idle period) often incurs model-loading overhead. Measure this explicitly:

1. After a fresh deployment or extended idle period (> 30 min), send the first STT request and first TTS request. Record latency as `t_cold`.
2. Immediately send 10 more identical requests. Record mean latency as `t_warm`.
3. Report the cold-start penalty: `t_cold / t_warm`.

| Metric | Target |
|:---|:---|
| Cold-start penalty (STT) | < 5× warm latency |
| Cold-start penalty (TTS) | < 5× warm latency |
| Second-request latency | Within 20% of steady-state |

This informs auto-scaling decisions and keep-alive configuration in production.

---

## 4. Phase 1A — STT Evaluation: `parakeet-1.1b-en-US-asr-offline`

### STT Coverage Map

| KPI Group | Tests | Key Metrics |
|:---|:---|:---|
| Clean accuracy | 1.1 | WER, CER, MER, WIL, SER |
| Batch performance | 1.2 | RTF, RTFx, API latency P50/P90/P99 |
| Streaming performance | 1.3 | TTFW, Partial Result Latency, Finalization Latency, TTFS, Stability Rate |
| Interface comparison | 1.4 | REST vs gRPC latency delta (batch + streaming) |
| Noise robustness | 1.5 | WER vs SNR curve |
| Accent / fairness | 1.6 | Per-group WER, fairness delta |
| Long-form | 1.7 | WER on full files, hallucination rate, repetition rate |
| Domain vocabulary | 1.8 | KRR, OOV rate, domain WER |
| Output quality | 1.9 | Punctuation F1, capitalisation accuracy, number formatting |
| Confidence calibration | 1.10 | ECE, AUROC, reliability diagram |
| Format robustness | 1.11 | Support matrix across codecs and sample rates |

> **Out of scope — Speaker Diarization:** The Riva platform supports diarization via `speaker_tag` on `WordInfo`, but `parakeet-1.1b-en-US-asr-offline` is a single-speaker ASR model. If AICore exposes diarization parameters and multi-speaker audio is a use case, add a DER/JER test set (AMI IHM, DIHARD) in a future iteration.

---

### Test 1.1 — Clean Speech Accuracy

**Purpose:** Establish the accuracy baseline across diverse English speech styles; the primary pass/fail gate.

#### Datasets

| Dataset | Subset | Hours | Style | Metric focus |
|:---|:---|:---|:---|:---|
| LibriSpeech | test-clean | 5.4 h | Narrated audiobooks | Primary baseline |
| LibriSpeech | test-other | 5.4 h | Challenging narration | Generalisation |
| TED-LIUM v3 | test | 3 h | Prepared oratory | Prepared speech |
| GigaSpeech | test (2,000 utt subset) | ~4 h | Mixed broadcast/web | Mixed style |
| SPGISpeech | test (2,000 utt subset) | ~8 h | Financial calls | Domain / oratory |
| Earnings-22 | test | 5 h | Spontaneous business | Hard spontaneous |
| AMI IHM | test | 9 h | Meeting conversations | Hardest condition |

Load via HuggingFace `esb/datasets`:

```python
from datasets import load_dataset

ds = load_dataset("esb/datasets", "librispeech", split="test", trust_remote_code=True)
# ds[i]["audio"]["array"], ds[i]["audio"]["sampling_rate"], ds[i]["norm_text"]
```

#### Metrics

| Metric | Formula | Tool |
|:---|:---|:---|
| **WER** | `(S + D + I) / N` | `jiwer.wer()` |
| **CER** | `(S_c + D_c + I_c) / N_c` | `jiwer.cer()` |
| **MER** | `(S + D + I) / (S + D + I + H)` | `jiwer.mer()` |
| **WIL** | `1 − WIP` | `jiwer.wil()` |
| **SER** | utterances with ≥ 1 error / total | custom via `jiwer.process_words()` |

#### Text normalisation (apply identically to reference and hypothesis)

```python
import jiwer

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])

wer = jiwer.wer(reference, hypothesis,
                truth_transform=normalise,
                hypothesis_transform=normalise)
```

Run normalisation **before** metric computation for all datasets.

#### Statistical rigour

- Bootstrap resampling (N = 1,000 utterance samples) to compute 95% CI on WER per dataset
- Apply **MAPSSWE test** for pairwise comparisons if a second model variant is later added
- Minimum utterance count: 500 per dataset split for reliable benchmarking

#### Acceptance thresholds

| Dataset | Target WER | Good | Acceptable |
|:---|:---|:---|:---|
| LibriSpeech test-clean | < 5% | 5–7% | 7–10% |
| LibriSpeech test-other | < 10% | 10–15% | 15–20% |
| TED-LIUM v3 | < 8% | 8–12% | 12–18% |
| GigaSpeech | < 12% | 12–16% | 16–22% |
| SPGISpeech | < 8% | 8–12% | 12–18% |
| Earnings-22 | < 15% | 15–22% | 22–30% |
| AMI IHM | < 18% | 18–25% | 25–35% |

#### Output artefacts

- Per-utterance `{id, reference, hypothesis, WER, CER, substitutions, deletions, insertions}` → `results/stt/accuracy/per_utterance.jsonl`
- Aggregate per-dataset summary table → `results/stt/accuracy/summary.csv`
- Alignment visualisations for the 20 worst-WER utterances per dataset

---

### Test 1.2 — Batch Processing Performance (RTF / RTFx)

**Purpose:** Measure how fast the model processes audio relative to audio duration; determines fitness for batch pipelines and roughly estimates maximum throughput.

#### Method

For each of 5 audio duration buckets (10 s, 30 s, 60 s, 300 s, 600 s), sample 20 utterances from LibriSpeech / TED-LIUM, send each in isolation, and record:

```python
import time, wave
import riva.client

# auth setup (see Section 2)
config = riva.client.RecognitionConfig(
    language_code="en-US",
    max_alternatives=1,
    enable_automatic_punctuation=True,
    enable_word_time_offsets=True,
)

with wave.open(audio_path) as wf:
    audio_bytes = wf.readframes(wf.getnframes())
    audio_duration_seconds = wf.getnframes() / wf.getframerate()
    config.sample_rate_hertz = wf.getframerate()
    config.audio_channel_count = wf.getnchannels()

start = time.perf_counter()
response = asr_service.offline_recognize(audio_bytes, config)
elapsed = time.perf_counter() - start

rtf  = elapsed / audio_duration_seconds
rtfx = audio_duration_seconds / elapsed
```

#### Metrics

| Metric | Formula | Target |
|:---|:---|:---|
| **RTF** | `t_api_call / t_audio_duration` | < 1.0 (must); < 0.1 (good for GPU batch) |
| **RTFx** | `t_audio_duration / t_api_call` | > 10 (good); > 3 (acceptable) |
| **API latency P50** | 50th percentile of elapsed across all calls | — |
| **API latency P90** | — | — |
| **API latency P95** | — | — |
| **API latency P99** | — | — |
| **Throughput** | hours of audio / wall-clock hour | Derive from RTFx |

#### Concurrency test

Run N simultaneous requests (N = 1, 5, 10, 20) using `asyncio` or `ThreadPoolExecutor` on the same 30 s reference clip. Report P50/P99 per concurrency level. Identify the concurrency point where P99 degrades beyond 3× P50 (the scalability cliff).

#### Output artefacts

- Per-call `{duration_s, elapsed_s, rtf, rtfx, concurrency}` → `results/stt/performance/calls.jsonl`
- RTF vs. audio-duration scatter plot
- P50/P99 latency vs. concurrency chart

---

### Test 1.3 — Streaming Performance

**Purpose:** Measure real-time responsiveness of the streaming STT endpoint — the critical path for voice agent use. These metrics are distinct from batch RTF and measure latency experienced by a user during a live conversation.

#### Definitions

| Metric | Definition | Formula | Target |
|:---|:---|:---|:---|
| **TTFW** (Time to First Word) | Wall-clock from first audio chunk sent to first word token received | `t_first_word − t_first_chunk_sent` | < 600 ms |
| **Partial Result Latency** | Delay from audio chunk sent to corresponding partial transcript received | `t_partial − t_chunk_sent` | < 100 ms per chunk |
| **Finalization Latency** | Delay from last audio frame to final stable transcript | `t_final − t_last_frame` | < 300 ms |
| **TTFS** (Time to Final Segment) | From user silence (VAD end-of-speech) to final transcript | `t_final − t_silence` | < 400 ms |
| **Stability Rate** | Fraction of partial-result words not revised in subsequent stream updates | `unchanged_words / total_partial_words` | > 80% |
| **Streaming RTF** | Processing rate relative to real-time | `t_elapsed / t_audio_duration` | < 1.0 (must); < 0.3 (target) |

#### Method

Simulate a real-time streaming session by sending audio in 100 ms chunks at real-time pace:

```python
import time, wave
import riva.client

CHUNK_DURATION_S = 0.1   # 100 ms chunks

def stream_stt(audio_path: str, asr_service: riva.client.ASRService):
    with wave.open(audio_path) as wf:
        sr = wf.getframerate()
        chunk_frames = int(sr * CHUNK_DURATION_S)
        total_frames = wf.getnframes()
        audio_duration = total_frames / sr

    streaming_config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            sample_rate_hertz=sr,
        ),
        interim_results=True,   # receive partial transcripts
    )

    partial_results = []
    final_result = None
    first_word_time = None
    chunk_send_times = []
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
                time.sleep(CHUNK_DURATION_S)   # real-time pacing

    stream_start = time.perf_counter()

    responses = asr_service.streaming_response_generator(
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
                final_result = {
                    "recv_t": recv_t,
                    "transcript": transcript,
                }

    stream_end = time.perf_counter()

    return {
        # TTFW: first audio chunk sent → first word received
        "ttfw": first_word_time - chunk_send_times[0]
                if first_word_time and chunk_send_times else None,
        # Finalization Latency: last audio chunk sent → is_final=true received
        "finalization_latency": final_result["recv_t"] - last_chunk_send_time
                                if final_result and last_chunk_send_time else None,
        # Streaming RTF
        "streaming_rtf": (stream_end - stream_start) / audio_duration,
        # Raw data for per-chunk partial latency analysis
        "chunk_send_times": chunk_send_times,
        "partials": partial_results,
        "final": final_result,
    }
```

> **Per-chunk Partial Result Latency** is computed in post-processing: for each partial result, find the most recent `chunk_send_time` that precedes it, and compute the delta. This attributes partial transcripts to the audio chunk that likely triggered them.

#### Streaming vs batch WER comparison

Run the same 500-utterance LibriSpeech test-clean subset through both batch and streaming endpoints. Expected: streaming WER is 1–3 percentage points higher than batch.

| Comparison | Expected gap | Flag threshold |
|:---|:---|:---|
| Streaming WER − Batch WER | +1–3% | > +5% requires investigation |

#### Test corpus

- 200 utterances from LibriSpeech test-clean (varied lengths: 3–20 s)
- 100 utterances from TED-LIUM (longer, 15–60 s continuous speech)

#### Output artefacts

- TTFW / Finalization Latency / TTFS distribution (box plots)
- Streaming vs batch WER side-by-side table
- Stability Rate per audio length bucket
- Per-chunk partial latency scatter plot

---

### Test 1.4 — REST vs gRPC Comparison

**Purpose:** Quantify the overhead difference between the two interfaces across both batch and streaming modes; informs which to recommend for each use case.

#### Batch comparison

Use the identical 30 s LibriSpeech utterance, sent 50 times sequentially via REST and 50 times via gRPC (batch mode). Measure total elapsed time per call.

#### Streaming comparison

Stream the same 30 s utterance in 100 ms chunks at real-time pace, 50 repetitions each interface. Measure TTFW and Finalization Latency.

#### Metrics

| Metric | REST | gRPC |
|:---|:---|:---|
| Batch P50 latency | — | — |
| Batch P90 latency | — | — |
| Batch P99 latency | — | — |
| Streaming TTFW P50 | — | — |
| Streaming Finalization P50 | — | — |
| Failed requests | — | — |

Report absolute delta and percentage improvement of the faster interface per mode. gRPC is expected to be faster for streaming (lower TTFW); REST may be comparable or operationally simpler for one-shot batch calls.

---

### Test 1.5 — Noise Robustness

**Purpose:** Characterise accuracy degradation as a function of noise type and SNR level, for both batch and streaming modes.

> Run the full noise matrix through the **batch** endpoint first (primary results). Then run a reduced set (white Gaussian at 0, 5, 10, 20 dB × 50 utterances) through the **streaming** endpoint to check whether streaming handles noise differently (smaller context window may affect robustness).

#### Base corpus

100 utterances from LibriSpeech test-clean (~ 10 min total). These serve as the clean reference.

#### Noise conditions

| Noise type | Source | SNR levels |
|:---|:---|:---|
| White Gaussian | `audiomentations.AddGaussianNoise` | −5, 0, 5, 10, 15, 20 dB |
| Babble (crowd) | MUSAN speech noise | −5, 0, 5, 10, 15 dB |
| Traffic / street | DEMAND `STREET` environment | 0, 5, 10, 15, 20 dB |
| Office ambient | DEMAND `OFFICE` environment | 0, 5, 10, 15, 20 dB |

Total conditions: 4 noise types × 5–6 SNR levels = ~22 conditions × 100 utterances = ~2,200 calls.

#### Noise injection code

```python
from audiomentations import AddGaussianNoise, AddBackgroundNoise
import soundfile as sf, numpy as np

def inject_noise(audio_array, sr, snr_db, noise_type="gaussian"):
    if noise_type == "gaussian":
        aug = AddGaussianNoise(min_snr_in_db=snr_db, max_snr_in_db=snr_db, p=1.0)
        return aug(audio_array, sample_rate=sr)
    # For MUSAN/DEMAND: AddBackgroundNoise with a path to noise files
    aug = AddBackgroundNoise(
        sounds_path="data/noise/musan",
        min_snr_in_db=snr_db,
        max_snr_in_db=snr_db,
        p=1.0,
    )
    return aug(audio_array, sample_rate=sr)
```

#### Metrics

- WER at each SNR level
- WER degradation curve: `ΔWERrel = (WER_noisy − WER_clean) / WER_clean × 100%`
- SNR threshold at which WER doubles (`WER_noisy = 2 × WER_clean`)

#### Acceptance thresholds

| Condition | Acceptable WER |
|:---|:---|
| 20 dB SNR | ≤ 1.5× clean WER |
| 10 dB SNR | ≤ 2.0× clean WER |
| 5 dB SNR | ≤ 3.0× clean WER |
| 0 dB SNR | ≤ 5.0× clean WER |

#### Output artefacts

- WER-vs-SNR line chart (one line per noise type)
- Table: noise type × SNR → WER, ΔWER%, RTF

---

### Test 1.6 — Accent & Dialect Robustness

**Purpose:** Quantify per-accent WER gap and compute fairness deltas.

#### Datasets

| Dataset | Accents / Groups | Notes |
|:---|:---|:---|
| Common Voice EN (test) | Filter by `accent` metadata field | 10+ accent groups |
| L2-ARCTIC | Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese L1 speakers | 6 non-native English groups |
| CORAAL | African American Language dialects | Regional dialect variation |

Select at least 100 utterances per accent group where available.

#### Metrics

- WER per accent group
- **Fairness delta:** `max(WER_group) − min(WER_group)` — target < 10 percentage points
- Ranking of accent groups by difficulty

#### Output artefacts

- Bar chart: WER per accent group with 95% CI error bars
- Table: group, utterance count, WER, CI lower, CI upper

---

### Test 1.7 — Long-Form Audio

**Purpose:** Assess accuracy and stability on audio longer than typical training segments; detect hallucination and repetition artefacts.

#### Datasets

| Dataset | Description | File lengths |
|:---|:---|:---|
| Earnings-21 | Long-form earnings calls | Up to 60 min per file |
| TED-LIUM v3 (full talks) | 10–20 min prepared talks | Medium long-form |
| LibriSpeech (concatenated) | Concatenate 10 × 30 s clips → 5 min files | Controlled synthetic |

#### Method

1. Submit full audio files without pre-segmentation.
2. Compare WER on full files vs pre-segmented (30 s chunks, transcript joined).
3. **Hallucination detection:** cross-reference hypothesis words with reference; flag word sequences of > 5 tokens present in hypothesis but absent from reference.
4. **Repetition detection:** sliding window of 5-gram overlap; count repeated phrases.

#### Metrics

| Metric | Formula | Target |
|:---|:---|:---|
| Full-file WER | Standard WER on full transcript | < 1.5× chunk-mode WER |
| WER degradation | `WER_full / WER_chunked` | < 1.5× |
| Hallucination rate | hallucinated tokens / total hypothesis tokens | < 1% |
| Repetition rate | repeated 5-gram sequences / total 5-grams | < 1% |
| Processing time | Wall-clock seconds per hour of audio | — |

#### Output artefacts

- Full-file vs chunked WER table per dataset
- Flagged hallucination examples
- Flagged repetition examples

---

### Test 1.8 — Domain-Specific Vocabulary

**Purpose:** Assess accuracy on domain-specific terminology that may not appear in the model's training distribution.

#### Datasets

| Dataset | Domain | Notes |
|:---|:---|:---|
| SPGISpeech test | Financial earnings calls | Rich in tickers, company names, financial terms |
| Earnings-22 | Spontaneous financial/business | Harder than SPGISpeech |
| Custom wordlist | 200 technical / domain terms | Generated from representative production vocab |

#### Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **Domain WER** | WER on domain test set | Compare to general English baseline |
| **KRR (Keyword Recall Rate)** | correctly transcribed domain terms / total domain terms | Extract via NER tagger (spaCy `en_core_web_trf`) |
| **OOV Rate** | reference words not in a 200K-word vocabulary / total | Proxy for domain difficulty |
| **NE-WER** | WER computed only on named entities | `jiwer` on NE-filtered pairs |

#### Output artefacts

- Domain WER vs general baseline comparison
- Top 50 most-misrecognised domain terms
- KRR per category (company names, financial terms, technical acronyms)

---

### Test 1.9 — Output Quality

**Purpose:** Assess whether the model emits correctly formatted text for downstream NLP pipelines (punctuation, case, number formatting).

#### Datasets

Datasets that include punctuation and mixed-case reference transcripts:

- GigaSpeech test (punctuated)
- SPGISpeech test (punctuated + cased)
- Earnings-22 test (punctuated + cased)

Do **not** apply `RemovePunctuation()` normalisation for this test.

#### Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **Punctuation Precision** | TP_punct / (TP_punct + FP_punct) | Per punctuation mark |
| **Punctuation Recall** | TP_punct / (TP_punct + FN_punct) | Per punctuation mark |
| **Punctuation F1** | 2 × (Prec × Rec) / (Prec + Rec) | Per mark: `.` `,` `?` `!` |
| **Capitalisation Accuracy** | correctly_cased_tokens / total_tokens_requiring_case | On proper nouns, sentence starts |
| **Number Formatting Accuracy** | correct numeric outputs / total numeric references | "twenty-three" → "23" |

#### Output artefacts

- Per-mark Punctuation F1 table
- Capitalisation accuracy with worst-case examples
- Number formatting accuracy with failure mode breakdown

---

### Test 1.10 — Confidence Score Calibration

**Purpose:** Determine whether the confidence score returned by the model is a reliable predictor of transcription correctness.

> Prerequisite: Phase 0.2 confirmed that per-word or per-utterance confidence scores are available. If not exposed by AICore, skip this test.

#### Method

1. Collect predictions + confidence scores across 1,000+ utterances (LibriSpeech + GigaSpeech)
2. Compute ground-truth correctness label per word (1 = correct, 0 = substitution/deletion)
3. Bucket confidence into 10 decile bins
4. Compute mean confidence and mean accuracy per bin

#### Metrics

| Metric | Formula | Notes |
|:---|:---|:---|
| **ECE (Expected Calibration Error)** | `Σ (n_b / N) × |acc_b − conf_b|` | Bin-weighted; lower is better; target < 0.05 |
| **AUROC** | AUC of ROC curve: confidence as error predictor | > 0.75 = useful signal |
| **Reliability diagram** | Plot mean conf vs mean acc per decile | Visual inspection |
| **Brier Score** | `(1/N) × Σ (conf_i − label_i)²` | Overall calibration quality |

> **Handling deletions and insertions:** Confidence calibration operates on *predicted words* that have an associated score. Deletions (reference words missing from hypothesis) have no predicted token and therefore no confidence value — exclude them from per-word calibration. Insertions (extra words in hypothesis) do have confidence scores and should be included (label = 0, since they are errors). Report the deletion rate separately as a complementary statistic.

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

# confidence: array of floats [0,1]; correct: array of 0/1
fraction_of_positives, mean_predicted_value = calibration_curve(
    correct, confidence, n_bins=10
)
auroc = roc_auc_score(correct, confidence)
```

#### Output artefacts

- Reliability diagram plot
- ECE, AUROC, Brier Score table

---

### Test 1.11 — Audio Format & Input Robustness

**Purpose:** Determine supported input audio configurations; identify failure modes before production use.

#### Test matrix

| Dimension | Values to test |
|:---|:---|
| Sample rate | 8 kHz, 16 kHz, 22.05 kHz, 44.1 kHz, 48 kHz |
| Channels | Mono, stereo |
| Bit depth | 8-bit PCM, 16-bit PCM, 32-bit float |
| File format | WAV, MP3, FLAC, OGG (where supported) |
| Duration extremes | 0.5 s (very short), 5 s, 60 s, 600 s, 3,600 s |
| Audio content edge cases | Silence only, background music only, white noise only |

Use a single known-good LibriSpeech utterance re-encoded to each combination. Record:

- Whether a valid transcript is returned
- WER relative to 16 kHz / 16-bit / WAV baseline
- Error codes / HTTP status codes on failure

#### Output artefacts

- Support matrix: format × result (pass/fail/degraded)
- WER vs sample rate chart (resample vs native)

---

## 5. Phase 1B — TTS Evaluation: `Magpie-Multilingual.EN-US.Aria`

### TTS Coverage Map

| KPI Group | Test | Key Metrics |
|:---|:---|:---|
| Naturalness (automated) | 2.1 | UTMOS, DNSMOS OVRL/SIG/BAK |
| Intelligibility | 2.2 | Round-trip WER/CER (Whisper large-v3) |
| Prosody | 2.3 | F0 RMSE, Duration RMSE, WPM, nPVI |
| Signal quality | 2.4 | MCD, PESQ, STOI (vs LJSpeech reference) |
| Latency | 2.5 | TTFB P50/P90/P99, RTF (REST vs gRPC) |
| Throughput | 2.6 | Requests/sec, P99 under load |
| Edge cases | 2.7 | Error rate, WER degradation on difficult inputs |
| Long-form consistency | 2.8 | Inter-segment SpeakerSim variance, F0 drift |
| Human eval | 2.9 | MUSHRA mean ± 95% CI |

---

### Test 2.1 — Automated Naturalness

**Purpose:** Automated proxy for perceived speech quality without human raters; used for rapid iteration and regression detection.

#### Test set

200 phonetically balanced English sentences:
- 100 Harvard Sentences (`data/tts/harvard_sentences.txt`)
- 50 VCTK test prompts
- 50 randomly sampled LibriTTS test transcripts

#### Metrics

| Metric | Tool | Target |
|:---|:---|:---|
| **UTMOS** | `utmos` — wav2vec 2.0 backbone | > 4.0 (good); > 4.3 (excellent) |
| **DNSMOS OVRL** | `DNSMOS` P.835 | > 3.5 |
| **DNSMOS SIG** | Speech signal clarity | > 3.5 |
| **DNSMOS BAK** | Background noise / artifacts | > 4.0 |
| **NISQA** | CNN+attention quality predictor | > 3.5 |

```python
from utmos import UTMOSScore

scorer = UTMOSScore()
scores = [scorer.score(audio_path) for audio_path in synthesized_files]
utmos_mean = sum(scores) / len(scores)
```

#### Reporting

- Mean ± std for each metric across the 200-sentence set
- Per-sentence scores exported for failure analysis
- Distribution histogram (UTMOS score distribution)

---

### Test 2.2 — Intelligibility: ASR Round-Trip WER

**Purpose:** Measure whether synthesised speech can be reliably recognised; directly proxies for listener comprehension.

#### Method

1. Synthesise each test sentence via the TTS API
2. Transcribe with Whisper large-v3 locally
3. Compare hypothesis to original text with normalisation

```python
import whisper, jiwer

asr = whisper.load_model("large-v3")

normalise = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

def round_trip_wer(text: str, audio_path: str) -> float:
    result = asr.transcribe(audio_path)
    return jiwer.wer(text, result["text"],
                     truth_transform=normalise,
                     hypothesis_transform=normalise)
```

#### Test sets (4 categories)

| Category | Count | Examples |
|:---|:---|:---|
| Standard prose | 200 | Harvard Sentences, LibriTTS |
| Numbers & figures | 50 | "The total is $1,234.56", "Call 0800-123-456" |
| Abbreviations & acronyms | 50 | "The API returns JSON", "Dr Smith at NASA" |
| Technical / domain | 50 | "NVIDIA's CUDA 12.6 supports RTX 4090" |

#### Metrics

| Metric | Target |
|:---|:---|
| **Round-trip WER** (standard prose) | < 2% |
| **Round-trip WER** (numbers) | < 5% |
| **Round-trip WER** (abbreviations) | < 5% |
| **Round-trip WER** (technical) | < 5% |
| **CER** overall | < 1% |

#### Output artefacts

- Per-sentence WER table with failure mode tags
- Top 20 most error-prone input patterns
- WER breakdown by category

---

### Test 2.3 — Prosody

**Purpose:** Characterise pitch, rhythm, and pacing accuracy relative to natural human speech.

#### Reference audio

Use LJSpeech — a single female American English speaker with high-quality studio recordings. Select 50 sentences whose exact text appears in LJSpeech metadata to allow direct comparison.

> **Speaker-mismatch caveat:** Magpie Aria may have a different pitch range than LJSpeech (female, ~210 Hz mean F0). If the voices are different genders or significantly different pitch ranges, **F0 RMSE is not meaningful** — use F0 Pearson correlation (contour shape) instead of absolute RMSE, and report the mean F0 of each voice for context. Duration RMSE, speaking rate, and nPVI remain valid regardless of pitch mismatch.

#### Metrics

**F0 (pitch) analysis** via `pyworld`:

```python
import pyworld as pw, numpy as np, librosa

def extract_f0(audio_path: str):
    wav, sr = librosa.load(audio_path, sr=16000)
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr)      # raw F0 estimate
    f0 = pw.stonemask(wav, f0, t, sr)  # refined
    voiced = f0 > 0
    return f0[voiced]  # voiced frames only

f0_ref = extract_f0("ljspeech/LJ001-0001.wav")
f0_syn = extract_f0("synthesized/LJ001-0001.wav")

# Align lengths (DTW if duration differs significantly)
min_len = min(len(f0_ref), len(f0_syn))
rmse = np.sqrt(np.mean((f0_ref[:min_len] - f0_syn[:min_len]) ** 2))
corr = np.corrcoef(f0_ref[:min_len], f0_syn[:min_len])[0, 1]
```

| Metric | Formula | Target |
|:---|:---|:---|
| **F0 RMSE** | `√(mean((f0_syn − f0_ref)²))` (voiced frames) | < 20 Hz |
| **F0 Pearson r** | Correlation of F0 contours | > 0.60 |
| **Speaking Rate** | words / audio_duration × 60 | 130–180 WPM |
| **Duration RMSE** | `√(mean((d_syn(i) − d_ref(i))²))` (MFA phoneme alignment) | < 20 ms |
| **nPVI** | Normalized Pairwise Variability Index | 45–65 for American English |

**Speaking rate:**

```python
import librosa

def speaking_rate_wpm(text: str, audio_path: str) -> float:
    duration = librosa.get_duration(path=audio_path)
    word_count = len(text.split())
    return (word_count / duration) * 60
```

**nPVI:**

```python
def npvi(durations):
    pairs = [(durations[i], durations[i+1])
             for i in range(len(durations) - 1)]
    return 100 * sum(abs(a - b) / ((a + b) / 2)
                     for a, b in pairs) / len(pairs)
```

#### Output artefacts

- F0 contour overlay plots (reference vs synthesised) for 10 sample sentences
- Prosody summary table: F0 RMSE, F0 r, WPM mean/std, Duration RMSE, nPVI

---

### Test 2.4 — Audio Signal Quality

**Purpose:** Measure objective signal-level quality relative to a reference recording of the same text.

> **Important — speaker-mismatch limitation:** PESQ and STOI are designed to measure *signal degradation* from a reference (e.g., codec artifacts, noise). They assume the reference and test signals are the *same speech* from the *same speaker*. When comparing a TTS voice (Magpie Aria) against a different speaker (LJSpeech), PESQ/STOI scores reflect speaker differences, not quality differences, and are **not meaningful**.
>
> **When to use these metrics:**
> - **PESQ / STOI:** Only if Magpie Aria produces audio that can be paired against a *same-speaker* reference (e.g., a vocoder quality test where you have original Aria recordings). Skip if no matched-speaker reference exists.
> - **MCD:** Also requires same-speaker reference for meaningful results. Skip on speaker-mismatched pairs.
>
> **What to use instead (no reference needed):** UTMOS (Test 2.1), DNSMOS (Test 2.1), and round-trip WER (Test 2.2) are reference-free and always valid.

If Phase 0 reveals that Aria reference recordings are available (e.g., NVIDIA provides sample utterances from the same voice), run PESQ/STOI/MCD against those. Otherwise, mark this test as **skipped — no matched-speaker reference** and rely on Tests 2.1 and 2.2 for quality assessment.

#### Metrics

| Metric | Tool | Range | Target |
|:---|:---|:---|:---|
| **MCD** (Mel Cepstral Distortion) | `librosa` + `dtw-python` | dB (lower better) | < 6 dB |
| **PESQ** | `pesq` — ITU-T P.862.2 | −0.5 to 4.5 | > 2.5 |
| **STOI** | `pystoi` — ESTOI variant | 0 to 1 | > 0.80 |

```python
from pesq import pesq
from pystoi import stoi
import soundfile as sf

ref, sr_ref = sf.read("ljspeech/LJ001-0001.wav")
syn, sr_syn = sf.read("synthesized/LJ001-0001.wav")
# Resample to 16 kHz for both if needed

pesq_score = pesq(16000, ref, syn, 'wb')   # wideband
stoi_score = stoi(ref, syn, 16000, extended=True)  # ESTOI
```

#### MCD computation

```python
import librosa, numpy as np
from dtw import dtw

def compute_mcd(ref_path, syn_path, n_mfcc=13):
    ref, sr = librosa.load(ref_path, sr=22050)
    syn, _  = librosa.load(syn_path, sr=22050)
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc).T
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=sr, n_mfcc=n_mfcc).T
    alignment = dtw(ref_mfcc, syn_mfcc, dist_method='euclidean')
    return (10.0 * np.sqrt(2) / np.log(10)) * alignment.distance / len(ref_mfcc)
```

---

### Test 2.5 — Streaming TTFB & RTF

**Purpose:** Measure latency behaviour of both interfaces; the primary performance KPI for voice-agent readiness.

#### Definitions

| Metric | REST | gRPC |
|:---|:---|:---|
| **TTFB** | Time from HTTP request sent to first byte of audio body received | Time from stream open to first audio chunk received |
| **RTF** | `elapsed_total / audio_duration` | `elapsed_total / audio_duration` |

#### Text length buckets

| Bucket | Character count | Example |
|:---|:---|:---|
| Very short | 10–40 chars | "Hello, how can I help?" |
| Short | 40–100 chars | One sentence |
| Medium | 100–300 chars | 2–3 sentences |
| Long | 300–600 chars | Short paragraph |
| Very long | 600–1200 chars | Two paragraphs |

50 requests per bucket per interface = 500 calls total.

#### Measurement (gRPC)

```python
import time
import riva.client

SAMPLE_RATE = 22050  # or 44100; match model output

# auth setup (see Section 2)
start = time.perf_counter()
first_chunk_received = False
all_chunks = []

responses = tts_service.synthesize_online(
    text,
    voice_name="Magpie-Multilingual.EN-US.Aria",
    language_code="en-US",
    sample_rate_hz=SAMPLE_RATE,
)

for resp in responses:
    if not first_chunk_received:
        ttfb = time.perf_counter() - start
        first_chunk_received = True
    all_chunks.append(resp.audio)

total_elapsed = time.perf_counter() - start
audio_duration = len(b"".join(all_chunks)) / (SAMPLE_RATE * 2)  # 16-bit PCM
rtf = total_elapsed / audio_duration
```

#### Metrics

| Metric | Target |
|:---|:---|
| **TTFB P50** (gRPC, short text) | < 150 ms |
| **TTFB P90** (gRPC, short text) | < 300 ms |
| **TTFB P99** (gRPC, short text) | < 500 ms |
| **RTF** (REST, medium text) | < 0.5 |
| **TTFB REST vs gRPC delta** | gRPC should be faster for first-byte |

#### Output artefacts

- TTFB distribution box plots: REST vs gRPC per bucket
- RTF vs text length scatter plot
- P50/P90/P99 latency summary table

---

### Test 2.6 — Throughput & Concurrency

**Purpose:** Find the saturation point and verify the service handles parallel requests gracefully.

#### Method

Using `asyncio` with `aiohttp` (REST) and async gRPC stubs, send N concurrent synthesis requests for a 200-character text (medium bucket). N = 1, 5, 10, 20, 50.

Per concurrency level, run 100 total requests and record:

- Requests per second (RPS)
- P50 / P99 latency
- Error rate (HTTP 5xx / gRPC errors)
- Audio quality check: synthesise same sentence at each concurrency and verify UTMOS does not drop > 0.3

#### Output artefacts

- Latency and RPS vs concurrency chart
- Error rate table
- Recommended maximum safe concurrency (last N before P99 > 500 ms or error rate > 1%)

---

### Test 2.7 — Edge Cases & Input Robustness

**Purpose:** Identify inputs that cause synthesis failures, degraded quality, or unexpected behaviour.

#### Test cases

| Category | Examples | Count |
|:---|:---|:---|
| Integers | "23", "100000", "−45" | 10 |
| Decimals | "3.14", "0.001", "$1,234.56" | 10 |
| Percentages | "99.9%", "0.5%" | 5 |
| Dates | "January 15, 2024", "15/01/24", "2024-01-15" | 10 |
| Times | "3:30 PM", "15:30", "half past three" | 5 |
| Phone numbers | "+49 30 12345678", "0800-123-456" | 5 |
| URLs | "nvidia.com/riva", "https://docs.nvidia.com" | 5 |
| Email addresses | "support@nvidia.com" | 5 |
| Abbreviations | "Dr.", "Prof.", "Ltd.", "NASA", "API" | 10 |
| All-caps | "NVIDIA", "GPU", "THIS IS URGENT" | 10 |
| Mixed case | "iPhone", "MacBook", "YouTube" | 5 |
| Punctuation-heavy | "Wait! Stop? No—really.", bullet lists | 10 |
| Very short | Single word, single character | 5 |
| Very long | 1,000+ character paragraph | 5 |
| Empty / whitespace | `""`, `"   "`, `"\n"` | 5 |
| Special characters | `"©", "®", "°", "€", "¥"` | 10 |
| Mixed EN+DE | "Das meeting starts at 3 Uhr" | 5 |

Per test case, record:
- Whether audio is returned (pass/fail)
- HTTP/gRPC status code on failure
- UTMOS score where audio is returned
- Round-trip WER for intelligibility check
- Qualitative note on anomalies (mispronounced, skipped, robotic)

#### Output artefacts

- Edge case results table: input → status, UTMOS, round-trip WER, notes
- Curated list of failure modes for product team

---

### Test 2.8 — Long-Form Voice Consistency

**Purpose:** Verify that the voice characteristics remain stable across extended synthesis; detect pitch drift, timbre changes, or style shifts.

#### Method

1. Select 5 long-form passages of 500–800 words each (news articles, LibriTTS chapters)
2. Split each passage into paragraphs (~80–120 words each)
3. Synthesise each paragraph as a separate API call (simulating streaming agent use)
4. Extract ECAPA-TDNN speaker embeddings per paragraph

```python
from speechbrain.pretrained import EncoderClassifier
import torch, soundfile as sf

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa"
)

def embed(audio_path: str) -> torch.Tensor:
    wav, sr = sf.read(audio_path, dtype="float32")
    wav_tensor = torch.tensor(wav).unsqueeze(0)
    return classifier.encode_batch(wav_tensor).squeeze()
```

5. Compute pairwise cosine similarity between all paragraph embeddings within a passage
6. Report mean similarity and variance

#### Metrics

| Metric | Formula | Target |
|:---|:---|:---|
| **Intra-passage mean SpeakerSim** | Mean pairwise cosine sim | > 0.90 |
| **Intra-passage SpeakerSim variance** | Std dev of pairwise scores | < 0.05 |
| **F0 drift** | Std dev of per-paragraph mean F0 | < 15 Hz |
| **Speaking rate drift** | Std dev of per-paragraph WPM | < 20 WPM |

#### Output artefacts

- Speaker similarity heatmap per passage (paragraph × paragraph)
- F0 mean per paragraph line chart
- Consistency summary table

---

### Test 2.9 — Human Evaluation (MUSHRA)

**Purpose:** Ground-truth perceptual quality assessment by native English listeners. Triggered once automated metrics are stable; not a routine test.

#### Trigger condition

Run if any of:
- UTMOS mean < 3.8 or > 4.3 (automated score seems too low or surprisingly high)
- Round-trip WER > 3% on standard prose
- Product team requires a formal quality sign-off

#### Test set construction

Select 30 sentences stratified across:

| Category | Count | Rationale |
|:---|:---|:---|
| Standard prose (Harvard Sentences) | 12 | Phonetically balanced baseline |
| Spontaneous-style (shorter, colloquial) | 8 | Conversational agent context |
| Technical (acronyms, numbers) | 6 | Production realism |
| Punctuation-rich (questions, exclamations) | 4 | Prosody stress test |

#### MUSHRA protocol

- **Tool:** WebMUSHRA (`github.com/audiolabs/webMUSHRA`)
- **Scale:** 0–100 continuous
- **Conditions per trial:**
  - Magpie Aria synthesised audio *(test)*
  - Natural human recording of the same sentence *(hidden reference)*
  - Narrowband low-pass filtered speech at 3.5 kHz *(anchor)*
- **Rater pool:** 20–30 native English speakers on Prolific Academic
- **Session length:** max 30 minutes; 30 trials
- **Quality control:** raters who score the hidden reference below 90 are excluded

**Hidden reference sourcing:** LJSpeech only covers read-speech prose — not colloquial, technical, or punctuation-rich prompts. Two options:

1. **Restrict the MUSHRA prompt set to the 12 Harvard Sentences** that have matched LJSpeech recordings, and run a separate MOS (Absolute Category Rating) study for the remaining 18 prompts where no natural reference exists.
2. **Record natural references:** hire a voice actor to read all 30 prompts in a studio environment. This is more expensive but enables a uniform MUSHRA design across all categories.

If neither option is feasible, fall back to an **A/B preference test** (no hidden reference required) for the non-LJSpeech prompts.

#### Analysis

- Mean MUSHRA per condition ± 95% CI
- Wilcoxon signed-rank test for pairwise comparisons (p < 0.05 threshold)
- Rater exclusion rate
- Per-category breakdown (prose / colloquial / technical / punctuation)

#### Acceptance threshold

| MUSHRA Score | Interpretation |
|:---|:---|
| > 80 | Excellent — deploy with confidence |
| 70–80 | Good — acceptable for most use cases |
| 50–70 | Fair — investigate specific failure modes |
| < 50 | Poor — escalate to NVIDIA |

---

## 6. Phase 2 — Voice Agent Evaluation (Outline)

Phase 2 integrates the STT and TTS models into a full voice agent pipeline and evaluates end-to-end behaviour. This section is an outline; detailed test specifications follow Phase 1 completion.

### Architecture under test

```
User Audio → VAD → parakeet (STT) → EOU detection → LLM → Magpie (TTS) → Audio Playback
```

### KPI groups for Phase 2

> Section references below (§) point to the **research doc** `voice_agent_evaluation.md`, not this plan.

| Group | Key Metrics | Research doc reference |
|:---|:---|:---|
| End-to-end latency | e2e_latency P50/P90/P95/P99, mouth-to-ear | §1.3 |
| Turn-taking quality | Turn Gap, Double-Talk Ratio, TPM | §1.4 |
| Barge-in / interruption | Barge-in F1, Reaction Time, False Activation Rate | §1.5 |
| EOU detection | Precision, Recall, False Endpoint Rate | §1.2 |
| Business outcomes | Task Success Rate, Containment Rate | §1.6 |
| Concurrency | P99 < 3× P50, Session Stability > 99.5% | §1.7 |
| Cost | Cost per session, per turn, per minute | §1.8 |

### Phase 2 test types

1. **Single-turn latency test** — synthetic audio → full pipeline → measure e2e_latency
2. **Multi-turn conversation test** — 10-turn scripted dialogues; measure per-turn latency, turn gap, overlap
3. **Barge-in test** — inject interruptions at defined points; measure reaction time and false activation rate
4. **Load test** — ramp concurrency from 1 to target; track P99 latency and session stability
5. **LLM-as-Judge** — sample 100 full sessions; score response relevance and task completion via Claude
6. **Cost attribution** — record token counts, audio durations, and compute cost per session formula

---

## 7. Evaluation Harness Design

### Directory structure

```
nvidia-model-evaluation/
├── docs/
│   ├── voice_agent_evaluation.md   # Research reference
│   └── evaluation_plan.md          # This document
├── eval/
│   ├── config.yaml                 # Endpoints, auth, model names
│   ├── stt/
│   │   ├── client.py               # nvidia-riva-client wrapper (gRPC) + REST fallback
│   │   ├── accuracy.py             # Test 1.1 — WER/CER pipeline
│   │   ├── performance.py          # Test 1.2 — RTF/RTFx measurement
│   │   ├── streaming.py            # Test 1.3 — streaming performance
│   │   ├── noise_robustness.py     # Test 1.5 — noise injection + eval
│   │   ├── accent.py               # Test 1.6 — per-accent WER
│   │   ├── long_form.py            # Test 1.7 — hallucination detection
│   │   ├── domain.py               # Test 1.8 — KRR, OOV
│   │   ├── output_quality.py       # Test 1.9 — punctuation F1
│   │   └── confidence.py           # Test 1.10 — ECE, AUROC
│   ├── tts/
│   │   ├── client.py               # nvidia-riva-client wrapper (gRPC) + REST fallback
│   │   ├── naturalness.py          # Test 2.1 — UTMOS/DNSMOS
│   │   ├── intelligibility.py      # Test 2.2 — round-trip WER
│   │   ├── prosody.py              # Test 2.3 — F0, duration, rate
│   │   ├── signal_quality.py       # Test 2.4 — MCD, PESQ, STOI
│   │   ├── latency.py              # Test 2.5 — TTFB, RTF
│   │   ├── concurrency.py          # Test 2.6 — throughput test
│   │   ├── edge_cases.py           # Test 2.7 — robustness cases
│   │   └── long_form.py            # Test 2.8 — voice consistency
│   ├── data/
│   │   ├── download_datasets.py    # HuggingFace dataset downloads
│   │   ├── harvard_sentences.txt   # 200 Harvard Sentences
│   │   ├── tts_test_sets/          # Category-labelled TTS test cases
│   │   └── noise/                  # MUSAN / DEMAND noise files
│   └── report/
│       ├── generate_report.py      # HTML summary report
│       └── templates/
├── results/
│   ├── stt/
│   └── tts/
└── requirements.txt
```

### `requirements.txt`

```
# NVIDIA Riva gRPC SDK
nvidia-riva-client

# REST / async HTTP
requests
aiohttp

# STT evaluation
jiwer>=3.0
audiomentations
spacy
scikit-learn
datasets
soundfile
pydub

# TTS evaluation
utmos
nisqa
pesq
pystoi
pyworld
dtw-python
librosa
speechbrain

# Reference transcription
openai-whisper

# Forced alignment (install separately: conda/pip + model download)
# montreal-forced-aligner

# Audio / reporting
numpy
scipy
matplotlib
pandas
```

### Config file (`eval/config.yaml`)

```yaml
# nvidia-riva-client gRPC connection
riva:
  grpc_uri: <aicore-grpc-host>:<port>
  use_ssl: true
  auth_token_env: AICORE_BEARER_TOKEN  # read from env var at runtime

stt:
  model_name: parakeet-1.1b-en-US-asr-offline
  rest_endpoint: https://<aicore-host>/v1/deployments/<id>/predictions
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 120

tts:
  model_name: Magpie-Multilingual.EN-US.Aria
  voice_name: Magpie-Multilingual.EN-US.Aria
  rest_endpoint: https://<aicore-host>/v1/deployments/<id>/predictions
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 60
  sample_rate: 22050

evaluation:
  stt_concurrency_levels: [1, 5, 10, 20]
  tts_concurrency_levels: [1, 5, 10, 20, 50]
  bootstrap_n: 1000
  results_dir: results/
  log_level: INFO
```

### Key design principles

- **Idempotency:** All tests write results to structured JSONL files. Interrupted runs can be resumed by checking which item IDs are already in the output.
- **Rate limiting:** Exponential backoff (base 1 s, max 30 s) on HTTP 429 / gRPC RESOURCE_EXHAUSTED.
- **Reproducibility:** Record model name, endpoint URL, auth header key, and request timestamp in every result record.
- **Separation of concerns:** API calls are in `client.py`; metric computation is in dedicated files; reporting is in `report/`. Tests can be re-run without re-calling the API by loading saved responses.

---

## 8. Pass / Fail Thresholds Summary

### STT

| Test | Metric | Pass | Flag | Fail |
|:---|:---|:---|:---|:---|
| 1.1 | WER (LibriSpeech clean) | < 5% | 5–10% | > 10% |
| 1.1 | WER (AMI meetings) | < 20% | 20–30% | > 30% |
| 1.2 | RTFx (batch) | > 10 | 3–10 | < 3 |
| 1.2 | API P99 latency (10 s audio, batch) | < 5 s | 5–15 s | > 15 s |
| 1.3 | TTFW P50 (streaming) | < 600 ms | 600 ms–1 s | > 1 s |
| 1.3 | Finalization Latency P50 | < 300 ms | 300–600 ms | > 600 ms |
| 1.3 | TTFS P50 | < 400 ms | 400–800 ms | > 800 ms |
| 1.3 | Stability Rate | > 80% | 65–80% | < 65% |
| 1.3 | Streaming WER − Batch WER | < +3% | +3–5% | > +5% |
| 1.5 | WER at 10 dB SNR | < 2× clean | 2–3× clean | > 3× clean |
| 1.5 | WER at 0 dB SNR | < 5× clean | 5–8× clean | > 8× clean |
| 1.7 | Hallucination rate | < 1% | 1–3% | > 3% |
| 1.7 | WER full vs chunked | < 1.3× | 1.3–2× | > 2× |
| 1.9 | Punctuation F1 | > 0.80 | 0.60–0.80 | < 0.60 |
| 1.10 | ECE | < 0.05 | 0.05–0.15 | > 0.15 |

### TTS

| Test | Metric | Pass | Flag | Fail |
|:---|:---|:---|:---|:---|
| 2.1 | UTMOS | > 4.0 | 3.5–4.0 | < 3.5 |
| 2.1 | DNSMOS OVRL | > 3.5 | 3.0–3.5 | < 3.0 |
| 2.2 | Round-trip WER (prose) | < 2% | 2–5% | > 5% |
| 2.3 | F0 RMSE | < 20 Hz | 20–40 Hz | > 40 Hz |
| 2.3 | Speaking rate | 130–180 WPM | 110–130 / 180–210 WPM | < 110 or > 210 WPM |
| 2.4 | MCD | < 6 dB | 6–8 dB | > 8 dB |
| 2.4 | PESQ | > 2.5 | 1.5–2.5 | < 1.5 |
| 2.5 | TTFB P50 (gRPC, short) | < 150 ms | 150–300 ms | > 300 ms |
| 2.5 | TTFB P99 (gRPC, short) | < 500 ms | 500 ms–1 s | > 1 s |
| 2.5 | RTF | < 0.5 | 0.5–1.0 | > 1.0 |
| 2.8 | Intra-passage SpeakerSim | > 0.90 | 0.80–0.90 | < 0.80 |
| 2.9 | MUSHRA | > 70 | 50–70 | < 50 |

---

## 9. Deliverables & Suggested Timeline

| Week | Milestone | Deliverables |
|:---|:---|:---|
| 1 | Phase 0 complete | Connectivity confirmed; schema discovery notes; smoke test results; `eval/config.yaml` |
| 1–2 | Test harness scaffolding | `eval/` directory with client wrappers, data download scripts, JSONL output structure |
| 2 | Dataset downloads | LibriSpeech, TED-LIUM, GigaSpeech, Common Voice EN, Earnings-22, MUSAN, DEMAND all cached locally |
| 3 | STT Tests 1.1–1.4 | Clean accuracy results, RTF/RTFx + streaming latency profile, REST vs gRPC comparison |
| 3–4 | STT Tests 1.5–1.7 | Noise robustness curves, accent WER bar charts, long-form analysis |
| 4 | STT Tests 1.8–1.11 | Domain/KRR analysis, output quality F1, confidence calibration, format support matrix |
| 4 | **STT interim report** | HTML summary: all STT metrics, charts, pass/fail table |
| 5 | TTS Tests 2.1–2.4 | UTMOS/DNSMOS scores, round-trip WER table, prosody report, MCD/PESQ/STOI table |
| 5–6 | TTS Tests 2.5–2.8 | TTFB/RTF latency profile, concurrency test, edge-case results, long-form consistency report |
| 6 | **TTS interim report** | HTML summary: all TTS metrics, charts, pass/fail table |
| 7 | Test 2.9 (MUSHRA) | MUSHRA study design, rater recruitment, WebMUSHRA deployment |
| 8 | MUSHRA results | Scores, CI, Wilcoxon tests, final perceptual report |
| 8 | **Phase 1 final report** | Combined STT + TTS evaluation report with executive summary, all findings, recommendations |
| 9+ | Phase 2 scoping | Voice agent pipeline setup, Phase 2 detailed test plan |

### Final report structure

```
1. Executive Summary
   - Pass/fail status per model
   - Top 3 strengths, top 3 risks per model
   - Recommendation for production use

2. STT Findings
   - Accuracy: WER/CER tables with CI across all datasets
   - Batch performance: RTF/RTFx, latency percentiles
   - Streaming performance: TTFW, Finalization Latency, TTFS, Stability Rate, streaming vs batch WER delta
   - Noise robustness curves
   - Accent fairness audit
   - Long-form behaviour analysis
   - Output quality assessment
   - Confidence calibration analysis

3. TTS Findings
   - Automated quality: UTMOS, DNSMOS, NISQA
   - Intelligibility: round-trip WER by category
   - Prosody report: F0, duration, rate, nPVI
   - Signal quality: MCD, PESQ, STOI
   - Latency profile: TTFB distribution, RTF
   - Concurrency and throughput
   - Edge case inventory
   - Long-form consistency analysis
   - Human evaluation (MUSHRA) results

4. REST vs gRPC Interface Comparison

5. Known Limitations & Open Questions

6. Recommendations for Phase 2 (Voice Agent)

Appendix A — Raw data tables
Appendix B — Audio samples (linked)
Appendix C — Dataset provenance and licences
```
