# NVIDIA Riva Model Evaluation

Automated evaluation harness for two NVIDIA speech AI models deployed on **SAP AI Core**:

| Model | Type | Model ID |
|-------|------|----------|
| Parakeet | Speech-to-Text (STT) | `parakeet-1.1b-en-US-asr-offline` |
| Magpie Aria | Text-to-Speech (TTS) | `Magpie-Multilingual.EN-US.Aria` |

Both REST and gRPC interfaces are tested. All measurements are made **client-side** — no access to server internals or Triton metrics is assumed.

---

## What It Tests

### Phase 0 — Discovery
Connectivity checks, schema discovery, smoke tests, and cold-start measurement against all four endpoints (gRPC/REST × STT/TTS).

### STT Evaluation — 11 Tests

| Test | What it measures |
|------|-----------------|
| 1.1 Clean Speech Accuracy | WER / CER / MER across 7 public benchmarks (LibriSpeech, TED-LIUM, GigaSpeech, SPGISpeech, Earnings-22, AMI, Common Voice) |
| 1.2 Batch Performance | RTF / RTFx across 5 audio duration buckets; concurrency at N=1,5,10,20 |
| 1.3 Streaming Performance | TTFW, partial result latency, finalization latency, streaming RTF, stability rate |
| 1.4 REST vs gRPC | Side-by-side P50/P90/P99 latency comparison on identical audio |
| 1.5 Noise Robustness | WER degradation curves across 4 noise types × 6 SNR levels (−5 dB to +30 dB) |
| 1.6 Accent Robustness | Per-accent WER on Common Voice EN + L2-ARCTIC; fairness delta |
| 1.7 Long-Form Audio | Full-file vs 30 s chunked WER; hallucination and repetition detection |
| 1.8 Domain Vocabulary | Key-term recall rate (KRR), NE-WER, OOV rate on SPGISpeech + Earnings-22 |
| 1.9 Output Quality | Punctuation F1 per mark, capitalisation accuracy, number formatting accuracy |
| 1.10 Confidence Calibration | ECE, AUROC, Brier score; reliability diagram |
| 1.11 Format Robustness | Pass/fail matrix across audio formats, sample rates, bit depths |

### TTS Evaluation — 8 Tests

| Test | What it measures |
|------|-----------------|
| 2.1 Naturalness | UTMOS and DNSMOS (OVRL/SIG/BAK) on 200 synthesized sentences |
| 2.2 Intelligibility | Round-trip WER: synthesize → Whisper large-v3 → jiwer, across 4 sentence categories |
| 2.3 Prosody | F0 RMSE, F0 Pearson r, speaking rate (WPM), nPVI against LJSpeech reference |
| 2.4 Signal Quality | MCD, PESQ, STOI (noted: speaker mismatch caveat applies) |
| 2.5 Streaming TTFB & RTF | Time-to-first-byte and real-time factor across 5 text length buckets × 2 interfaces |
| 2.6 Throughput & Concurrency | RPS, P50/P99, error rate at N=1,5,10,20,50 concurrent requests |
| 2.7 Edge Cases | ~100 test cases across 17 categories: empty input, numbers, SSML, Unicode, very long text, etc. |
| 2.8 Long-Form Consistency | ECAPA-TDNN speaker similarity, F0 drift, speaking rate drift across multi-paragraph passages |

---

## Project Structure

```
nvidia-model-evaluation/
├── eval/
│   ├── config.yaml           # Endpoints, auth, model names — edit this before running
│   ├── config.py             # Typed config dataclasses
│   ├── utils.py              # Shared: WER normalization, JSONL I/O, stats, retry
│   ├── run.py                # CLI entry point
│   ├── phase0/
│   │   └── discovery.py      # Phase 0 discovery + smoke tests
│   ├── stt/
│   │   ├── client.py         # STTClient: gRPC + REST, batch + streaming
│   │   └── accuracy.py … format_robustness.py   # Tests 1.1–1.11
│   ├── tts/
│   │   ├── client.py         # TTSClient: gRPC + REST, batch + streaming
│   │   └── naturalness.py … long_form.py        # Tests 2.1–2.8
│   ├── data/
│   │   ├── tts_test_sets.py  # All TTS test sentences, edge cases, passages
│   │   ├── harvard_sentences.txt
│   │   └── download_datasets.py  # HuggingFace dataset downloader
│   └── report/
│       └── generate_report.py    # HTML report with pass/fail badges
├── results/                  # All output written here (created on first run)
├── docs/
│   ├── evaluation_plan.md    # Full test specifications and pass/fail thresholds
│   └── implementation_plan.md
├── requirements.txt
└── RUNBOOK.md                # Step-by-step setup and run guide for the Azure VM
```

---

## Quick Start

**Run this on the Azure VM** (Germany West Central), not a laptop. The VM is co-located with AI Core (Frankfurt / eu-central-1), which keeps network jitter out of latency measurements. See `RUNBOOK.md` for the full setup guide.

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Fill in eval/config.yaml with your AI Core endpoints and deployment IDs

# 3. Export your AI Core bearer token
export AICORE_BEARER_TOKEN="eyJ..."

# 4. Verify connectivity
python -m eval.run phase0

# 5. Run everything (~7 hours on an A100)
python -m eval.run all

# 6. Open the report
open results/report.html
```

Run a single test:
```bash
python -m eval.run stt --test accuracy
python -m eval.run tts --test naturalness
```

Validate config without making API calls:
```bash
python -m eval.run all --dry-run
```

---

## Outputs

Every test writes results immediately to `results/<suite>/<test>/`:
- `calls.jsonl` — one record per API call (used for idempotent resume)
- `summary.csv` — aggregated metrics
- `*.wav` — synthesized audio files (TTS tests)

After all tests complete, `results/report.html` contains a full HTML report with per-test tables and pass/fail badges derived from the thresholds in `docs/evaluation_plan.md`.

---

## Key Design Decisions

**Idempotent resume** — every API call is written to JSONL immediately. Re-running a test skips already-completed items, so a crashed run picks up where it left off.

**Both interfaces** — every test that involves latency runs on both gRPC and REST so the two can be compared directly.

**Client-side only** — all metrics are measured in the harness. No server access, no Triton metrics, no GPU counters required.

**Local GPU for evaluation tools** — the models themselves run on AI Core. The VM GPU is used only for local evaluation tools: Whisper large-v3 (round-trip WER), UTMOS/DNSMOS (naturalness scoring), and SpeechBrain ECAPA-TDNN (speaker similarity).

---

## Dependencies

Key packages (see `requirements.txt` for the full list):

| Package | Used for |
|---------|----------|
| `nvidia-riva-client` | gRPC calls to AI Core STT/TTS |
| `openai-whisper` | Round-trip WER transcription (TTS 2.2, 2.7) |
| `jiwer` | WER / CER / MER computation |
| `speechbrain` | ECAPA-TDNN speaker embeddings (TTS 2.8) |
| `pyworld` | F0 extraction (TTS 2.3, 2.8) |
| `utmos` | Neural MOS estimation (TTS 2.1, 2.7) |
| `librosa` | Audio loading, MFCC, duration |
| `audiomentations` | Noise injection for STT 1.5 |
| `spacy` | NER for key-term recall (STT 1.8) |
| `datasets` | HuggingFace dataset loading |
| `aiohttp` | Async REST concurrency (TTS 2.6) |

---

## Documentation

- `docs/evaluation_plan.md` — full test specifications, metrics definitions, and pass/fail thresholds
- `RUNBOOK.md` — VM setup, configuration, and step-by-step run instructions
