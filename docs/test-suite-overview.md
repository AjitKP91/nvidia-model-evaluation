# Speech Model Evaluation — Test Suite Overview

An automated harness that evaluates **Speech-to-Text (STT)** and **Text-to-Speech (TTS)**
models. Both REST and gRPC interfaces are tested, and **all measurements are
client-side** — no access to server internals or GPU metrics is assumed.

## Phase 0 — Discovery
Connectivity checks, API schema discovery, smoke tests, and cold-start latency
measurement across all endpoints.

## STT Evaluation — 13 Tests

| # | Test | What it measures |
|---|------|-----------------|
| 1.1 | Clean Speech Accuracy | WER / CER / MER / WIL across 8 Open-ASR-Leaderboard benchmarks (LibriSpeech clean + other, VoxPopuli, GigaSpeech, SPGISpeech, TED-LIUM 3, AMI-IHM, Earnings-22); reports both normalized and raw cased+punctuated WER plus a sub/ins/del breakdown. Common Voice is the 9th leaderboard set but left the HF Hub (Oct 2025) and is skipped unless a local copy is supplied. |
| 1.2 | Batch Performance | Real-time factor (RTF/RTFx) across 5 audio-duration buckets; concurrency at N = 1, 5, 10, 20 |
| 1.3 | Streaming Performance | Time-to-first-word, partial-result latency, finalization latency, streaming RTF, stability rate |
| 1.4 | REST vs gRPC | Side-by-side P50/P90/P99 latency on identical audio |
| 1.5 | Noise Robustness | WER degradation across **4 noise types — Gaussian, babble, traffic, office** — over SNR levels from **−5 dB to +20 dB** (Gaussian: −5, 0, 5, 10, 15, 20 dB; babble/traffic/office: 0–20 dB) |
| 1.6 | Accent Robustness | Per-accent WER (VCTK + L2-ARCTIC); fairness delta across accents |
| 1.7 | Long-Form Audio | Full-file vs 30 s chunked WER; hallucination and repetition detection |
| 1.8 | Domain Vocabulary | Key-term recall rate, named-entity WER, out-of-vocabulary rate (financial/domain corpora) |
| 1.9 | Output Quality | Punctuation F1 per mark, capitalization accuracy, number-formatting accuracy |
| 1.10 | Confidence Calibration | Expected calibration error (ECE), AUROC, Brier score; reliability diagram |
| 1.11 | Format Robustness | Pass/fail matrix across audio formats, sample rates, and bit depths |
| 1.12 | Multilingual Accuracy | Per-language WER (space-delimited) or CER (CJK) on FLEURS across the 16 SAP Joule languages (13 GA + 3 preview), with unweighted macro-averages and bootstrap CIs |
| 1.13 | Language Acceptance | Phase-0 smoke probe: streams ~3 FLEURS clips for the Nemotron-3.5 streaming model's supported languages and records ok / empty / error / load_failed per language, so the operator sees which languages actually produce output on this endpoint before the full evaluation. Idempotent (ok languages skip on re-run); also runs automatically inside Phase 0. |

**Text normalization (1.1 / 1.12):** before scoring, both reference and hypothesis
pass through the same normalizer used by the Open ASR Leaderboard and NVIDIA's
model cards — Whisper's `EnglishTextNormalizer` for English (digit↔word,
contractions, currency) and `BasicTextNormalizer` for every other language
(case/punctuation stripped, diacritics preserved) — so our WER/CER numbers are
directly comparable to published figures.

**Multilingual (1.12):** FLEURS (`google/fleurs`, CC-BY-4.0) provides one
consistent read-speech corpus covering all Joule languages. The model under test
(nemotron-3.5-asr-streaming-0.6b) auto-detects language, so the per-request
language code is only a hint; CJK is scored by CER since it lacks word
boundaries. Set `stt.multilingual_languages` in `config.yaml` to restrict the run.
For CER languages, all whitespace is stripped before scoring (matching Open ASR
Leaderboard practice for Chinese/Japanese) because this endpoint emits one space
per character (`イ ン タ ー ネ ッ ト`); counting those spaces would otherwise inflate
CER toward 1.0.

**Noise sources (1.5):** Gaussian noise is generated synthetically; babble, traffic,
and office noise are injected from real recordings (MUSAN for babble; DEMAND STREET and
OFFICE recordings for traffic and office). Each condition is measured at a fixed SNR and
compared against a clean baseline to produce a relative WER-degradation curve.

**Opt-in per-language depth (1.2, 1.5, 1.7, 1.9, 1.10):** several STT tests can run beyond
English to measure the parts of model behaviour that vary by language. This is opt-in and
defaults off — with the config lists empty, these tests run their original English path
byte-for-byte unchanged. Two `config.yaml` keys under `stt:` enable it:
- `multilingual_extra_languages` — adds **Tier-A** per-language passes for **1.5** (noise, same
  SNR grid on FLEURS audio, CER-scored for CJK/Thai) and **1.9** (output quality on FLEURS
  `raw_transcription` with language-aware punctuation/caps/number rules).
- `spot_check_languages` — adds **Tier-B** spot-checks for **1.2** (RTF/RTFx + output
  chars/tokens per audio-second), **1.7** (long-form on concatenated FLEURS clips), and **1.10**
  (confidence calibration). Representative, not exhaustive.
Per-language results are written to `results/stt/<test>/<langkey>.jsonl` and tagged with a
`language` field. If the endpoint ever returned blank for a language, that language would record
a graceful `no_output` row rather than a fabricated score — though as of the latest run every
in-scope language (Arabic included) transcribes; see
`docs/multilingual-test-gap-analysis.md`. Tests **1.3, 1.4, 1.6, 1.8, 1.11** remain English-only
by design: 1.3/1.4/1.11 measure serving-stack / audio-frontend properties, while 1.6 (dialect)
and 1.8 (domain) are **flagged coverage gaps** that need corpora FLEURS cannot provide and are
deliberately not faked.

## TTS Evaluation — 9 Tests

| # | Test | What it measures |
|---|------|-----------------|
| 2.1 | Naturalness | UTMOS and DNSMOS (OVRL/SIG/BAK) on 200 synthesized sentences. Per-language: DNSMOS for every language, UTMOS English-only (see multilingual note). |
| 2.2 | Intelligibility | Round-trip WER/CER: synthesize → transcribe with Whisper large-v3 → compare, across 4 sentence categories (English) and FLEURS text per language (12 languages; CER for zh/ja, WER otherwise). |
| 2.3 | Prosody | **Reference-free**: intra-utterance F0 std/range (pitch dynamism), acoustic nPVI (rhythm), pause ratio, and speaking rate (WPM). Flags monotone/robotic output. Intrusive F0-vs-reference metrics are not computable for a fixed synthetic voice. |
| 2.4 | Signal Quality | **N/A for this model.** MCD/PESQ/STOI are full-reference metrics needing same-speaker ground truth, which does not exist for a fixed synthetic voice. Reported as N/A, not faked. Use 2.1 + 2.2 for quality. |
| 2.5 | Streaming Latency | Time-to-first-byte and RTF across 5 text-length buckets × 2 interfaces |
| 2.6 | Throughput & Concurrency | Requests/sec, P50/P99 latency, error rate at N = 1, 5, 10, 20 concurrent |
| 2.7 | Edge Cases | ~100 cases across **17 categories** (see below) |
| 2.8 | Long-Form Consistency | Speaker-similarity drift (ECAPA-TDNN), F0 drift, and speaking-rate drift across multi-paragraph passages |
| 2.9 | Language Acceptance | Probe (TTS analogue of 1.13): synthesize 2–3 FLEURS sentences per configured language with its voice+locale → transcribe with Whisper (auto-detect) → record ok / wrong_language / empty / error. The honest gate: per-language 2.1/2.2/2.3 only score languages that pass. Idempotent. |

**Edge-case categories (2.7):** empty/whitespace, single word, very long text,
numbers, punctuation, abbreviations, proper nouns, SSML markup, special characters,
repetition, mixed case, technical/domain terms, questions & commands, lists,
URLs & emails, code snippets, and boundary lengths.

**Multilingual TTS (2.1 / 2.2 / 2.3 / 2.9):** the model under test (Magpie TTS
Multilingual) serves **12** languages on this deployment — **en, es, fr, de, it,
pt, hi, ko, vi, zh, ja, ar** — each with its own voice (confirmed live via
`GetRivaSynthesisConfig`; re-probe with `scripts/probe_tts_voices.py --synth`).
Opt-in via `tts.multilingual_languages` in `config.yaml` (keys from those 12,
defaulting to `[]` = English-only, original behaviour byte-for-byte). Intelligibility
is scored with WER for the space-delimited languages and **CER for Chinese/Japanese**
(no word boundaries). Honest gates: UTMOS is **English-only** (it was validated
on English with Chinese as out-of-domain, so it is not a valid naturalness score for
non-English — DNSMOS, which is language-agnostic, is reported for all); any language the
deployment does not actually synthesize is recorded as a `wrong_language`/`empty`
serving gap by 2.9 rather than scored. 2.4 is N/A for the reasons above. Run **2.9
first** to gate which languages round-trip before scoring 2.1/2.2/2.3.

## Key Design Principles
- **Both interfaces** — every latency-sensitive test runs on both gRPC and REST for direct comparison.
- **Client-side only** — all metrics measured in the harness; no server access required.
- **Idempotent resume** — every API call is logged immediately, so an interrupted run resumes where it stopped.
- **Local evaluation models** — a local GPU runs the scoring tools (Whisper for round-trip WER, UTMOS/DNSMOS for naturalness, ECAPA-TDNN for speaker similarity); the models under test run remotely.
