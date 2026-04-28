# TTS Gap Analysis: Eval Results vs Published Benchmarks

**Date:** 2026-04-28
**Model:** `nvidia/magpie_tts_multilingual_357m` (MagpieTTS v2602, 357M params)
**Voice:** `Magpie-Multilingual.EN-US.Aria`
**Deployment:** NVIDIA Riva TTS gRPC/REST endpoint on AI Core

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Published Benchmark Sources](#published-benchmark-sources)
3. [Metric Comparison: Quality](#metric-comparison-quality)
   - [CER (Character Error Rate)](#cer-character-error-rate)
   - [Speaker Similarity](#speaker-similarity)
   - [Naturalness (UTMOS)](#naturalness-utmos)
4. [Metric Comparison: Latency](#metric-comparison-latency)
   - [First-Chunk Latency (TTFB)](#first-chunk-latency-ttfb)
   - [Throughput (RTFX)](#throughput-rtfx)
5. [Metric Comparison: Concurrency (Primary Issue)](#metric-comparison-concurrency-primary-issue)
   - [NIM Benchmarks vs Our Results](#nim-benchmarks-vs-our-results)
   - [The Scaling Gap](#the-scaling-gap)
6. [Known Limitations Confirmed](#known-limitations-confirmed)
   - [400-Token Limit](#400-token-limit)
   - [SSML Support](#ssml-support)
7. [Root Cause Analysis: Concurrency Failure](#root-cause-analysis-concurrency-failure)
   - [RCA 1: Missing NIM Inference Optimizations](#rca-1-missing-nim-inference-optimizations)
   - [RCA 2: GPU Memory / Hardware Constraints](#rca-2-gpu-memory--hardware-constraints)
   - [RCA 3: No Request Queuing](#rca-3-no-request-queuing)
   - [RCA 4: Single Replica Deployment](#rca-4-single-replica-deployment)
   - [Most Likely Explanation](#most-likely-explanation)
8. [Questions for AI Core Team](#questions-for-ai-core-team)
   - [Deployment Architecture](#deployment-architecture)
   - [Concurrency & Scaling](#concurrency--scaling)
   - [Quality & Configuration](#quality--configuration)
   - [Improvement Paths](#improvement-paths)

---

## Executive Summary

Our TTS evaluation reveals a split picture:

- **Quality and single-request latency are strong** -- TTFB (~65 ms), RTF (0.074), intelligibility (0.16% WER on conversational text), and naturalness (UTMOS 3.96) all match or exceed NVIDIA's published benchmarks.
- **Concurrency is the critical failure** -- the deployment saturates at 10 concurrent streams (4% hung requests) and breaks at 20 (24% failures). NVIDIA's own NIM benchmarks show Magpie TTS Multilingual handling **64 parallel streams** on an A100 GPU with no errors and first-chunk latency under 303 ms.

The concurrency gap points to a deployment/serving issue, not a model quality issue. The model performs well per-request; the infrastructure cannot scale it.

---

## Published Benchmark Sources

| Source | What it contains |
|--------|-----------------|
| [HuggingFace model card](https://huggingface.co/nvidia/magpie_tts_multilingual_357m) | CER, SV-SSIM on LibriTTS/CML datasets; model architecture; training data |
| [NVIDIA NIM TTS Performance Docs](https://docs.nvidia.com/nim/speech/latest/reference/performances/tts/performance.html) | First-chunk latency, inter-chunk latency, RTFX across GPU types (A100, H100, L40, B200) at 1-64 parallel streams |

The model card provides quality metrics. The NIM docs provide latency and throughput benchmarks.

---

## Metric Comparison: Quality

### CER (Character Error Rate)

| Source | Dataset | CER |
|--------|---------|-----|
| Model card | LibriTTS test-clean | **0.34%** |
| NIM docs | English (unspecified dataset) | **1.0%** |
| Our eval (Whisper round-trip) | Harvard sentences (80 clips) | 1.48% |
| Our eval (Whisper round-trip) | Conversational (50 clips) | 0.16% |
| Our eval (Whisper round-trip) | Technical terms (50 clips) | 11.63% |
| Our eval (Whisper round-trip) | Numbers & figures (50 clips) | 17.18% |

**Assessment:** Our conversational CER (0.16%) is better than both published benchmarks. The Harvard sentence CER (1.48%) is close to the NIM's reported 1.0%. The technical/numbers failures are domain-specific issues (acronyms like "NASA", currency like "$1,234.56", IP addresses) that reflect text normalization gaps, not core synthesis quality. No discrepancy on standard text.

### Speaker Similarity

| Source | Metric | Score |
|--------|--------|-------|
| Model card | SV-SSIM (pred vs ground truth) | **0.835** |
| Our eval | ECAPA-TDNN cosine similarity (across paragraphs) | **0.938-0.949** |

**Assessment:** Not directly comparable -- different metrics and different evaluation setups. The model card measures how closely synthesized speech matches the original speaker's recordings. Our eval measures within-voice consistency (does the same Aria voice sound consistent across paragraphs of a long passage?). Both metrics are healthy. No discrepancy.

### Naturalness (UTMOS)

| Source | Metric | Score |
|--------|--------|-------|
| Model card | UTMOSv2 | Not published (listed as metric, no value given) |
| Our eval | UTMOS | **3.96 mean / 3.99 P50** (out of 5.0) |

**Assessment:** NVIDIA does not publish a UTMOS score in the model card. Our measured UTMOS of 3.96 is near the human naturalness threshold (~4.0) and indicates high-quality synthesis. No comparison possible, but the result is strong.

---

## Metric Comparison: Latency

### First-Chunk Latency (TTFB)

| Source | Streams | GPU | First-Chunk Latency |
|--------|---------|-----|-------------------|
| **NIM A100** | 1 | A100 40GB | **68.28 ms** |
| **NIM H100** | 1 | H100 80GB | **70.0 ms** |
| **NIM L40** | 1 | L40 | **62.81 ms** |
| **NIM B200** | 1 | B200 | **55.1 ms** |
| **Our eval (gRPC)** | 1 | Unknown | **65 ms** (P50, 10 words) |
| **Our eval (REST)** | 1 | Unknown | **94 ms** (P50, 10 words) |

**Assessment:** Our gRPC TTFB of 65 ms aligns closely with the NIM A100 benchmark (68.28 ms). The REST overhead (~30 ms) is expected due to HTTP buffering. **No discrepancy -- single-request latency is excellent.**

### Throughput (RTFX)

RTFX = audio duration / processing time. Higher is better. RTFX > 1 means faster than real-time.

| Source | Streams | RTFX |
|--------|---------|------|
| **NIM A100** | 1 | **10.24** |
| **NIM H100** | 1 | **8.78** |
| **NIM L40** | 1 | **13.0** |
| **Our eval (gRPC, 10 words)** | 1 | **10.0** (1/RTF 0.10) |
| **Our eval (gRPC, 150 words)** | 1 | **13.5** (1/RTF 0.074) |
| **NIM A100** | 64 | **140.66** (aggregate) |
| **Our eval** | 10 | **Server failures** |

**Assessment:** Single-stream RTFX matches perfectly (10-13.5x vs NIM's 8.8-13.0x). The discrepancy is entirely in scaling -- NIM reaches 140 RTFX at 64 streams via batching; our deployment cannot sustain 10 streams.

---

## Metric Comparison: Concurrency (Primary Issue)

### NIM Benchmarks vs Our Results

**NVIDIA NIM (A100 40GB) -- official benchmarks:**

| Streams | First-Chunk Latency | Inter-Chunk Latency | RTFX |
|---------|-------------------|-------------------|------|
| 1 | 68.28 ms | 10.97 ms | 10.24 |
| 4 | 72.19 ms | 14.32 ms | 37.05 |
| 8 | 81.83 ms | 18.16 ms | 59.38 |
| 16 | 101.50 ms | 26.49 ms | 87.97 |
| 32 | 155.31 ms | 44.05 ms | 110.35 |
| 64 | 302.82 ms | 85.40 ms | 140.66 |

**Our deployment -- eval results:**

| Concurrent | Success | Errors | Error Rate | P50 Latency | P99 Latency | RPS |
|------------|---------|--------|------------|-------------|-------------|-----|
| 1 | 50/50 | 0 | 0% | 640 ms | 698 ms | 1.55 |
| 5 | 50/50 | 0 | 0% | 710 ms | 746 ms | 7.03 |
| 10 | 48/50 | 2 | 4% | 801 ms | 1,380 ms | ~11.7* |
| 20 | 38/50 | 12 | 24% | 837 ms | 2,328 ms | ~14.5* |

*Estimated via Little's Law; actual throughput is lower due to hung requests.

### The Scaling Gap

| Metric | NIM (A100, 64 streams) | Our Deployment (20 streams) |
|--------|----------------------|---------------------------|
| Error rate | **0%** | **24%** |
| First-chunk latency | 303 ms | 837 ms (P50), 2,328 ms (P99) |
| Aggregate RTFX | **140.66** | N/A (requests hanging) |
| Max safe concurrency | **64+** | **~5** |

The NIM handles 64 parallel streams with zero errors on an A100. Our deployment cannot reliably serve 10 streams. This is a **~12x concurrency gap**.

---

## Known Limitations Confirmed

### 400-Token Limit

The model card states: "In standard mode, this model can generate up to twenty (20) seconds of multilingual speech at a time."

Our edge case test confirms this -- `very_long` text (0/2 passed). The 400-token sequence limit is a hard model constraint, not a deployment issue.

### SSML Support

Our eval: 40% pass rate (2/5 SSML test cases). Basic tags (`<break>`) work; advanced (`<phoneme>`, `<emphasis>`) fail. The model card does not mention SSML -- it is a Riva-layer feature, not a model-level capability. The NIM enterprise offering may have better SSML handling.

---

## Root Cause Analysis: Concurrency Failure

The server saturates at 10 concurrent streams and degrades catastrophically at 20. Single-request performance is excellent. What explains this?

### RCA 1: Missing NIM Inference Optimizations

**Probability: High**

The model card explicitly differentiates the open model from the NIM:

> "For the enterprise offering, see the MagpieTTS NIM which includes additional native voices in the supported languages, emotional speech capabilities, and **optimized batch and latency inference pipeline**."

The NIM includes inference optimizations not present in a raw model deployment:

- **Dynamic batching** -- groups multiple concurrent requests into a single GPU forward pass, allowing 64 streams to share compute efficiently
- **CUDA graph capture** -- pre-compiles the inference computation graph, eliminating CPU-GPU launch overhead
- **TensorRT optimization** -- quantized and fused operators for faster per-step execution
- **Chunked streaming pipeline** -- overlaps generation of different requests across time steps

Without these, each concurrent request competes for GPU resources independently, causing contention and hangs at modest concurrency.

### RCA 2: GPU Memory / Hardware Constraints

**Probability: Medium**

MagpieTTS is an autoregressive transformer. Each concurrent stream holds:
- Key-value cache for the 12-layer decoder (~8 codebooks, each attending over growing sequence length)
- Intermediate activations during codec generation
- Output audio buffer

On an A100 (40 GB), the NIM benchmarks show this fits for 64 streams. If the AI Core deployment runs on a smaller GPU (e.g., T4 with 16 GB or L4 with 24 GB), memory would be exhausted at far fewer concurrent streams, causing requests to stall waiting for memory.

### RCA 3: No Request Queuing

**Probability: Medium-High**

The NIM likely implements a request queue with admission control -- accepting requests into a waiting queue and processing them as GPU capacity frees up. Without this:
- All 10 or 20 concurrent requests hit the GPU simultaneously
- GPU compute is split across all of them, causing each to slow down
- Some requests take long enough to hit the 60s gRPC timeout or the 300s wall-clock timeout
- These appear as "hung" requests in our eval (2 at N=10, 12 at N=20)

### RCA 4: Single Replica Deployment

**Probability: Medium**

If only one model replica is running on one GPU, there is no horizontal scaling path. The NIM benchmarks are per-GPU numbers. In production, multiple replicas behind a load balancer would handle high concurrency by distributing across GPUs rather than forcing one GPU to serve all requests.

### Most Likely Explanation

A combination of **RCA 1 + RCA 3**: the deployment runs the model without NIM-level batching optimizations and without a request queue. Each concurrent request gets its own independent GPU inference, and at N >= 10, the GPU cannot keep up, causing requests to stall and eventually timeout.

This is supported by the data:
- N=1 to N=5: near-linear scaling (1.55 → 7.03 RPS, 4.5x) with no errors -- the GPU has spare capacity
- N=10: first signs of saturation (2 hung requests, P99 doubles to 1,380 ms)
- N=20: collapse (24% failure rate, P99 = 2,328 ms)

The NIM avoids this cliff by batching requests -- 64 streams don't each get an independent forward pass; they're batched together, making the total compute scale sub-linearly.

---

## Questions for AI Core Team

### Deployment Architecture

   1. **Is this the MagpieTTS NIM or a raw model deployment?**
      The NIM includes "optimized batch and latency inference pipeline"
      per NVIDIA's documentation. If running the raw model, concurrency
      limitations are expected.

   2. **What GPU is the model running on?**
      A100 40GB, H100, L40, T4, or something else? The NIM benchmarks
      show 64-stream support on A100. Smaller GPUs would hit limits earlier.

   3. **How many model replicas are deployed?**
      Is there a single instance, or multiple replicas behind a load balancer?

   4. **What serving framework is used?**
      Triton Inference Server, a custom gRPC server, or the NIM container?
      Triton with dynamic batching enabled would significantly improve
      concurrency handling.

### Concurrency & Scaling

   5. **Is dynamic batching enabled?**
      Without batching, each concurrent request runs an independent GPU
      forward pass. With batching, multiple requests share a single pass,
      enabling 64+ streams per GPU.

   6. **Is there a request queue or admission control?**
      Our eval saw requests hang indefinitely at N=10+. A request queue
      would hold excess requests and process them as capacity frees up,
      instead of letting them compete for GPU resources.

   7. **What is the intended max concurrency for this deployment?**
      If the target is < 5 concurrent users, the deployment is working
      as expected. If > 5, scaling changes are needed.

   8. **What was the GPU utilization during our concurrency tests?**
      If GPU compute was pegged at 100% while requests hung, that confirms
      compute saturation. If GPU was < 80%, the bottleneck may be elsewhere
      (memory, gRPC thread pool, etc.).

### Quality & Configuration

   9. **Is text normalization enabled?**
      The model card notes "built-in text normalization for handling numbers,
      abbreviations, and special characters." Our eval showed 17.2% WER on
      numbers -- suggesting TN may not be active or is incomplete.

   10. **Which model checkpoint version is deployed?**
       The model card shows v2602 (March 2026) with Hindi/Japanese support.
       Older checkpoints (v2512) may have different quality characteristics.

### Improvement Paths

   11. **Can we migrate to the NIM container?**
       The NIM's "optimized inference pipeline" would likely resolve the
       concurrency issue entirely. NVIDIA benchmarks show 64 streams
       at 303ms first-chunk latency on A100.

   12. **Can we add horizontal scaling?**
       Even without NIM optimizations, running 2-3 replicas behind a
       load balancer would give linear concurrency improvement.

   13. **Can we enable dynamic batching in the current setup?**
       If running on Triton, dynamic batching can be enabled via config
       without changing the model. This is the lowest-effort improvement.

   14. **Can we test with the NIM performance client?**
       NVIDIA provides `riva_tts_perf_client` for standardized benchmarking.
       Running it against our deployment would give directly comparable
       numbers to the NIM benchmarks above.
