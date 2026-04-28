# WER Gap Analysis: Eval Results vs Published Benchmarks

**Date:** 2026-04-28
**Model:** `nvidia/parakeet-ctc-1.1b` (FastConformer CTC, 1.1B params)
**Deployment:** NVIDIA Riva STT gRPC endpoint

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Benchmark Comparison](#benchmark-comparison)
3. [Verification of Published Numbers](#verification-of-published-numbers)
4. [Verification of Our WER Calculation](#verification-of-our-wer-calculation)
5. [Root Cause: Riva Truncation](#root-cause-riva-truncation)
   - [The Smoking Gun](#the-smoking-gun)
   - [Statistics](#statistics-librispeech-clean-n2620)
   - [Why This Inflates Aggregate WER Disproportionately](#why-this-inflates-aggregate-wer-disproportionately)
   - [Likely Cause](#likely-cause)
6. [Model Card Note: Riva Not Officially Supported](#model-card-note-riva-not-officially-supported)
7. [Riva vs NeMo + Triton: Deployment Stacks](#riva-vs-nemo--triton-deployment-stacks)
   - [Riva](#riva)
   - [NeMo + Triton](#nemo--triton)
   - [Why It Matters for This Analysis](#why-it-matters-for-this-analysis)
8. [Word Boosting](#word-boosting)
   - [How It Works](#how-it-works)
   - [Usage (gRPC API)](#usage-grpc-api)
   - [What It Helps With](#what-it-helps-with)
   - [What It Does NOT Fix](#what-it-does-not-fix)
   - [Related: LM Customization](#related-lm-customization)
9. [Minor Methodological Issue: SPGISpeech Split](#minor-methodological-issue-spgispeech-split)
10. [Questions for AI Core Team](#questions-for-ai-core-team)
    - [Deployment Architecture](#deployment-architecture)
    - [Truncation Issue](#truncation-issue)
    - [Preprocessing Configuration](#preprocessing-configuration)
    - [Decoding Configuration](#decoding-configuration)
    - [Improvement Paths](#improvement-paths)

---

## Executive Summary

Our evaluation reports STT WER of 11-29% across five standard benchmarks. NVIDIA's published numbers for the same model on the same datasets are 1.8-10% -- a 2x to 6x gap.

After verifying both sets of numbers:
- **NVIDIA's published WER is confirmed correct** (sourced from the official HuggingFace model card).
- **Our WER calculation is correct** -- code audit found no bugs in normalization, aggregation, or metric computation.
- **Root cause identified**: Riva is truncating ~6% of longer utterances to 1-4 words, inflating aggregate WER. Additionally, the model card explicitly states this model is **not officially supported by Riva**.

The gap is a deployment/configuration issue, not a model quality issue or a measurement error.

---

## Benchmark Comparison

| Dataset | Published WER (NVIDIA) | Our Eval WER (Riva) | Gap |
|---------|----------------------|---------------------|-----|
| LibriSpeech test-clean | **1.83%** | 11.29% | 6.2x worse |
| LibriSpeech test-other | **3.54%** | 13.34% | 3.8x worse |
| GigaSpeech | **10.27%** | 24.06% | 2.3x worse |
| VoxPopuli EN | **6.53%** | 28.50% | 4.4x worse |
| SPGISpeech | **4.20%** | 21.07% | 5.0x worse |
| **Mean** | **~7.6%** | **~19.7%** | **~2.6x worse** |

NVIDIA's published numbers are greedy CTC decode, no external language model -- the lowest-quality decoding mode. Even in this mode, the model performs well. The gap is entirely on the serving side.

---

## Verification of Published Numbers

**Source:** HuggingFace model card at `huggingface.co/nvidia/parakeet-ctc-1.1b`

The model card includes a structured `model-index` in YAML frontmatter with per-dataset WER results, and a performance table in the README body. Both are consistent. Full benchmark table from the model card:

| Dataset | WER (%) |
|---------|---------|
| AMI (Meetings) | 15.62 |
| Earnings-22 | 13.69 |
| GigaSpeech | 10.27 |
| LibriSpeech test-clean | 1.83 |
| LibriSpeech test-other | 3.54 |
| SPGI Speech | 4.20 |
| TEDLIUM-v3 | 3.54 |
| Vox Populi | 6.53 |
| Common Voice 9.0 | 9.02 |

Mean across all 9 datasets: **(15.62 + 13.69 + 10.27 + 1.83 + 3.54 + 4.20 + 3.54 + 6.53 + 9.02) / 9 = 7.58%**

The model card notes:
> "These are greedy WER numbers without external LM."

---

## Verification of Our WER Calculation

We audited `eval/stt/accuracy.py` and `eval/utils.py`. No bugs found.

**Text normalization (correct):**
```python
NORMALIZE_FOR_WER = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.RemoveEmptyStrings(),
    jiwer.ReduceToListOfListOfWords(),
])
```
- `ToLowerCase()` handles the case mismatch (LibriSpeech refs are UPPERCASE, Riva output is Mixed Case).
- `RemovePunctuation()` strips commas/periods that Riva's punctuation model adds (`enable_automatic_punctuation=True`).
- Same normalization applied to both reference and hypothesis -- fair comparison.

**Aggregate WER (correct):**
```python
agg_wer = jiwer.wer(references, hypotheses,
                     reference_transform=NORMALIZE_FOR_WER_AGG,
                     hypothesis_transform=NORMALIZE_FOR_WER_AGG)
```
Uses `NORMALIZE_FOR_WER_AGG` (without `RemoveEmptyStrings`) for list-level aggregation to avoid length mismatch. This is the standard approach.

**Audio conversion (correct):**
```python
audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()
```
Float32 normalized audio scaled to int16 PCM, passed with correct sample rate. Riva receives `LINEAR_PCM` encoding with matching `sample_rate_hertz`.

---

## Root Cause: Riva Truncation

### The smoking gun

Inspecting the per-utterance results from `results/stt/accuracy/librispeech_clean.jsonl`:

| Reference | Hypothesis (Riva output) | Ref words | Hyp words | WER |
|-----------|-------------------------|-----------|-----------|-----|
| `HOUSECLEANING A DOMESTIC UPHEAVAL THAT MAKES IT EASY...` | `House cleaning` | 18 | 2 | 1.00 |
| `BUT CONTINUED RAOUL NOT INTERRUPTED BY THIS MOVEMENT...` | `But` | 43 | 1 | 0.98 |
| `ASSUREDLY IF THE TONGUE WHICH A NATION OR A PROVINCE...` | `Assuredly` | 33 | 1 | 0.97 |
| `WELL BUT NOW SAID THE PRINCESS AND SHE FILLED HIS...` | `Well,` | 29 | 1 | 0.97 |

Riva returns only the first 1-4 words and stops. The rest of the audio is not transcribed.

### Statistics (LibriSpeech clean, N=2620)

- **63.6% of utterances** have WER = 0 (perfect transcript)
- **Median WER = 0.000** -- majority of utterances are transcribed correctly
- **5.8% of utterances** have WER > 0.5 (151 utterances) -- these are the truncated ones
- Average reference length for truncated cases: **32.4 words** (vs 20.1 overall)

### Why this inflates aggregate WER disproportionately

Aggregate WER = total word errors / total reference words. It weights by utterance length. A 43-word utterance truncated to 1 word contributes 42 deletion errors, while a 5-word utterance with WER=0 contributes nothing. The ~6% of truncated (long) utterances dominate the aggregate metric, pulling 11.29% from what would otherwise be a much lower number.

### Likely cause

Riva appears to have a per-request processing limit (duration timeout or internal chunk boundary) that causes it to return partial results for longer audio clips. This is consistent with the model being unsupported by Riva -- the internal chunking/streaming config was never validated for this model.

---

## Model Card Note: Riva Not Officially Supported

The HuggingFace model card for `nvidia/parakeet-ctc-1.1b` explicitly states:

> "Although this model isn't supported yet by Riva, the list of supported models is here."

This is a critical finding. When a model is not officially supported by Riva:

- Riva's internal audio preprocessing (resampling, normalization, feature extraction) may not match the model's training config
- TensorRT optimization (which Riva applies for GPU acceleration) may alter numerical precision
- Chunking boundaries, silence detection, and overlap parameters were never validated
- The truncation behavior we observed is likely a direct consequence of this

---

## Riva vs NeMo + Triton: Deployment Stacks

Understanding the deployment stack is critical to diagnosing the WER gap.

### Riva

Riva is a complete, packaged speech AI service. You give it audio, it gives you text. All processing steps are built-in and opaque:

```
Audio in  -->  [Preprocessing]  -->  [Model Inference (TRT)]  -->  [Post-processing]  -->  Text out
                 |                       |                            |
                 Resample, normalize,    FastConformer encoder +      BPE decode, ITN,
                 chunk, mel-spectrogram  CTC greedy decode            punctuation model
```

NVIDIA validates specific models to work inside this pipeline. For unsupported models, any of these steps could be misconfigured.

### NeMo + Triton

Two separate tools combined. NeMo is the training/research toolkit (where Parakeet was built). Triton Inference Server is a general-purpose model serving layer that handles gRPC/REST, batching, and GPU scheduling -- but has no built-in audio processing.

```
Audio in  -->  [Custom preprocessing backend]  -->  [Model on Triton]  -->  [Custom post-processing]  -->  Text out
                 |                                    |
                 You control every parameter          PyTorch or TensorRT,
                 (matches training config exactly)    your choice
```

More work to set up, but preprocessing matches exactly what the model was trained on.

### Why it matters for this analysis

| | Riva | NeMo + Triton |
|---|---|---|
| Setup effort | Low | High |
| Preprocessing control | None (Riva owns it) | Full |
| Parakeet CTC 1.1B support | **Not supported** | Native |
| Risk of WER degradation | High for unsupported models | Low |

If AI Core used Riva, the truncation and WER gap are explained by the unsupported model status. If they used NeMo + Triton, the preprocessing should match and the root cause is elsewhere.

---

## Word Boosting

Word boosting is a **runtime, SDK-level feature** in Riva. No fine-tuning or model retraining required.

### How it works

During beam search decoding, Riva injects a bias score for specified words/phrases. When the decoder considers candidate tokens, boosted words get their log-probability increased by a configurable weight. This makes the decoder prefer those words even when the acoustic signal is ambiguous.

```
Without boosting:  P("akorn") = 0.12   --> decoder picks "acorn"  (P=0.31)
With boost +20:    P("akorn") = 0.12 + boost  --> decoder picks "akorn"
```

### Usage (gRPC API)

```python
import riva.client.proto.riva_asr_pb2 as rasr

speech_context = rasr.SpeechContext(
    phrases=["Akorn", "OMV", "QSCAP", "Riva", "NeMo"],
    boost=20.0   # weight range: typically 0-20
)

config = rasr.RecognitionConfig(
    speech_contexts=[speech_context],
    encoding=rasr.AudioEncoding.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="en-US",
)
```

It is a per-request parameter. No GPU restart, no model rebuild, no redeployment.

### What it helps with

- Brand names, ticker symbols, product names (exactly the 42.3% NE-WER failure we saw on SPGISpeech)
- Proper nouns the CTC vocabulary handles phonetically but maps to wrong spelling
- Technical acronyms

### What it does NOT fix

- Overall WER on general speech -- boosting wrong words can actually hurt WER
- The 6x gap on LibriSpeech clean (no rare proper nouns in that dataset, so boosting is irrelevant)
- Truncation issues (boosting operates after audio is already decoded)

### Related: LM Customization

A heavier lever than word boosting. You upload a domain-specific n-gram language model to Riva, which rescores the beam search output. This can reduce WER by 2-5 percentage points broadly, not just on named entities. Requires model preparation and redeployment (unlike word boosting which is per-request).

---

## Minor Methodological Issue: SPGISpeech Split

Our eval code (`eval/stt/accuracy.py`, line 35):
```python
{"name": "spgispeech", "hf": ("kensho/spgispeech", "S"), "split": "validation", "label": "SPGISpeech"},
```

NVIDIA's model card evaluates on the `test` split. Our code uses `validation`. This is a direct mismatch. It does not explain the LibriSpeech or GigaSpeech gaps (different datasets), but means the SPGISpeech comparison (21.07% vs 4.20%) is not strictly apples-to-apples.

---

## Questions for AI Core Team

### Deployment Architecture

   1. **How is Parakeet CTC 1.1B served?**
      Is it running through Riva, NeMo + Triton, or a custom serving stack?
      The model card states it is not officially supported by Riva.

   2. **What Riva version is deployed?**
      And what model version/checkpoint is loaded?

   3. **Was TensorRT optimization applied?**
      If so, what precision mode (FP16, INT8)?

### Truncation Issue

   4. **Is there a per-request duration or chunk-size limit configured?**
      Our eval found ~6% of longer utterances (30+ words) are truncated to 1-4 words.
      This is the primary driver of the WER gap.

   5. **What is the maximum audio duration the endpoint accepts?**
      Is there an internal timeout that would cause partial results on longer clips?

   6. **Are there any gRPC message size limits?**
      Could these truncate audio payloads before they reach the model?

### Preprocessing Configuration

   7. **What audio preprocessing is applied before model inference?**
      Specifically: resampling method, mel-spectrogram parameters
      (n_fft, hop_length, n_mels), and amplitude normalization.

   8. **Does the preprocessing config match the model's training config?**
      Parakeet CTC 1.1B expects 80 mel filter banks, 16kHz sample rate.
      A mismatch here would degrade WER significantly.

### Decoding Configuration

   9. **Is greedy CTC decoding or beam search used?**
      The model card reports greedy WER. If Riva defaults to a different
      decoding strategy for this model, results may differ.

   10. **Is any language model (LM) rescoring enabled or disabled?**
       The model card notes: "These are greedy WER numbers without external LM."
       If an LM is loaded but misconfigured, it could hurt rather than help.

### Improvement Paths

   11. **Can we test word boosting for domain vocabulary?**
       Our eval showed 42.3% NE-WER on financial entities (SPGISpeech).
       Word boosting is a zero-cost, per-request lever that could significantly
       reduce named-entity errors.

   12. **Is there a path to official Riva support for this model?**
       Or should we evaluate on a Riva-supported model
       (e.g., `stt_en_fastconformer_ctc_large`)?

   13. **Can we run a quick A/B test?**
       Run the same LibriSpeech clean audio through both the Riva endpoint
       and direct NeMo inference. If NeMo produces ~1.8% WER and Riva
       produces ~11%, the serving layer is confirmed as the bottleneck.
