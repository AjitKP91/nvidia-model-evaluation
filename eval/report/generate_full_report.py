"""Comprehensive single-file HTML report — a beginner-friendly website.

Covers Phase 0 discovery, STT 1.1-1.13 (including the full 34-language
multilingual acceptance matrix), and TTS 2.1-2.9 — each test led by a
plain-language explanation (what it is, how the code computes it, which dataset
it uses, with clickable links) followed by an interactive chart and a compact
results table.

Design:
  * STDLIB ONLY for generation (csv, json, ast, html, pathlib, datetime). Charts
    are rendered client-side by Chart.js (CDN); the generator only emits
    <canvas> elements plus a JSON spec it reads at load.
  * HONEST: a test with no data on disk renders an explicit notice, never a
    fabricated number or chart.
  * Clean light theme + a persisted dark-mode toggle, validated dataviz palette,
    sticky sidebar navigation with per-test links for BOTH STT and TTS.

Usage:
    python3 eval/report/generate_full_report.py \
        --run results/run-04-11-26/riva \
        --acceptance <run>/stt/language_acceptance/language_acceptance.csv \
        --out docs/full-report.html [--dual-run-link dual-run-report.html]
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# small parsing helpers  (data layer — unchanged behaviour)
# ----------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def maybe_literal(val):
    """Parse a cell that may hold a Python dict/list string (e.g. percentiles)."""
    if not isinstance(val, str):
        return val
    s = val.strip()
    if s[:1] in "{[" and s[-1:] in "}]":
        try:
            return ast.literal_eval(s)
        except Exception:
            return val
    return val


def fnum(val, nd=3):
    try:
        return f"{float(val):.{nd}f}"
    except (TypeError, ValueError):
        return esc(val)


def fval(val, nd=3):
    """Float or None (for chart JSON) — never raises."""
    try:
        return round(float(val), nd)
    except (TypeError, ValueError):
        return None


def pct(val, nd=1):
    try:
        return f"{float(val) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return esc(val)


def esc(val) -> str:
    return html.escape("" if val is None else str(val))


# ----------------------------------------------------------------------------
# chart registry — the generator collects specs; Chart.js builds them at load
# ----------------------------------------------------------------------------

_CHARTS: list[dict] = []


def add_chart(cid: str, ctype: str, data: dict, options: dict | None = None,
              height: int = 320) -> str:
    """Register a Chart.js spec and return the <canvas> + wrapper HTML.

    `data` and `options` may use string color tokens like "@s1" / "@good" /
    "@grid" / "@ink" / "@muted"; the client resolves them from CSS variables so
    the chart re-themes on the dark-mode toggle. Returns '' and records nothing
    if there is no data (honest-empty)."""
    _CHARTS.append({"id": cid, "type": ctype, "data": data,
                    "options": options or {}})
    return (f'<div class="chartbox" style="height:{height}px">'
            f'<canvas id="{cid}"></canvas></div>')


# ----------------------------------------------------------------------------
# shared glossary — linked from test intros
# ----------------------------------------------------------------------------

GLOSSARY = {
    "WER": "Word Error Rate — the share of words wrong (substitutions + "
           "insertions + deletions) after text normalisation. Lower is better; "
           "0% is perfect.",
    "CER": "Character Error Rate — like WER but per character. Used for "
           "Chinese/Japanese where word boundaries are ambiguous.",
    "MER": "Match Error Rate — errors as a fraction of the aligned length; a "
           "companion to WER that is bounded 0–1.",
    "RTF": "Real-Time Factor — processing time ÷ audio duration. Below 1.0 "
           "means faster than real time.",
    "RTFx": "Inverse RTF (audio ÷ processing). Higher is faster; RTFx 20 means "
            "20× faster than real time.",
    "MOS": "Mean Opinion Score — a 1–5 naturalness rating. Here predicted by a "
           "neural model (UTMOS/DNSMOS) rather than human listeners.",
    "DNSMOS": "A language-agnostic ONNX model (ITU-T P.835) that predicts SIG "
              "(speech), BAK (background) and OVRL (overall) quality.",
    "UTMOS": "A neural MOS predictor validated on English only — reported for "
             "English and omitted for other languages.",
    "TTFB": "Time To First Byte — how long until the first audio chunk arrives "
            "(streaming responsiveness).",
    "TTFW": "Time To First Word — streaming STT latency to the first word.",
    "ECE": "Expected Calibration Error — gap between a model's confidence and "
           "its actual accuracy. Lower is better.",
    "AUROC": "Area under the ROC curve for using confidence to detect errors; "
             "0.5 is random, 1.0 is perfect.",
    "Brier": "Mean squared error of probabilistic predictions; lower is better.",
    "round-trip WER": "Synthesize text → transcribe the audio with Whisper "
                      "large-v3 → compare to the input. Low WER means the speech "
                      "was intelligible.",
    "nPVI": "normalised Pairwise Variability Index — a rhythm measure of how "
            "much successive units vary in duration.",
    "fairness delta": "The gap in WER between an accent group and the best "
                      "group — a fairness measure across accents.",
}


def _ds(name, url):
    return {"name": name, "url": url}


# Base URL for linking "how it works" copy to the actual source on GitHub.
# Pinned to a branch so line anchors stay meaningful across the default branch.
CODE_BASE = "https://github.com/AjitKP91/nvidia-model-evaluation/blob/main/"


def _code(label, path, line=None):
    """A clickable link to a source file (optionally a line anchor)."""
    url = CODE_BASE + path + (f"#L{line}" if line else "")
    return {"name": label, "url": url}


# ----------------------------------------------------------------------------
# TEST_META — authored, code-accurate copy for every test
#   summary : one plain line (always visible)
#   what    : what the test measures, in plain language
#   how     : the real code logic (functions / metric computation)
#   datasets: list of {name, url} (clickable)
#   metrics : short glossary of this test's key columns
# ----------------------------------------------------------------------------

TEST_META = {
    "1.1": {
        "summary": "How accurately the model transcribes clean, read English "
                   "speech across eight standard benchmarks.",
        "what": "The baseline accuracy test. It plays clean English audio from "
                "well-known public corpora and compares the transcript to the "
                "reference, so you can place the model against published numbers.",
        "how": "Clean English clips are streamed to the model over gRPC and the "
               "returned transcript is compared to the dataset's reference text. "
               "Before comparing, both strings pass through Whisper's "
               "EnglishTextNormalizer, which expands numbers and currency, fixes "
               "contractions and strips casing and punctuation so only genuine "
               "word errors count. The two cleaned word sequences are then lined "
               "up by edit distance (jiwer): WER = (words substituted + inserted + "
               "deleted) ÷ reference words, with MER and character-level CER "
               "reported alongside. Each of the eight benchmarks is scored "
               "separately and gets its own PASS/warn verdict, and a raw cased-and-"
               "punctuated WER is also reported so the un-normalised picture is "
               "visible too.",
        "code": [
            _code("accuracy.py — run() (stream, score, per-dataset verdict)",
                  "eval/stt/accuracy.py", 97),
            _code("utils.py — score_pair() (WER/MER/CER edit distance)",
                  "eval/utils.py", 144),
            _code("utils.py — normalize_for_language() (EnglishTextNormalizer)",
                  "eval/utils.py", 126),
        ],
        "datasets": [
            _ds("LibriSpeech (test-clean / test-other)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
            _ds("VoxPopuli (en)",
                "https://huggingface.co/datasets/facebook/voxpopuli"),
            _ds("GigaSpeech", "https://huggingface.co/datasets/speechcolab/gigaspeech"),
            _ds("SPGISpeech", "https://huggingface.co/datasets/kensho/spgispeech"),
            _ds("TED-LIUM 3", "https://huggingface.co/datasets/LIUM/tedlium"),
            _ds("AMI (meetings)", "https://huggingface.co/datasets/edinburghcstr/ami"),
            _ds("Earnings-22", "https://huggingface.co/datasets/revdotcom/earnings22"),
        ],
        "metrics": ["WER", "CER", "MER"],
    },
    "1.2": {
        "summary": "Batch throughput and latency — how fast the model "
                   "transcribes, by clip length and under concurrent load.",
        "what": "A speed test (not accuracy). It measures how quickly audio is "
                "processed for short/medium/long clips and how latency holds up "
                "when many requests run at once.",
        "how": "This times the model rather than scoring its words. Each clip is "
               "sent and the wall-clock time is divided by the clip's audio "
               "duration to get RTF (real-time factor); RTFx is its inverse, so "
               "RTFx 10 means ten seconds of audio transcribed per second of "
               "compute. Clips are grouped into short / medium / long buckets to "
               "show how speed scales with length. A second pass replays the same "
               "workload at 1, 5, 10 and 20 simultaneous requests and records "
               "median (P50) and tail (P99) latency plus mean RTF at each level, "
               "exposing how throughput holds up under concurrent load.",
        "code": [
            _code("performance.py — run() (RTF/RTFx + concurrency sweep)",
                  "eval/stt/performance.py", 37),
            _code("performance.py — _pick_utterances_by_duration() (length buckets)",
                  "eval/stt/performance.py", 25),
        ],
        "datasets": [
            _ds("LibriSpeech (length-bucketed clips)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
        ],
        "metrics": ["RTF", "RTFx"],
    },
    "1.3": {
        "summary": "Streaming responsiveness — how quickly partial and final "
                   "words arrive during live transcription.",
        "what": "Measures the live-dictation experience: how soon the first word "
                "appears, how long results take to finalise, how stable partial "
                "hypotheses are, and whether streaming hurts accuracy.",
        "how": "The clip is streamed in small chunks and every partial and final "
               "result from StreamingRecognize is timestamped. From those "
               "timestamps the test derives time-to-first-word (how soon any text "
               "appears), finalisation latency (how long after the audio ends the "
               "final text arrives) and partial latency. A stability rate measures "
               "how often an interim word survives unchanged into the final "
               "transcript — jitter-free streaming scores near 1.0. The same audio "
               "is also transcribed in batch and the WER compared, so you can see "
               "whether streaming costs accuracy. English-only: streaming timing "
               "is a property of the serving stack, not the language.",
        "code": [
            _code("streaming.py — run() (drive StreamingRecognize, collect timings)",
                  "eval/stt/streaming.py", 68),
            _code("streaming.py — _compute_stability_rate() (partial survival)",
                  "eval/stt/streaming.py", 38),
            _code("streaming.py — _compute_partial_latencies()",
                  "eval/stt/streaming.py", 57),
        ],
        "datasets": [
            _ds("LibriSpeech (streamed clips)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
        ],
        "metrics": ["TTFW", "RTF", "WER"],
    },
    "1.4": {
        "summary": "REST vs gRPC latency — not applicable: this deployment "
                   "serves gRPC StreamingRecognize only.",
        "what": "Would compare the REST and gRPC transports for the same audio. "
                "The SAP AI Core gateway exposes only gRPC streaming, so there is "
                "no REST path to compare and the test records an honest N/A.",
        "how": "The test is written to send identical audio over both REST and "
               "gRPC and compare their latency head-to-head. This deployment's SAP "
               "AI Core gateway only implements gRPC StreamingRecognize, so there "
               "is no REST endpoint to measure. Rather than invent a comparison, "
               "the code detects the missing interface and records an explicit "
               "N/A — the report shows that rationale instead of a chart.",
        "code": [
            _code("rest_vs_grpc.py — run() (detects missing REST, records N/A)",
                  "eval/stt/rest_vs_grpc.py", 29),
        ],
        "datasets": [],
        "metrics": [],
    },
    "1.5": {
        "summary": "Noise robustness — how accuracy degrades as background noise "
                   "gets louder, across four noise types.",
        "what": "Mixes clean speech with real-world noise at controlled signal-"
                "to-noise ratios and tracks WER as conditions worsen — a "
                "stress test for noisy rooms, cafés, streets and babble.",
        "how": "Clean speech is mixed with noise at a fixed signal-to-noise ratio "
               "(SNR — higher dB means quieter noise) and then transcribed, so you "
               "can watch accuracy fall as conditions worsen. Four noise types are "
               "used: synthetic Gaussian hiss plus real babble (MUSAN) and street "
               "and office recordings (DEMAND), each injected across SNRs from −5 "
               "to +20 dB. For every (noise type, SNR) pair it scores WER exactly "
               "as Test 1.1 does, and reports delta_rel — the percentage increase "
               "in WER versus the clean baseline — so the degradation curve is easy "
               "to read.",
        "code": [
            _code("noise_robustness.py — run() (grid over noise × SNR)",
                  "eval/stt/noise_robustness.py", 58),
            _code("noise_robustness.py — _inject_noise() (SNR mixing)",
                  "eval/stt/noise_robustness.py", 47),
            _code("noise_robustness.py — _run_english() (score vs clean baseline)",
                  "eval/stt/noise_robustness.py", 92),
        ],
        "datasets": [
            _ds("LibriSpeech (clean source)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
            _ds("MUSAN (noise)", "https://www.openslr.org/17/"),
            _ds("DEMAND (environmental noise)",
                "https://zenodo.org/record/1227121"),
        ],
        "metrics": ["WER"],
    },
    "1.6": {
        "summary": "Accent robustness — WER across English accents and the "
                   "fairness gap between the best and worst.",
        "what": "Checks whether the model treats speakers of different accents "
                "equally by scoring the same kind of speech across many accent "
                "groups and measuring the spread.",
        "how": "The same kind of read speech is transcribed for many speaker "
               "groups, each a different accent, and WER is computed per group. "
               "VCTK supplies UK and Commonwealth accents; L2-ARCTIC supplies "
               "non-native (L2) English accents; clips are bucketed by their "
               "accent label. The headline is fairness_delta = each group's WER "
               "minus the best group's WER, which shows how much worse the model "
               "does on its weakest accent than its strongest — the gap, not just "
               "the average.",
        "code": [
            _code("accent.py — run() (per-accent WER + fairness delta)",
                  "eval/stt/accent.py", 124),
            _code("accent.py — _load_vctk_groups()",
                  "eval/stt/accent.py", 39),
            _code("accent.py — _load_l2arctic_groups()",
                  "eval/stt/accent.py", 64),
        ],
        "datasets": [
            _ds("VCTK (UK/Commonwealth accents)",
                "https://datashare.ed.ac.uk/handle/10283/3443"),
            _ds("L2-ARCTIC (non-native accents)",
                "https://psi.engr.tamu.edu/l2-arctic-corpus/"),
        ],
        "metrics": ["WER", "fairness delta"],
    },
    "1.7": {
        "summary": "Long-form audio — accuracy and stability on multi-minute "
                   "recordings, whole-file vs chunked.",
        "what": "Tests whether the model holds up on long recordings (meetings, "
                "talks) without drifting, hallucinating text, or repeating "
                "itself.",
        "how": "Each long recording is transcribed two ways — as one whole file "
               "and as a sequence of 30-second chunks — and both are scored "
               "(full_wer vs chunked_wer) to reveal whether long context helps or "
               "hurts. Two failure detectors then run on the output: a "
               "hallucination rate flags text the model invented that isn't in the "
               "reference, and a repetition rate flags looping n-grams (the model "
               "getting stuck repeating a phrase). For CJK languages the detectors "
               "switch to character n-grams and are marked approximate rather than "
               "reporting a bogus word-based number.",
        "code": [
            _code("long_form.py — run() (whole vs chunked, drift detection)",
                  "eval/stt/long_form.py", 85),
            _code("long_form.py — _detect_hallucinations()",
                  "eval/stt/long_form.py", 29),
            _code("long_form.py — _detect_repetitions()",
                  "eval/stt/long_form.py", 62),
        ],
        "datasets": [
            _ds("TED-LIUM 3 (long talks)",
                "https://huggingface.co/datasets/LIUM/tedlium"),
            _ds("AMI (long meetings)",
                "https://huggingface.co/datasets/edinburghcstr/ami"),
        ],
        "metrics": ["WER"],
    },
    "1.8": {
        "summary": "Domain vocabulary — accuracy on jargon-heavy speech and how "
                   "well key terms and named entities are recognised.",
        "what": "Measures performance on specialised content (finance, "
                "meetings) where rare terms and proper nouns matter more than "
                "overall WER.",
        "how": "On jargon-heavy speech, overall WER hides what matters, so this "
               "zooms in on the hard words. It computes domain_wer (WER on the "
               "specialised clips), a key-term recall rate (KRR — of the domain "
               "terms and named entities present in the reference, how many the "
               "model got right) and NE-WER (error rate restricted to named "
               "entities only). It also surfaces the most-frequently-"
               "misrecognised terms, so you can see exactly where the vocabulary "
               "breaks down.",
        "code": [
            _code("domain.py — run() (domain WER, KRR, NE-WER, OOV)",
                  "eval/stt/domain.py", 77),
            _code("domain.py — _compute_krr() (key-term recall)",
                  "eval/stt/domain.py", 38),
            _code("domain.py — _compute_ne_wer() (named-entity WER)",
                  "eval/stt/domain.py", 47),
        ],
        "datasets": [
            _ds("SPGISpeech (finance)",
                "https://huggingface.co/datasets/kensho/spgispeech"),
            _ds("Earnings-22 (earnings calls)",
                "https://huggingface.co/datasets/revdotcom/earnings22"),
        ],
        "metrics": ["WER"],
    },
    "1.9": {
        "summary": "Output quality — how well the transcript handles "
                   "punctuation, capitalisation and numbers.",
        "what": "Clean words aren't enough: this checks the formatting that "
                "makes transcripts readable — sentence punctuation, proper "
                "capitalisation and correctly written numbers.",
        "how": "Beyond getting words right, a usable transcript needs correct "
               "formatting, so this scores three things against formatted "
               "references. Punctuation is scored as an F1 per mark — the "
               "precision and recall of each '.', ',', '?' and so on, so both "
               "missing and spurious marks count. Capitalisation accuracy checks "
               "the right letters are upper-cased, and is reported N/A for scripts "
               "with no case (CJK/Thai) rather than a false zero. Number accuracy "
               "checks digits are written correctly, reported N/A for native-digit "
               "scripts. Per-language runs use language-aware punctuation sets "
               "(Spanish ¿¡, CJK 。、, Arabic ؟،).",
        "code": [
            _code("output_quality.py — run() (punct / caps / number scoring)",
                  "eval/stt/output_quality.py", 136),
            _code("output_quality.py — _compute_punct_f1()",
                  "eval/stt/output_quality.py", 33),
            _code("output_quality.py — _compute_capitalisation_accuracy()",
                  "eval/stt/output_quality.py", 87),
            _code("output_quality.py — _compute_number_accuracy()",
                  "eval/stt/output_quality.py", 114),
        ],
        "datasets": [
            _ds("LibriSpeech / curated formatted references",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
        ],
        "metrics": [],
    },
    "1.10": {
        "summary": "Confidence calibration — whether the model's confidence "
                   "scores actually track its accuracy.",
        "what": "A trustworthy model should be confident when right and unsure "
                "when wrong. This test checks that the reported confidence means "
                "what it says.",
        "how": "A confidence score is only useful if it tracks whether the model "
               "was actually right. Each recognised word is labelled correct or "
               "not by aligning to the reference, then bucketed by its confidence. "
               "Expected Calibration Error (ECE) is the average gap between a "
               "bucket's stated confidence and its measured accuracy (0 = "
               "perfectly calibrated). AUROC measures how well confidence alone "
               "separates right from wrong words (0.5 = useless, 1.0 = perfect "
               "error detector). The Brier score is the mean squared error of "
               "confidence-as-probability. A per-bin reliability curve is plotted "
               "against the ideal y=x line.",
        "code": [
            _code("confidence.py — run() (bin words, build reliability curve)",
                  "eval/stt/confidence.py", 45),
            _code("confidence.py — _calibration_summary() (ECE / AUROC / Brier)",
                  "eval/stt/confidence.py", 145),
            _code("confidence.py — _compute_word_correctness() (align to reference)",
                  "eval/stt/confidence.py", 25),
        ],
        "datasets": [
            _ds("LibriSpeech (word-level confidences)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
        ],
        "metrics": ["ECE", "AUROC", "Brier"],
    },
    "1.11": {
        "summary": "Format robustness — whether the endpoint accepts varied "
                   "audio encodings, sample rates and channel layouts.",
        "what": "An operational check that the service tolerates the audio "
                "formats real clients send (WAV/FLAC, 8–48 kHz, mono/stereo) "
                "rather than only one golden format.",
        "how": "This is an operational compatibility check, not an accuracy test. "
               "One fixed utterance is re-encoded into many combinations of "
               "container, codec, sample rate, bit depth and channel layout (WAV "
               "and FLAC, 8–48 kHz, mono/stereo, and so on) and submitted to the "
               "endpoint. Each variant is recorded pass or fail along with any "
               "error the server returned, producing a support matrix. It stays "
               "English-only because decoding and resampling happen in the audio "
               "frontend and don't depend on the language.",
        "code": [
            _code("format_robustness.py — run() (submit each variant, pass/fail)",
                  "eval/stt/format_robustness.py", 67),
            _code("format_robustness.py — _convert_format() (re-encode audio)",
                  "eval/stt/format_robustness.py", 39),
        ],
        "datasets": [
            _ds("LibriSpeech (re-encoded variants)",
                "https://huggingface.co/datasets/openslr/librispeech_asr"),
        ],
        "metrics": [],
    },
    "1.12": {
        "summary": "Multilingual accuracy — WER/CER per language for the model's "
                   "supported languages.",
        "what": "Measures transcription accuracy language by language (not just "
                "English), using a common multilingual benchmark so languages are "
                "comparable.",
        "how": "For each language the harness pulls up to 200 clean read-speech "
               "clips from FLEURS and sends each one to the model, getting back a "
               "predicted transcript. Before anything is counted, both the correct "
               "reference text and the model's transcript are run through the same "
               "text normaliser (Whisper's — the English one for English, the "
               "basic one for every other language): this lowercases, strips "
               "punctuation and standardises spacing, so differences in formatting "
               "don't count as mistakes — only genuinely wrong words do. The two "
               "cleaned strings are then compared by edit distance. For "
               "space-separated languages the score is WER = (words substituted + "
               "inserted + deleted) ÷ (number of words in the reference); a WER of "
               "0.15 means roughly 15 words in 100 were wrong. Chinese, Japanese "
               "and Thai have no spaces between words, so they are scored the same "
               "way but per character (CER) after all spaces are removed — this "
               "endpoint emits one space per character, and counting those would "
               "otherwise push the score toward 1.0. Each clip is scored "
               "individually; the per-clip scores are averaged into the language's "
               "number, with a bootstrap confidence interval, and compared against "
               "PASS/FLAG/FAIL thresholds. Every language is reported on its own "
               "line — there is deliberately no blended macro-average, because the "
               "languages span very different difficulty tiers and one combined "
               "number would hide more than it shows.",
        "datasets": [
            _ds("FLEURS (102-language read speech)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "note": "Why the Metric column is mixed, and why Korean is CER. WER (word "
                "error rate) needs word boundaries, which it finds by splitting on "
                "spaces — so it is used for the space-separated languages here "
                "(English, German, Vietnamese, and the rest). Japanese and Chinese "
                "(Mandarin) write text as an unbroken run of characters with no "
                "spaces, so a word-level rate is meaningless — a single wrong "
                "character would score the whole line as one bad 'word'. Korean does "
                "use spaces, but we score it by CER too, matching NVIDIA's model "
                "card (\"Japanese, Korean, and Mandarin are evaluated using CER\"): "
                "Korean spacing (띄어쓰기) is highly variable and its inflectional "
                "word-endings make WER punish near-correct words as whole-word "
                "errors — Korean scored ~26% by WER but ~10% by CER (card: 7.3%). "
                "CER is the identical edit-distance math, counted per character "
                "after whitespace is removed. Greek is left out entirely: the card "
                "places it in the adaptation-ready tier with no published score, and "
                "our ~50% WER reflects a language the model isn't tuned for — see "
                "the note under the table.",
        "code": [
            _code("multilingual.py — run() (load clips, transcribe, aggregate)",
                  "eval/stt/multilingual.py", 77),
            _code("utils.py — score_pair() (WER/CER edit distance)",
                  "eval/utils.py", 144),
            _code("utils.py — normalize_for_language() (Whisper normaliser)",
                  "eval/utils.py", 126),
            _code("utils.py — bootstrap_ci() (confidence interval)",
                  "eval/utils.py", 280),
        ],
        "metrics": ["WER", "CER"],
    },
    "1.13": {
        "summary": "Language acceptance — a live probe of all 34 model languages: "
                   "does each one actually return a transcript on this endpoint?",
        "what": "The multilingual centrepiece. Rather than trusting the model "
                "card, it sends one real clip per language and records whether "
                "the deployment returns usable text — rendered SAP-Joule-first.",
        "how": "Before trusting any accuracy number, this confirms the endpoint "
               "actually returns text for each language. For every one of the "
               "model's languages it streams a few real FLEURS clips and "
               "classifies the outcome: ok (usable text came back), empty (the "
               "endpoint returned nothing), error (the call failed) or load_failed "
               "(the dataset couldn't be fetched). It keeps a sample reference / "
               "hypothesis pair and a sample WER/CER for a sanity glance. A "
               "language that returns nothing is logged as a graceful no_output "
               "row — never a fabricated score — and ok languages are skipped on "
               "re-run so it's cheap to repeat.",
        "code": [
            _code("language_acceptance.py — run() (probe every language, classify)",
                  "eval/stt/language_acceptance.py", 175),
            _code("language_acceptance.py — probe() (stream clips, status + sample)",
                  "eval/stt/language_acceptance.py", 60),
        ],
        "datasets": [
            _ds("FLEURS (one clip per language)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["WER", "CER"],
    },
}

TEST_META.update({
    "2.1": {
        "summary": "Naturalness — how human and clean the synthesized voice "
                   "sounds, scored by neural MOS predictors.",
        "what": "Rates how natural the generated speech sounds without a human "
                "panel, using models that predict listener opinion scores.",
        "how": "The model synthesizes a set of sentences and two neural predictors "
               "rate how natural the audio sounds, with no human panel needed. "
               "DNSMOS (an ONNX model implementing ITU-T P.835) returns SIG "
               "(speech quality), BAK (background cleanliness) and OVRL (overall) "
               "on a 1–5 MOS scale and is language-agnostic, so it is reported for "
               "every language. UTMOS also predicts a MOS but was trained and "
               "validated on English, so for non-English it would be scoring "
               "out-of-domain audio — the code records utmos: null with a note "
               "there rather than a misleading number.",
        "code": [
            _code("naturalness.py — run() (synthesize, score UTMOS + DNSMOS)",
                  "eval/tts/naturalness.py", 32),
            _code("dnsmos.py — score_dnsmos() (P.835 SIG/BAK/OVRL)",
                  "eval/tts/dnsmos.py", 145),
            _code("utmos.py — score_utmos() (English-only MOS predictor)",
                  "eval/tts/utmos.py", 44),
        ],
        "datasets": [
            _ds("Internal sentence set + FLEURS text (per language)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["MOS", "DNSMOS", "UTMOS"],
    },
    "2.2": {
        "summary": "Intelligibility — can the synthesized speech be transcribed "
                   "back correctly? (round-trip WER).",
        "what": "The clearest real-world test of TTS quality: if a strong ASR "
                "model can read the speech back accurately, a human can too.",
        "how": "This is the most real-world TTS check: synthesize the text, then "
               "transcribe the synthesized audio back with Whisper large-v3 and "
               "compare to the original text. The idea is simple — if a strong ASR "
               "can read the speech back accurately, a human can too. The "
               "round-trip is scored with the same normaliser and edit-distance "
               "WER as the STT tests (CER for Chinese and Japanese), broken down "
               "by content category (numbers, punctuation, etc.) for English and "
               "per language via FLEURS text for multilingual runs.",
        "code": [
            _code("intelligibility.py — run() (synthesize → transcribe → score)",
                  "eval/tts/intelligibility.py", 95),
            _code("intelligibility.py — _transcribe_whisper() (Whisper large-v3)",
                  "eval/tts/intelligibility.py", 69),
            _code("utils.py — score_pair() (round-trip WER/CER)",
                  "eval/utils.py", 144),
        ],
        "datasets": [
            _ds("Curated categories + FLEURS (per language)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["round-trip WER", "CER"],
    },
    "2.3": {
        "summary": "Prosody — pitch variation, rhythm, pausing and speaking rate "
                   "of the synthetic voice.",
        "what": "Flat, robotic speech is unpleasant even when intelligible. This "
                "measures expressiveness: pitch movement, rhythmic variation, "
                "pausing and tempo.",
        "how": "Intelligible speech can still sound flat and robotic, so this "
               "measures expressiveness directly from the audio — no reference "
               "recording required. It extracts the pitch (F0) contour and reports "
               "its standard deviation and range (how much the voice moves), an "
               "acoustic nPVI that captures rhythmic variation between successive "
               "voiced segments, the pause ratio (share of silence) and the "
               "speaking rate in words per minute. Together these flag monotone or "
               "unnaturally paced output. The intrusive 'F0 versus a reference' "
               "metrics aren't computable for a fixed synthetic voice — see 2.4.",
        "code": [
            _code("prosody.py — run() (per-utterance features + aggregate)",
                  "eval/tts/prosody.py", 257),
            _code("prosody.py — _measure_utterance() (F0, pause, WPM)",
                  "eval/tts/prosody.py", 142),
            _code("prosody.py — _npvi() (rhythm variation)",
                  "eval/tts/prosody.py", 102),
        ],
        "datasets": [
            _ds("Internal passages + FLEURS text (per language)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["nPVI"],
    },
    "2.4": {
        "summary": "Signal-quality (MCD/PESQ/STOI) — not applicable: these need "
                   "a same-speaker reference recording this model can't provide.",
        "what": "Would compare the waveform to a ground-truth recording of the "
                "same speaker saying the same words. A fixed synthetic persona "
                "has no such reference, so any score would measure speaker "
                "mismatch, not quality.",
        "how": "MCD, PESQ and STOI are full-reference metrics: each compares the "
               "output waveform to a recording of the same speaker saying the same "
               "words. A fixed synthetic persona has no such ground-truth "
               "recording, so any score would measure speaker mismatch, not "
               "synthesis quality. The code therefore records an explicit N/A "
               "rationale instead of emitting misleading numbers; a valid intrusive "
               "run would need a zero-shot voice-cloning deployment with held-out "
               "real-speaker audio, which is kept behind a guard.",
        "code": [
            _code("signal_quality.py — run() (records the N/A rationale)",
                  "eval/tts/signal_quality.py", 93),
            _code("signal_quality.py — _run_intrusive() (guarded MCD/PESQ/STOI path)",
                  "eval/tts/signal_quality.py", 111),
        ],
        "datasets": [],
        "metrics": [],
    },
    "2.5": {
        "summary": "Streaming latency — how quickly audio starts and streams, by "
                   "text length.",
        "what": "Measures responsiveness for live use: how soon the first audio "
                "chunk arrives (TTFB) and how fast speech is produced relative to "
                "its duration.",
        "how": "A responsiveness test for live synthesis. For each text-length "
               "bucket it streams the audio over gRPC and records time-to-first-"
               "byte (how soon the first audio chunk arrives — what a user "
               "perceives as lag) and RTF (synthesis time ÷ audio duration, so "
               "below 1.0 is faster than real time). Results are stored as P50/P99 "
               "percentile pairs so both typical and worst-case latency are "
               "visible per text length.",
        "code": [
            _code("latency.py — run() (TTFB + RTF per text-length bucket)",
                  "eval/tts/latency.py", 47),
        ],
        "datasets": [
            _ds("Length-bucketed prompt set (internal)", ""),
        ],
        "metrics": ["TTFB", "RTF"],
    },
    "2.6": {
        "summary": "Throughput & concurrency — requests-per-second and latency "
                   "as parallel load increases.",
        "what": "A capacity test: how many simultaneous synthesis requests the "
                "endpoint sustains and how latency and error rate respond.",
        "how": "A capacity test. The same synthesis request is replayed at rising "
               "concurrency — 1, 5, 10 and 20 parallel calls — and at each level "
               "it records how many succeeded, the error rate, requests per second "
               "(throughput) and P50/P99 latency. Throughput and latency are shown "
               "as two separate single-axis charts versus concurrency (never one "
               "dual-axis plot), so rising latency isn't visually conflated with "
               "rising throughput.",
        "code": [
            _code("concurrency.py — run() (sweep concurrency levels)",
                  "eval/tts/concurrency.py", 111),
            _code("concurrency.py — _run_concurrent_grpc() (parallel replay, RPS)",
                  "eval/tts/concurrency.py", 71),
        ],
        "datasets": [
            _ds("Fixed prompt replayed under load (internal)", ""),
        ],
        "metrics": ["RTF"],
    },
    "2.7": {
        "summary": "Edge cases — does the voice handle tricky input (numbers, "
                   "symbols, emoji, code, long words) gracefully?",
        "what": "Stress-tests the text frontend with awkward inputs across "
                "seventeen categories to find where synthesis breaks or "
                "degrades.",
        "how": "The text frontend is stress-tested with awkward inputs across "
               "seventeen categories — empty strings, very long text, numbers, "
               "punctuation, abbreviations, SSML markup, emoji and special "
               "characters, URLs and emails, code snippets, and more. Each item is "
               "synthesized and classified pass / fail / degraded, with a timeout "
               "guard so one bad case can't stall the suite. The report shows a "
               "pass_rate per category, so you can see which input types the voice "
               "handles cleanly and which break it.",
        "code": [
            _code("edge_cases.py — run() (run categories, classify, pass_rate)",
                  "eval/tts/edge_cases.py", 39),
            _code("edge_cases.py — _round_trip_wer() (degraded-output check)",
                  "eval/tts/edge_cases.py", 25),
        ],
        "datasets": [
            _ds("Curated edge-case suite (17 categories, internal)", ""),
        ],
        "metrics": [],
    },
    "2.8": {
        "summary": "Long-form consistency — does the voice stay the same person "
                   "across a long passage (no drift)?",
        "what": "Checks that voice identity, pitch and tempo stay stable "
                "paragraph-to-paragraph over long passages instead of drifting.",
        "how": "Over a long multi-paragraph passage a synthetic voice can slowly "
               "drift — change timbre, pitch or pace. This synthesizes each "
               "paragraph, computes an ECAPA-TDNN speaker embedding for each, and "
               "measures the cosine similarity between paragraphs: high, flat "
               "similarity means the voice stayed the same person throughout. It "
               "also tracks mean-F0 drift and speaking-rate (WPM) drift across the "
               "passage to catch pitch or tempo wandering that similarity alone "
               "might miss.",
        "code": [
            _code("long_form.py — run() (synthesize passages, measure drift)",
                  "eval/tts/long_form.py", 83),
            _code("long_form.py — _compute_speaker_embeddings() (ECAPA-TDNN)",
                  "eval/tts/long_form.py", 21),
            _code("long_form.py — _pairwise_cosine_similarity()",
                  "eval/tts/long_form.py", 61),
        ],
        "datasets": [
            _ds("Multi-paragraph passages (internal)", ""),
        ],
        "metrics": [],
    },
    "2.9": {
        "summary": "Language acceptance — does each configured TTS language "
                   "synthesize and transcribe back correctly?",
        "what": "The TTS counterpart to 1.13: confirms which configured "
                "languages actually produce usable speech on this deployment, "
                "verified by a Whisper round-trip.",
        "how": "The TTS counterpart to 1.13. For each configured language it "
               "synthesizes a short FLEURS sentence with that language's own voice "
               "and locale, transcribes the result with Whisper (letting Whisper "
               "detect the language), and classifies it ok / wrong_language / "
               "empty / error. 'wrong_language' catches the case where the server "
               "produced fluent speech but in the wrong language — e.g. feeding an "
               "Arabic voice Latin text. It records the round-trip WER and a "
               "sample transcript, and this gate decides which languages 2.1, 2.2 "
               "and 2.3 are allowed to score.",
        "code": [
            _code("tts_language_acceptance.py — run() (probe each configured language)",
                  "eval/tts/tts_language_acceptance.py", 190),
            _code("tts_language_acceptance.py — probe() (synth → Whisper → classify)",
                  "eval/tts/tts_language_acceptance.py", 65),
        ],
        "datasets": [
            _ds("FLEURS probe text per language",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["round-trip WER"],
    },
    "phase0": {
        "summary": "Discovery — connectivity, schema and parameter checks run "
                   "before the scored suite.",
        "what": "A pre-flight that confirms the endpoint is reachable, the "
                "request/response schemas match, parameters behave, and smoke "
                "tests pass.",
        "how": "Run before any scored test, discovery confirms the deployment is "
               "usable and records what it found. Sub-checks probe connectivity, "
               "the STT / streaming / TTS request-response schemas, parameter "
               "behaviour, a handful of smoke syntheses and transcriptions, plus "
               "the language-acceptance summary. Each sub-phase writes to "
               "phase0/discovery.json "
               "immediately, so even a crash mid-run leaves a readable record of "
               "everything verified so far.",
        "code": [
            _code("discovery.py — _run_subphase() (runs + persists each check)",
                  "eval/phase0/discovery.py", 361),
            _code("discovery.py — smoke_tests()",
                  "eval/phase0/discovery.py", 233),
        ],
        "datasets": [],
        "metrics": [],
    },
})


# ----------------------------------------------------------------------------
# CSS — dataviz palette, light default + dark toggle, sidebar layout
# ----------------------------------------------------------------------------

CSS = """
:root{
  --font-sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --headerh:58px;
  --surface:#fbfbf9; --plane:#f4f4f1; --card:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --line:rgba(11,11,11,.10); --axis:#c3c2b7;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --goodtext:#006300;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948;
  --accent:#2a78d6; --accent-ink:#1c5cab; --accent-weak:rgba(42,120,214,.09);
  --ring:rgba(42,120,214,.45);
  --shadow:0 1px 2px rgba(11,11,11,.04),0 2px 8px rgba(11,11,11,.05);
  --shadow-sm:0 1px 2px rgba(11,11,11,.05);
  --shadow-md:0 2px 4px rgba(11,11,11,.05),0 8px 24px rgba(11,11,11,.08);
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --surface:#232322; --plane:#0d0d0d; --card:#1c1b1a;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --line:rgba(255,255,255,.12); --axis:#383835;
  --goodtext:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --accent:#3987e5; --accent-ink:#86b6ef; --accent-weak:rgba(57,135,229,.16);
  --ring:rgba(134,182,239,.55);
  --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.35);
  --shadow-sm:0 1px 2px rgba(0,0,0,.35);
  --shadow-md:0 2px 6px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.5);
}}
:root[data-theme="dark"]{
  --surface:#232322; --plane:#0d0d0d; --card:#1c1b1a;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --line:rgba(255,255,255,.12); --axis:#383835;
  --goodtext:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --accent:#3987e5; --accent-ink:#86b6ef; --accent-weak:rgba(57,135,229,.16);
  --ring:rgba(134,182,239,.55);
  --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.35);
  --shadow-sm:0 1px 2px rgba(0,0,0,.35);
  --shadow-md:0 2px 6px rgba(0,0,0,.4),0 10px 28px rgba(0,0,0,.5);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font:15px/1.6 var(--font-sans);color:var(--ink);
  margin:0;background:var(--plane);-webkit-font-smoothing:antialiased}
a{color:var(--accent-ink)}
:focus-visible{outline:2px solid var(--ring);outline-offset:2px;border-radius:4px}
*:focus:not(:focus-visible){outline:none}
header.top{position:sticky;top:0;z-index:40;background:var(--card);
  border-bottom:1px solid var(--line);min-height:var(--headerh);
  padding:10px 24px;display:flex;align-items:center;gap:14px;
  box-shadow:var(--shadow-sm)}
header.top .brand{margin:0;font-size:15.5px;font-weight:700;letter-spacing:-.01em;
  display:flex;align-items:center;gap:9px;color:var(--ink)}
header.top .brand .dot{width:9px;height:9px;border-radius:50%;flex:none;
  background:var(--accent);box-shadow:0 0 0 3px var(--accent-weak)}
header.top .meta{color:var(--muted);font-size:12.5px;line-height:1.5}
header.top .spacer{flex:1}
.hbtn{font:inherit;font-size:13px;font-weight:600;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);
  padding:8px 13px;border-radius:9px;text-decoration:none;display:inline-flex;
  align-items:center;gap:7px;min-height:38px}
.hbtn svg{width:16px;height:16px;flex:none}
.hbtn:hover{border-color:var(--accent);color:var(--accent-ink)}
.themeBtn .icon-sun{display:none}
[data-theme="dark"] .themeBtn .icon-sun{display:inline}
[data-theme="dark"] .themeBtn .icon-moon{display:none}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .themeBtn .icon-sun{display:inline}
  :root:not([data-theme="light"]) .themeBtn .icon-moon{display:none}
}
.hamburger{display:none;align-items:center;justify-content:center;
  width:42px;height:42px;padding:0;border:1px solid var(--line);border-radius:9px;
  background:var(--surface);color:var(--ink);cursor:pointer}
.hamburger svg{width:20px;height:20px}
.hamburger:hover{border-color:var(--accent);color:var(--accent-ink)}
.layout{display:flex;align-items:flex-start;max-width:1320px;margin:0 auto}
nav.side{position:sticky;top:var(--headerh);align-self:flex-start;width:264px;flex:none;
  height:calc(100vh - var(--headerh));overflow-y:auto;padding:20px 14px 40px;
  border-right:1px solid var(--line)}
nav.side .navclose{display:none}
nav.side .grp{font-size:10.5px;font-weight:700;text-transform:uppercase;
  letter-spacing:.07em;color:var(--muted);margin:18px 10px 6px;
  padding-bottom:6px;border-bottom:1px solid var(--line)}
nav.side a{display:block;color:var(--ink2);text-decoration:none;font-size:13px;
  padding:7px 11px;border-radius:8px;border-left:2px solid transparent;
  transition:background .12s,color .12s}
nav.side a:hover{background:var(--surface);color:var(--ink)}
nav.side a.active{color:var(--accent-ink);border-left-color:var(--accent);
  background:var(--accent-weak);font-weight:600}
.navbackdrop{display:none}
main{flex:1;min-width:0;padding:28px 40px 100px;max-width:980px}
.bandhead{margin:36px 0 10px;padding-bottom:10px;border-bottom:2px solid var(--accent)}
.bandhead:first-child{margin-top:4px}
.bandhead h2{margin:0;font-size:24px;font-weight:800;letter-spacing:-.02em}
.bandhead p{margin:6px 0 0;color:var(--ink2);font-size:14px;max-width:70ch}
section.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px;margin:0 0 22px;box-shadow:var(--shadow);
  scroll-margin-top:calc(var(--headerh) + 14px)}
section.card.hero{padding:26px 28px 24px;margin-top:18px;
  background:linear-gradient(180deg,var(--accent-weak),transparent 120px),var(--card);
  box-shadow:var(--shadow-md)}
.hero .kicker{display:inline-flex;align-items:center;gap:8px;margin:0 0 14px;
  padding:5px 12px 5px 11px;border:1px solid var(--line);border-radius:999px;
  background:var(--surface);font-family:var(--font-mono);font-size:10.5px;
  font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:var(--accent-ink);
  box-shadow:var(--shadow-sm)}
.hero .kicker .lv{width:6px;height:6px;border-radius:50%;background:var(--good);
  box-shadow:0 0 0 3px var(--accent-weak)}
.herometa{display:flex;flex-wrap:wrap;gap:9px;margin:18px 0 2px}
.herometa .mi{display:flex;align-items:center;gap:10px;min-width:0;
  padding:9px 13px;border:1px solid var(--line);border-radius:11px;
  background:var(--surface);box-shadow:var(--shadow-sm)}
.herometa .mi>svg{width:17px;height:17px;flex:none;color:var(--accent);opacity:.85;
  stroke-width:1.8}
.herometa .mi>div{min-width:0;display:flex;flex-direction:column}
.herometa .ml{font-size:9.5px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);line-height:1.2}
.herometa .mv{font-family:var(--font-mono);font-size:12.5px;font-weight:600;
  color:var(--ink);line-height:1.4;word-break:break-word}
.hero .herotitle{margin:0 0 8px;font-size:30px;font-weight:700;letter-spacing:-.025em;
  line-height:1.1}
.hero .summ{margin-top:0}
section.card>h3{margin:0 0 4px;font-size:17px;font-weight:700;display:flex;
  align-items:center;gap:10px;flex-wrap:wrap}
.summ{color:var(--ink2);font-size:14.5px;margin:4px 0 14px;max-width:72ch}
details.dd{border:1px solid var(--line);border-radius:10px;background:var(--surface);
  margin:0 0 16px;overflow:hidden}
details.dd>summary{cursor:pointer;list-style:none;padding:11px 15px;font-size:13px;
  font-weight:600;color:var(--accent-ink);user-select:none;display:flex;
  align-items:center;gap:8px}
details.dd>summary::-webkit-details-marker{display:none}
details.dd>summary::before{content:"▸";font-size:11px;transition:transform .15s}
details.dd[open]>summary::before{transform:rotate(90deg)}
details.dd .body{padding:4px 16px 16px;border-top:1px solid var(--line)}
details.dd h4{margin:14px 0 4px;font-size:12px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted)}
details.dd p{margin:0 0 8px;font-size:13.5px;color:var(--ink2)}
details.dd .dslist{display:flex;flex-wrap:wrap;gap:8px;margin:2px 0}
details.dd .dslist a,.dslist .dsflat{font-size:12.5px;background:var(--card);
  border:1px solid var(--line);border-radius:999px;padding:4px 11px;
  text-decoration:none;color:var(--accent-ink)}
.dslist .dsflat{color:var(--muted)}
.dslist.code a{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.mdef{font-size:12.5px;color:var(--ink2);margin:2px 0}
.mdef b{color:var(--ink)}
details.dd .ddnote{font-size:13px;color:var(--ink2);background:var(--surface);
  border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:6px;padding:10px 13px;margin:2px 0 8px}
details.dd .ddnote b{color:var(--ink)}
.chartbox{position:relative;margin:4px 0 16px;width:100%}
.vtoggle{margin-left:auto;display:inline-flex;align-self:center;
  border:1px solid var(--line);border-radius:999px;overflow:hidden;
  background:var(--surface)}
.vtoggle button{appearance:none;-webkit-appearance:none;border:0;background:transparent;
  cursor:pointer;font:600 11.5px var(--font-sans);
  color:var(--ink2);padding:5px 14px;letter-spacing:.02em;line-height:1.4}
.vtoggle button.active{background:var(--accent);color:#fff}
.vtoggle button:not(.active):hover{color:var(--ink)}
.views .view-table{display:none}
.views[data-view=table] .view-chart{display:none}
.views[data-view=table] .view-table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch}
.heat{display:grid;gap:3px;margin:10px 0 6px;overflow-x:auto;
  font:600 11.5px var(--font-mono);font-variant-numeric:tabular-nums}
.heat .hcell{padding:8px 9px;border-radius:5px;text-align:center;white-space:nowrap;
  min-width:52px}
.heat .hhead{background:transparent;color:var(--muted);font-weight:700;
  font-size:11px;text-transform:uppercase;letter-spacing:.03em;align-self:end;padding-bottom:4px}
.heat .hlabel{background:transparent;color:var(--ink);text-align:right;
  padding-right:12px;font-weight:600}
.heat .hna{background:var(--surface);color:var(--muted);font-weight:600}
.heatleg{display:flex;align-items:center;gap:8px;margin:2px 0 4px;
  color:var(--muted);font-size:11px}
.heatleg i{display:inline-block;height:9px;width:120px;border-radius:999px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:7px 10px;text-align:left}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.03em;border-bottom:1px solid var(--axis)}
tbody tr:hover td{background:var(--surface)}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0 4px}
.stat{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);
  border-left:3px solid var(--accent);border-radius:11px;padding:14px 16px;
  box-shadow:var(--shadow-sm)}
.stat .big{font-family:var(--font-mono);font-size:27px;font-weight:700;
  letter-spacing:-.02em;font-variant-numeric:tabular-nums;line-height:1.1}
.stat .lbl{color:var(--muted);font-size:12px;margin-top:5px;line-height:1.35}
.stat.pass{border-left-color:var(--good)} .stat.pass .big{color:var(--goodtext)}
.stat.fail{border-left-color:var(--crit)} .stat.fail .big{color:var(--crit)}
.stat.warn{border-left-color:var(--serious)} .stat.warn .big{color:var(--serious)}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:999px;font-size:11px;font-weight:700;letter-spacing:.02em}
.badge::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor}
.badge.pass{background:rgba(12,163,12,.12);color:var(--goodtext)}
.badge.fail{background:rgba(208,59,59,.13);color:var(--crit)}
.badge.warn{background:rgba(236,131,90,.15);color:var(--serious)}
.badge.muted{background:var(--surface);color:var(--muted)}
.notice{border-radius:10px;padding:12px 15px;font-size:13px;margin:10px 0}
.notice.muted{background:var(--surface);color:var(--ink2);border:1px solid var(--line)}
.notice.warn{background:rgba(236,131,90,.10);color:var(--ink2);
  border:1px solid rgba(236,131,90,.35)}
.notice.fail{background:rgba(208,59,59,.08);color:var(--ink2);
  border:1px solid rgba(208,59,59,.30)}
.notice b{color:var(--ink)}
code{background:var(--surface);border:1px solid var(--line);padding:1px 5px;
  border-radius:5px;font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
footer{color:var(--muted);font-size:12px;text-align:center;padding:40px 20px;
  border-top:1px solid var(--line);margin-top:30px}
@media (max-width:900px){
  .hamburger{display:inline-flex}
  nav.side{position:fixed;top:0;left:0;z-index:60;width:300px;max-width:84vw;
    height:100vh;background:var(--card);border-right:1px solid var(--line);
    box-shadow:var(--shadow-md);padding:18px 16px 40px;
    transform:translateX(-100%);transition:transform .22s ease;
    overscroll-behavior:contain}
  nav.side.open{transform:translateX(0)}
  nav.side .navclose{display:inline-flex;align-items:center;justify-content:center;
    position:absolute;top:12px;right:12px;width:40px;height:40px;padding:0;
    border:1px solid var(--line);border-radius:9px;background:var(--surface);
    color:var(--ink);cursor:pointer}
  nav.side .navclose svg{width:18px;height:18px}
  nav.side .grp:first-of-type{margin-top:36px}
  .navbackdrop.show{display:block;position:fixed;inset:0;z-index:55;
    background:rgba(0,0,0,.45);backdrop-filter:blur(1px)}
  body.navopen{overflow:hidden}
  main{padding:20px 18px 80px;max-width:100%}
  header.top{padding:8px 16px}
  .hero .herotitle{font-size:24px}
  .herometa .mi{flex:1 1 100%}
  section.card{padding:18px 17px}
  section.card table{display:block;overflow-x:auto;-webkit-overflow-scrolling:touch}
  header.top .hbtn span{display:none}
  header.top .hbtn{width:42px;height:42px;min-height:42px;padding:0;justify-content:center}
  header.top .hbtn svg{width:17px;height:17px}
}
@media (max-width:560px){
  header.top .brand{display:none}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  nav.side{transition:none}
  nav.side a{transition:none}
}
"""


# ----------------------------------------------------------------------------
# presentation helpers
# ----------------------------------------------------------------------------


def table(rows: list[dict], cols: list[tuple[str, str]], highlight=None) -> str:
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    head = "".join(f"<th>{esc(h)}</th>" for _, h in cols)
    body = []
    for r in rows:
        style = f' style="{highlight(r)}"' if highlight else ""
        tds = "".join(f"<td>{r.get(k, '')}</td>" for k, _ in cols)
        body.append(f"<tr{style}>{tds}</tr>")
    return (f"<table><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>")


def notice(msg: str, kind: str = "muted") -> str:
    return f'<p class="notice {kind}">{msg if "<" in msg else esc(msg)}</p>'


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def card(big: str, label: str, kind: str = "") -> str:
    return (f'<div class="stat {kind}"><div class="big">{esc(big)}</div>'
            f'<div class="lbl">{esc(label)}</div></div>')


def deepdive(meta: dict) -> str:
    """The expandable 'How this test works' block."""
    if not meta:
        return ""
    parts = ['<details class="dd"><summary>How this test works — what, how, '
             'dataset &amp; metrics</summary><div class="body">']
    if meta.get("what"):
        parts.append(f'<h4>What it measures</h4><p>{esc(meta["what"])}</p>')
    if meta.get("how"):
        parts.append(f'<h4>How the code does it</h4><p>{esc(meta["how"])}</p>')
    ds = meta.get("datasets") or []
    if ds:
        chips = []
        for d in ds:
            if d.get("url"):
                chips.append(f'<a href="{esc(d["url"])}" target="_blank" '
                             f'rel="noopener">{esc(d["name"])} ↗</a>')
            else:
                chips.append(f'<span class="dsflat">{esc(d["name"])}</span>')
        parts.append('<h4>Dataset</h4><div class="dslist">'
                     + "".join(chips) + "</div>")
    code = meta.get("code") or []
    if code:
        clinks = [f'<a href="{esc(c["url"])}" target="_blank" rel="noopener">'
                  f'{esc(c["name"])} ↗</a>' for c in code if c.get("url")]
        if clinks:
            parts.append('<h4>Code</h4><div class="dslist code">'
                         + "".join(clinks) + "</div>")
    mk = [m for m in (meta.get("metrics") or []) if m in GLOSSARY]
    if mk:
        parts.append("<h4>Metrics in this test</h4>")
        for m in mk:
            parts.append(f'<p class="mdef"><b>{esc(m)}</b> — {esc(GLOSSARY[m])}</p>')
    if meta.get("note"):
        parts.append(f'<h4>Note</h4><p class="ddnote">{esc(meta["note"])}</p>')
    parts.append("</div></details>")
    return "".join(parts)


def views(chart_html: str, table_html: str):
    """Mark a card body as having a chart view and a table view.

    ``test_card`` renders a Chart/Table toggle in the header when both views are
    non-empty (chart shown by default); if only one is non-empty it shows that
    single view with no toggle. Plain strings bypass this entirely."""
    return ("__views__", chart_html or "", table_html or "")


def _body_views(body) -> tuple[str, str]:
    """Return (header_toggle_html, body_html) for a card body."""
    if isinstance(body, tuple) and body and body[0] == "__views__":
        chart_html, table_html = body[1], body[2]
        has_c = bool(chart_html and chart_html.strip())
        has_t = bool(table_html and table_html.strip())
        if has_c and has_t:
            toggle = (
                '<span class="vtoggle" role="group" aria-label="Show as chart or table">'
                '<button type="button" class="active" aria-pressed="true" '
                "onclick=\"toggleView(this,'chart')\">Chart</button>"
                '<button type="button" aria-pressed="false" '
                "onclick=\"toggleView(this,'table')\">Table</button></span>")
            html = ('<div class="views" data-view="chart">'
                    f'<div class="view view-chart">{chart_html}</div>'
                    f'<div class="view view-table">{table_html}</div></div>')
            return toggle, html
        return "", (chart_html if has_c else table_html)
    return "", (body or "")


def test_card(anchor: str, title: str, meta: dict, chip_html: str,
              body) -> str:
    summ = (f'<p class="summ">{esc(meta.get("summary", ""))}</p>'
            if meta.get("summary") else "")
    toggle, body_html = _body_views(body)
    return (f'<section class="card" id="{anchor}"><h3>{esc(title)}{chip_html}{toggle}</h3>'
            f'{summ}{deepdive(meta)}{body_html}</section>')


def band(anchor: str, title: str, subtitle: str) -> str:
    return (f'<div class="bandhead" id="{anchor}"><h2>{esc(title)}</h2>'
            f'<p>{esc(subtitle)}</p></div>')

# ----------------------------------------------------------------------------
# chart builders (emit Chart.js specs with "@token" colors resolved client-side)
# ----------------------------------------------------------------------------

_SERIES = ["@s1", "@s2", "@s3", "@s4", "@s5", "@s6", "@s7", "@s8"]


def _axis_titles(x="", y=""):
    sc = {}
    if x:
        sc["x"] = {"title": {"display": True, "text": x}}
    if y:
        sc["y"] = {"title": {"display": True, "text": y}}
    return sc


def bar(cid, labels, values, *, horizontal=False, color="@s1", colors=None,
        series_label="", x_title="", y_title="", height=320):
    if not labels:
        return ""
    ds = {"label": series_label, "data": values,
          "backgroundColor": colors or color, "borderColor": colors or color,
          "borderWidth": 0, "borderRadius": 4, "maxBarThickness": 46}
    opts = {"indexAxis": "y" if horizontal else "x",
            "plugins": {"legend": {"display": False}},
            "scales": _axis_titles(x_title, y_title)}
    if horizontal:
        # match the tooltip to the row the cursor is on — nearest by the category
        # (y) axis only, so a short bar's endpoint can't steal the hover.
        opts["interaction"] = {"mode": "nearest", "axis": "y", "intersect": False}
        height = max(height, 60 + 26 * len(labels))
    return add_chart(cid, "bar", {"labels": labels, "datasets": [ds]}, opts, height)


def grouped_bar(cid, labels, series, *, horizontal=False, x_title="", y_title="",
                height=340):
    """series: list of (name, values)."""
    if not labels:
        return ""
    datasets = [{"label": n, "data": v, "backgroundColor": _SERIES[i % 8],
                 "borderColor": _SERIES[i % 8], "borderRadius": 4,
                 "maxBarThickness": 32}
                for i, (n, v) in enumerate(series)]
    opts = {"indexAxis": "y" if horizontal else "x",
            "plugins": {"legend": {"display": len(series) > 1, "position": "top"}},
            "scales": _axis_titles(x_title, y_title)}
    if horizontal:
        opts["interaction"] = {"mode": "index", "axis": "y", "intersect": False}
        height = max(height, 60 + 30 * len(labels))
    return add_chart(cid, "bar", {"labels": labels, "datasets": datasets}, opts,
                     height)


def lines(cid, series, *, x_title="", y_title="", height=340):
    """series: list of dicts {name, points:[{x,y}], dashed?:bool}."""
    if not series:
        return ""
    datasets = []
    for i, s in enumerate(series):
        c = "@muted" if s.get("dashed") else _SERIES[i % 8]
        datasets.append({"label": s["name"], "data": s["points"],
                         "borderColor": c, "backgroundColor": c,
                         "borderWidth": 2, "borderDash": [6, 4] if s.get("dashed") else [],
                         "pointRadius": 3, "pointHoverRadius": 5, "tension": .25,
                         "fill": False})
    opts = {"plugins": {"legend": {"display": len(series) > 1, "position": "top"}},
            "scales": {"x": dict({"type": "linear"},
                                 **(_axis_titles(x_title).get("x", {}))),
                       **_axis_titles("", y_title)}}
    return add_chart(cid, "line", {"datasets": datasets}, opts, height)


def reliability(cid, conf, acc, height=340):
    pts = [{"x": c, "y": a} for c, a in zip(conf, acc) if c is not None and a is not None]
    if not pts:
        return ""
    diag = [{"x": 0, "y": 0}, {"x": 1, "y": 1}]
    datasets = [
        {"label": "Ideal (perfect calibration)", "data": diag, "borderColor": "@muted",
         "borderDash": [6, 4], "borderWidth": 2, "pointRadius": 0, "fill": False},
        {"label": "Measured", "data": pts, "borderColor": "@s1",
         "backgroundColor": "@s1", "borderWidth": 2, "pointRadius": 4,
         "pointHoverRadius": 6, "tension": .1, "fill": False},
    ]
    opts = {"plugins": {"legend": {"display": True, "position": "top"}},
            "scales": {"x": {"type": "linear", "min": 0, "max": 1,
                             "title": {"display": True, "text": "Confidence"}},
                       "y": {"min": 0, "max": 1,
                             "title": {"display": True, "text": "Accuracy"}}}}
    return add_chart(cid, "line", {"datasets": datasets}, opts, height)


def _verdict_colors(rows, key="verdict", good="PASS"):
    return ["@good" if (r.get(key) or "").upper() == good else "@serious"
            for r in rows]


# ----------------------------------------------------------------------------
# STT render functions — chart (from raw rows) + compact table
# ----------------------------------------------------------------------------


def render_accuracy(rows):
    if not rows:
        return table(rows, [])
    labels = [r.get("dataset", "") for r in rows]
    vals = [fval(r.get("wer"), 4) for r in rows]
    vals = [None if v is None else round(v * 100, 2) for v in vals]
    colors = _verdict_colors(rows)
    chart = bar("c_acc", labels, vals, horizontal=True, colors=colors,
                x_title="WER (%) — lower is better")
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        r["cer"] = pct(r.get("cer"))
        r["mer"] = pct(r.get("mer"))
        v = r.get("verdict", "")
        r["verdict"] = badge(v, "pass" if v == "PASS" else "warn") if v else ""
    cols = [("dataset", "Dataset"), ("n_utterances", "N"), ("wer", "WER"),
            ("cer", "CER"), ("mer", "MER"), ("verdict", "Verdict")]
    return views(chart, table(rows, cols))


def render_performance(rows, conc):
    chart = ""
    if rows:
        labels = [f'{r.get("bucket_s", "")}s' for r in rows]
        vals = [fval(r.get("rtfx_mean"), 1) for r in rows]
        chart += bar("c_perf", labels, vals, color="@s1",
                     x_title="Audio-duration bucket",
                     y_title="RTFx — higher is faster")
    if conc:
        pts = [{"x": fval(r.get("concurrency"), 0), "y": fval(r.get("p50"), 3)}
               for r in conc if fval(r.get("concurrency"), 0) is not None]
        chart += lines("c_perf_conc",
                       [{"name": "P50 latency (s)", "points": pts}],
                       x_title="Concurrency", y_title="P50 latency (s)")
    for r in rows:
        r["rtfx_mean"] = fnum(r.get("rtfx_mean"), 1)
        r["rtf_mean"] = fnum(r.get("rtf_mean"), 4)
        r["latency_p50"] = fnum(r.get("latency_p50"), 3)
        r["latency_p99"] = fnum(r.get("latency_p99"), 3)
    cols = [("bucket_s", "Bucket (s)"), ("n_calls", "N"), ("rtfx_mean", "RTFx"),
            ("rtf_mean", "RTF"), ("latency_p50", "P50 (s)"), ("latency_p99", "P99 (s)")]
    tbl = "<h4>Throughput by audio duration</h4>" + table(rows, cols)
    if conc:
        for r in conc:
            r["rtf_mean"] = fnum(r.get("rtf_mean"), 4)
            r["p50"] = fnum(r.get("p50"), 3)
            r["p99"] = fnum(r.get("p99"), 3)
        cc = [("concurrency", "Concurrency"), ("n_calls", "N"), ("p50", "P50 (s)"),
              ("p99", "P99 (s)"), ("rtf_mean", "RTF")]
        tbl += "<h4>Concurrency</h4>" + table(conc, cc)
    return views(chart, tbl)


def render_streaming(rows):
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    r = rows[0]
    tiles = []
    bar_lbl, bar_val, ptbl = [], [], []
    # True latencies (seconds) — these share the bar's seconds axis.
    for key, lbl in [("ttfw", "Time-to-first-word"),
                     ("finalization_latency", "Finalization latency"),
                     ("partial_latency", "Partial latency")]:
        d = maybe_literal(r.get(key))
        if isinstance(d, dict):
            tiles.append(card(f"{fnum(d.get('p50'), 3)}s", f"{lbl} P50", ""))
            bar_lbl.append(lbl)
            bar_val.append(fval(d.get("p50"), 3))
            ptbl.append({"metric": lbl, "unit": "seconds", "p50": fnum(d.get("p50"), 3),
                         "p90": fnum(d.get("p90"), 3), "p95": fnum(d.get("p95"), 3),
                         "p99": fnum(d.get("p99"), 3)})
    # Streaming RTF is a ratio (×realtime), NOT a latency — keep it OFF the
    # seconds axis so the bar stays unit-consistent; surface it as a KPI tile and
    # a table row instead.
    rtf = maybe_literal(r.get("streaming_rtf"))
    if isinstance(rtf, dict):
        tiles.append(card(fnum(rtf.get("p50"), 3), "Streaming RTF P50 (×realtime)", ""))
        ptbl.append({"metric": "Streaming RTF", "unit": "×realtime",
                     "p50": fnum(rtf.get("p50"), 3), "p90": fnum(rtf.get("p90"), 3),
                     "p95": fnum(rtf.get("p95"), 3), "p99": fnum(rtf.get("p99"), 3)})
    tiles.append(card(pct(r.get("stability_rate_mean")), "Stability rate", "pass"))
    tiles.append(card(pct(r.get("batch_wer")), "Batch WER", ""))
    tiles.append(card(pct(r.get("stream_wer")), "Stream WER", ""))
    tiles_html = '<div class="cards">' + "".join(tiles) + "</div>"
    chart = bar("c_stream", bar_lbl, bar_val, horizontal=True, color="@s1",
                x_title="P50 latency (s) — lower is better") if bar_lbl else ""
    cols = [("metric", "Metric"), ("unit", "Unit"), ("p50", "P50"), ("p90", "P90"),
            ("p95", "P95"), ("p99", "P99")]
    tbl = table(ptbl, cols) if ptbl else ""
    return views(tiles_html + chart, tiles_html + tbl)


def render_rest_grpc(rows):
    if not rows:
        return notice(
            "N/A — this deployment serves gRPC StreamingRecognize only; there is "
            "no REST transport to compare. Recorded as N/A rather than fabricated.",
            "warn")
    for r in rows:
        r["p50"] = fnum(r.get("p50"), 4)
        r["p99"] = fnum(r.get("p99"), 4)
    cols = [("mode", "Mode"), ("interface", "Interface"), ("p50", "P50 (s)"),
            ("p99", "P99 (s)")]
    return table(rows, cols)


def render_noise(rows):
    if not rows:
        return table(rows, [])
    # aggregate rows (per noise_type+snr, no per-language split): empty language
    agg = [r for r in rows if not (r.get("language") or "").strip()]
    src = agg or rows
    by_type: dict[str, list] = {}
    for r in src:
        nt = (r.get("noise_type") or "").strip() or "noise"
        snr = r.get("snr_db")
        try:
            x = float(snr)
        except (TypeError, ValueError):
            x = 40.0 if "inf" in str(snr).lower() else None  # clean baseline anchor
        w = fval(r.get("wer"), 4)
        if x is None or w is None:
            continue
        by_type.setdefault(nt, []).append({"x": x, "y": round(w * 100, 2)})
    series = []
    for nt, pts in by_type.items():
        pts.sort(key=lambda p: p["x"])
        series.append({"name": nt, "points": pts, "dashed": nt.lower() == "clean"})
    chart = lines("c_noise", series, x_title="SNR (dB) — higher is cleaner",
                  y_title="WER (%)") if series else ""
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        dr = r.get("delta_rel")
        r["delta_rel"] = ("+" + pct(dr)) if dr not in (None, "", "0.0") else "baseline"
    cols = [("noise_type", "Noise"), ("snr_db", "SNR (dB)"), ("wer", "WER"),
            ("delta_rel", "Δ vs clean"), ("n_utterances", "N")]
    return views(chart, table(rows, cols))


def render_accent(rows):
    if not rows:
        return table(rows, [])
    labels = [r.get("accent", "") for r in rows]
    vals = [None if fval(r.get("wer"), 4) is None else round(fval(r.get("wer"), 4) * 100, 2)
            for r in rows]
    chart = bar("c_accent", labels, vals, horizontal=True, color="@s3",
                x_title="WER (%) — lower is better")
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        r["fairness_delta"] = pct(r.get("fairness_delta"))
    cols = [("accent", "Accent"), ("source", "Source"), ("n_utterances", "N"),
            ("wer", "WER"), ("fairness_delta", "Fairness Δ")]
    return views(chart, table(rows, cols))


def render_longform(rows):
    if not rows:
        return table(rows, [])
    show_chart = 1 <= len(rows) <= 24
    chart = ""
    if show_chart:
        labels = [str(r.get("id", ""))[:18] for r in rows]
        full = [None if fval(r.get("full_wer"), 4) is None
                else round(fval(r.get("full_wer"), 4) * 100, 2) for r in rows]
        chunk = [None if fval(r.get("chunked_wer"), 4) is None
                 else round(fval(r.get("chunked_wer"), 4) * 100, 2) for r in rows]
        chart = grouped_bar("c_lf", labels,
                            [("Full-file WER", full), ("Chunked WER", chunk)],
                            horizontal=True, x_title="WER (%)")
    for r in rows:
        r["full_wer"] = pct(r.get("full_wer"))
        r["chunked_wer"] = pct(r.get("chunked_wer"))
        r["hallucination_rate"] = pct(r.get("hallucination_rate"))
        r["repetition_rate"] = pct(r.get("repetition_rate"))
        r["duration_s"] = fnum(r.get("duration_s"), 1)
    cols = [("id", "File"), ("duration_s", "Dur (s)"), ("full_wer", "Full WER"),
            ("chunked_wer", "Chunked WER"), ("hallucination_rate", "Halluc."),
            ("repetition_rate", "Repet.")]
    tbl = table(rows, cols)
    if len(rows) > 24:
        tbl = (f'<details class="dd"><summary>Per-file results '
               f'({len(rows)} files)</summary><div class="body">{tbl}</div></details>')
    return views(chart, tbl)


def render_domain(rows):
    if not rows:
        return table(rows, [])
    r0 = rows[0]
    tiles = ('<div class="cards">'
             + card(pct(r0.get("domain_wer")), "Domain WER", "")
             + card(pct(r0.get("krr_mean")), "Key-term recall (KRR)", "pass")
             + card(pct(r0.get("ne_wer_mean")), "Named-entity WER", "")
             + "</div>") if len(rows) == 1 else ""
    top = (r0.get("top_misrecognised") or "")
    toplist = (f'<h4>Most-misrecognised terms</h4><p class="mdef">{esc(top[:220])}</p>'
               if top else "")
    for r in rows:
        r["domain_wer"] = pct(r.get("domain_wer"))
        r["krr_mean"] = pct(r.get("krr_mean"))
        r["ne_wer_mean"] = pct(r.get("ne_wer_mean"))
    cols = [("dataset", "Dataset"), ("n_utterances", "N"), ("domain_wer", "Domain WER"),
            ("krr_mean", "Key-term recall"), ("ne_wer_mean", "NE WER")]
    return views(tiles, table(rows, cols) + toplist)


LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ko": "Korean",
    "el": "Greek", "ja": "Japanese", "zh": "Chinese", "vi": "Vietnamese",
    "cs": "Czech", "it": "Italian", "tr": "Turkish", "th": "Thai",
    "id": "Indonesian", "ar": "Arabic", "ru": "Russian", "uk": "Ukrainian",
}

# Sequential single-hue ramp (indigo, the @s1 family) for the output-quality
# heatmap: 0 → very light tint, 1 → deep indigo. Baked to hex because the grid
# is static HTML (not a Chart.js canvas resolving @tokens); each cell carries
# its own fill, so it reads the same in light and dark mode, and the text
# colour is chosen per cell for contrast.
_SEQ_LO_HEX = "#eef1fb"
_SEQ_HI_HEX = "#2b3aa8"


def _seq_color(v: float) -> tuple[str, str]:
    """Map a 0..1 score to (background_hex, text_hex) on the indigo ramp."""
    v = max(0.0, min(1.0, float(v)))
    lo = (0xee, 0xf1, 0xfb)
    hi = (0x2b, 0x3a, 0xa8)
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * v) for i in range(3))
    lum = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255
    fg = "#15182e" if lum > 0.62 else "#ffffff"
    return "#%02x%02x%02x" % rgb, fg


def render_output_quality(rows):
    if not rows:
        return table(rows, [])

    # The data is a language × dimension score matrix (period/comma F1,
    # capitalisation, number formatting per language). The old chart plotted
    # every (metric, language) row as its own bar — ~100 bars, unreadable and
    # far too long. A heatmap is the right form for a two-category × one-measure
    # matrix: one screen, and "which language is weak on which dimension" reads
    # at a glance. Exclamation/question F1 are intentionally left OUT of the
    # heatmap (see note) and kept in the Table view for transparency.
    def _colkey(metric: str) -> str | None:
        m = metric or ""
        if "Capitalisation" in m:
            return "caps"
        if "Number Formatting" in m:
            return "num"
        if "F1" in m:
            if "(.)" in m or "(。)" in m:
                return "period"
            if "(,)" in m or "(，)" in m or "(、)" in m:
                return "comma"
        return None  # ? ! ¿ ¡ ; : · — untested marks, excluded from heatmap

    by_lang: dict[str, dict] = {}
    order: list[str] = []
    overall: dict[str, float] = {}
    for r in rows:
        lang = (r.get("language") or "").strip()
        key = _colkey(r.get("metric", ""))
        if key is None:
            continue
        v = fval(r.get("value"), 4)
        if not lang:                      # aggregate (all-language) row
            if v is not None:
                overall[key] = v
            continue
        if lang not in by_lang:
            by_lang[lang] = {}
            order.append(lang)
        # keep the first non-None value seen for a (lang, column)
        if key not in by_lang[lang] and v is not None:
            by_lang[lang][key] = v

    # drop languages with no scorable punctuation/caps/number data (no_output)
    order = [c for c in order if by_lang.get(c)]

    cols_def = [("period", "Period ."), ("comma", "Comma ,"),
                ("caps", "Capitals"), ("num", "Numbers")]

    # headline tiles from the all-language aggregate
    tiles = ""
    if overall:
        tiles = ('<div class="cards">'
                 + card(pct(overall.get("caps")), "Capitalisation (overall)", "")
                 + card(pct(overall.get("num")), "Number formatting (overall)", "")
                 + card(fnum(overall.get("period"), 2), "Period F1 (overall)", "")
                 + card(fnum(overall.get("comma"), 2), "Comma F1 (overall)", "")
                 + "</div>")

    # heatmap grid
    cells = ['<div class="hcell hhead"></div>']
    for _, h in cols_def:
        cells.append(f'<div class="hcell hhead">{esc(h)}</div>')
    for code in order:
        name = LANG_NAMES.get(code, code.upper())
        cells.append(f'<div class="hcell hlabel">{esc(name)}</div>')
        for key, hlabel in cols_def:
            v = by_lang[code].get(key)
            if v is None:
                cells.append('<div class="hcell hna" title="not applicable '
                             'for this script">n/a</div>')
            else:
                bg, fg = _seq_color(v)
                cells.append(
                    f'<div class="hcell" style="background:{bg};color:{fg}" '
                    f'title="{esc(name)} — {esc(hlabel)}: {v * 100:.0f}%">'
                    f'{v * 100:.0f}</div>')
    c_lo, c_hi = _SEQ_LO_HEX, _SEQ_HI_HEX
    legend = (f'<div class="heatleg"><span>0</span>'
              f'<i style="background:linear-gradient(90deg,{c_lo},{c_hi})"></i>'
              f'<span>100 &nbsp;·&nbsp; score % (period/comma shown as F1×100), '
              f'darker = higher = better</span></div>')
    heat = (f'<div class="heat" style="grid-template-columns:'
            f'minmax(104px,1.1fr) repeat({len(cols_def)},1fr)">'
            + "".join(cells) + "</div>")

    note = notice(
        "<b>Exclamation (!) and question (?) marks are omitted from the heatmap.</b> "
        "FLEURS read speech is declarative — a 200-clip set contains essentially "
        "no '!' and at most one '?', so their F1 is 0 for lack of any reference "
        "mark to detect, not because the model fails to emit them. The full "
        "per-mark breakdown (including ! and ?) is in the Table view.", "muted")

    chart_view = tiles + legend + heat + note

    cols = [("language", "Lang"), ("metric", "Metric"),
            ("value", "Value"), ("n_samples", "N")]
    table_rows = [dict(r, language=(r.get("language") or "all")) for r in rows]
    return views(chart_view, table(table_rows, cols))


def render_confidence(rows):
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    r = rows[0]
    cards_html = (
        '<div class="cards">'
        + card(fnum(r.get("ece"), 3), "ECE (lower=better)", "warn")
        + card(fnum(r.get("auroc"), 3), "AUROC", "")
        + card(fnum(r.get("brier_score"), 3), "Brier", "")
        + card(esc(r.get("n_words_evaluated")), "words evaluated", "")
        + "</div>")
    bins = maybe_literal(r.get("bins"))
    chart = ""
    btbl = ""
    if isinstance(bins, list) and bins:
        conf = [fval(b.get("confidence"), 4) for b in bins]
        acc = [fval(b.get("accuracy"), 4) for b in bins]
        chart = reliability("c_conf", conf, acc)
        for b in bins:
            b["accuracy"] = fnum(b.get("accuracy"), 3)
            b["confidence"] = fnum(b.get("confidence"), 3)
            b["gap"] = fnum(b.get("gap"), 3)
        btbl = "<h4>Reliability bins</h4>" + table(
            bins, [("bin", "Bin"), ("n", "N"), ("confidence", "Mean conf."),
                   ("accuracy", "Accuracy"), ("gap", "Gap")])
    return views(cards_html + chart, btbl)


def render_format(rows):
    if not rows:
        return table(rows, [])
    for r in rows:
        r["status"] = badge(r.get("status", "?"),
                            "pass" if r.get("status") == "pass" else "fail")
    cols = [("test", "Test"), ("value", "Value"), ("status", "Status"),
            ("error", "Error")]
    return table(rows, cols)


# NVIDIA model-card FLEURS scores for nemotron-3.5-asr-streaming-0.6b, from the
# published card (huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b):
# auto-detect mode at the 1.12 s chunk (the card's best operating point, matching
# our run). Values are percentages. Same metric per language as our run — WER,
# except Japanese/Korean/Mandarin which the card scores by CER. Greek is omitted
# because the card publishes no number for it (adaptation-ready tier).
NVIDIA_CARD_FLEURS = {
    "en": 8.84, "de": 8.22, "fr": 9.02, "es": 4.13, "pt": 5.47, "nl": 11.70,
    "it": 4.32, "tr": 11.32, "ar": 12.06, "vi": 11.22, "pl": 16.55, "cs": 21.45,
    "ja": 11.66, "zh": 19.87, "ko": 7.30,
}


def render_multilingual(rows):
    if not rows:
        return notice(
            "Full per-language accuracy (1.12) is opt-in and was not run here. "
            "The live 34-language acceptance probe (1.13) is shown below.", "muted")
    # Per-language rows only — no macro-average (dropped by design; a single
    # blended score across tiers misleads more than it informs).
    rows = [r for r in rows
            if r.get("status") != "summary" and r.get("language") != "MACRO-AVG"]
    # Greek is excluded: NVIDIA's model card lists Greek only in the
    # "adaptation-ready" tier (recognised by the tokenizer, not tuned for
    # production) and publishes no benchmark for it. Our ~50% WER reflects that
    # untuned state, so charting it beside production languages misleads. We keep
    # it out of the chart/table and explain why in a note below.
    greek = [r for r in rows if r.get("key") == "el"]
    rows = [r for r in rows if r.get("key") != "el"]
    labels = [r.get("language", r.get("key", "")) for r in rows]
    ours = [None if fval(r.get("primary_value"), 4) is None
            else round(fval(r.get("primary_value"), 4) * 100, 2) for r in rows]
    # Second series = NVIDIA's published card figure (None where the card has no
    # number — Chart.js simply leaves that bar out, so our-only languages still
    # show their single Ours bar). A direct Ours-vs-card comparison is the whole
    # point of this test, so it becomes the default visual.
    card_series = [NVIDIA_CARD_FLEURS.get(r.get("key")) for r in rows]
    chart = grouped_bar(
        "c_ml", labels,
        [("Ours (FLEURS, 200 clips)", ours), ("NVIDIA card (FLEURS)", card_series)],
        horizontal=True, x_title="WER / CER (%) — lower is better")
    disp = []
    for r in rows:
        key = r.get("key")
        card_val = NVIDIA_CARD_FLEURS.get(key)
        disp.append({"language": esc(r.get("language")),
                     "metric": esc((r.get("primary_metric") or r.get("metric") or "").upper()),
                     "primary_value": pct(r.get("primary_value")),
                     "card": ("—" if card_val is None else f"{card_val:.1f}%"),
                     "n": esc(r.get("n_utterances"))})
    cols = [("language", "Language"), ("metric", "Metric"),
            ("primary_value", "Our score"),
            ("card", "NVIDIA card (FLEURS)"), ("n", "N")]
    notes = notice(
        "<b>NVIDIA card</b> figures (the second bar / the right-hand column) are the "
        "model card's FLEURS score for nemotron-3.5-asr-streaming-0.6b, auto-detect "
        "mode at the 1.12&nbsp;s chunk (the card's best operating point, matching our "
        "run). Same metric per language (WER, or CER for Japanese/Korean/Mandarin). "
        "Our numbers run ~1–5&nbsp;pts higher, consistent with our 200-clip FLEURS "
        "subset, the Whisper normaliser, and two locale mismatches (Egyptian Arabic "
        "vs MSA, LatAm vs EU Spanish).", "muted")
    if greek:
        g = greek[0]
        notes += notice(
            "<b>Greek is excluded</b> from the chart and table above. NVIDIA's "
            "model card places Greek in the <i>adaptation-ready</i> tier "
            "(recognised by the tokenizer but not tuned for production) and "
            "publishes no accuracy number for it. Our run measured Greek at "
            f"WER&nbsp;{pct(g.get('primary_value'))} — consistent with an untuned "
            "language: the transcript is phonetically close (CER&nbsp;"
            f"{pct(g.get('cer'))}) but inflectional word-endings are missed, which "
            "WER counts as whole-word errors. Shown here for transparency, kept out "
            "of the comparison because there is no card figure to compare against.",
            "muted")
    return views(chart + notes, table(disp, cols) + notes)

# ----------------------------------------------------------------------------
# TTS render functions
# ----------------------------------------------------------------------------


def _per_lang_rows(test_dir: Path) -> list[dict]:
    rows: list[dict] = []
    if not test_dir.exists():
        return rows
    for f in sorted(test_dir.glob("summary_*.csv")):
        rows.extend(read_csv(f))
    return rows


def render_percentile_table(rows, label_key, label_hdr, metric_keys):
    out = []
    for r in rows:
        rr = {label_hdr: esc(r.get(label_key))}
        for mk, mh in metric_keys:
            d = maybe_literal(r.get(mk))
            if isinstance(d, dict):
                rr[mh] = f"P50 {fnum(d.get('p50'), 3)} / P99 {fnum(d.get('p99'), 3)}"
            else:
                rr[mh] = fnum(d, 3)
        out.append(rr)
    cols = [(label_hdr, label_hdr)] + [(mh, mh) for _, mh in metric_keys]
    return table(out, cols)


def render_naturalness(rows, lang_rows=None):
    if not rows:
        base = notice("No data on disk for this test in the selected run.")
    else:
        r = rows[0]
        parts = [card(esc(r.get("n_sentences")), "sentences (en)", "")]
        for key, lbl in [("utmos", "UTMOS (en)"), ("dnsmos_ovrl", "DNSMOS OVRL"),
                         ("dnsmos_sig", "DNSMOS SIG"), ("dnsmos_bak", "DNSMOS BAK")]:
            d = maybe_literal(r.get(key))
            if isinstance(d, dict):
                parts.append(card(fnum(d.get("mean"), 2), lbl,
                                  "pass" if lbl.startswith("UTMOS") else ""))
        base = '<div class="cards">' + "".join(parts) + "</div>"

    if not lang_rows:
        return base

    labels, vals, out = [], [], []
    for r in lang_rows:
        if r.get("status") in ("no_output", "load_failed"):
            out.append({"language": esc(r.get("language")),
                        "n_sentences": esc(r.get("n_sentences")),
                        "dnsmos_ovrl": badge(r.get("status"), "warn"),
                        "dnsmos_sig": "—", "dnsmos_bak": "—", "utmos": "—"})
            continue
        rr = {"language": esc(r.get("language")),
              "n_sentences": esc(r.get("n_sentences")), "utmos": "n/a (en-only)"}
        for k in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
            d = maybe_literal(r.get(k))
            rr[k] = fnum(d.get("mean"), 2) if isinstance(d, dict) else "—"
        out.append(rr)
        d = maybe_literal(r.get("dnsmos_ovrl"))
        if isinstance(d, dict):
            labels.append(r.get("language"))
            vals.append(fval(d.get("mean"), 2))
    chart = bar("c_nat", labels, vals, horizontal=True, color="@s3",
                x_title="DNSMOS OVRL (1–5, higher is better)") if labels else ""
    cols = [("language", "Lang"), ("n_sentences", "N"), ("dnsmos_ovrl", "DNSMOS OVRL"),
            ("dnsmos_sig", "DNSMOS SIG"), ("dnsmos_bak", "DNSMOS BAK"), ("utmos", "UTMOS")]
    caveat = notice(
        "Per-language naturalness uses DNSMOS (language-agnostic). UTMOS is "
        "omitted for non-English — it was validated on English only, so it is "
        "not a valid score for other languages.", "muted")
    header = "<h4>Per-language (multilingual)</h4>"
    return views(base + header + chart + caveat,
                 base + header + table(out, cols) + caveat)


def render_intelligibility(rows, lang_rows=None):
    cat_chart = ""
    if rows:
        labels = [r.get("category", "") for r in rows]
        vals = [None if fval(r.get("round_trip_wer"), 4) is None
                else round(fval(r.get("round_trip_wer"), 4) * 100, 2) for r in rows]
        cat_chart = bar("c_intel", labels, vals, color="@s1",
                        x_title="Category",
                        y_title="Round-trip WER (%) — lower is better")
    for r in rows:
        r["round_trip_wer"] = pct(r.get("round_trip_wer"))
        r["round_trip_cer"] = pct(r.get("round_trip_cer"))
    cols = [("category", "Category"), ("n_sentences", "N"),
            ("round_trip_wer", "Round-trip WER"), ("round_trip_cer", "CER"),
            ("normalizer", "Normalizer")]
    cat_tbl = table(rows, cols)

    if not lang_rows:
        return views(cat_chart, cat_tbl)

    labels, vals, out = [], [], []
    for r in lang_rows:
        if r.get("status") in ("no_output", "load_failed"):
            out.append({"language": esc(r.get("language")),
                        "n_sentences": esc(r.get("n_sentences")),
                        "round_trip_wer": badge(r.get("status"), "warn")})
            continue
        out.append({"language": esc(r.get("language")),
                    "n_sentences": esc(r.get("n_sentences")),
                    "round_trip_wer": pct(r.get("round_trip_wer"))})
        w = fval(r.get("round_trip_wer"), 4)
        if w is not None:
            labels.append(r.get("language"))
            vals.append(round(w * 100, 2))
    lchart = bar("c_intel_lang", labels, vals, horizontal=True, color="@s2",
                 x_title="Round-trip WER (%) — lower is better") if labels else ""
    cols2 = [("language", "Lang"), ("n_sentences", "N"),
             ("round_trip_wer", "Round-trip WER")]
    note = notice("FLEURS text synthesised per language, transcribed with "
                  "Whisper large-v3 using the language hint.", "muted")
    h = "<h4>Per-language (FLEURS round-trip)</h4>"
    return views(cat_chart + h + lchart + note,
                 cat_tbl + h + table(out, cols2) + note)


def render_prosody(rows, lang_rows=None):
    all_rows = []
    for r in rows:
        rr = dict(r)
        rr.setdefault("language", "en")
        all_rows.append(rr)
    all_rows.extend(lang_rows or [])
    labels, vals, display = [], [], []
    for r in all_rows:
        if r.get("status") in ("no_output", "load_failed"):
            display.append({"language": esc(r.get("language")),
                            "status": badge(r.get("status"), "warn")})
            continue
        display.append({k: esc(r.get(k)) for k in
                        ("language", "f0_mean_hz", "f0_std_intra_hz",
                         "f0_range_intra_hz", "npvi", "pause_ratio", "wpm_mean",
                         "wpm_std", "n_sentences")})
        v = fval(r.get("f0_std_intra_hz"), 2)
        if v is not None:
            labels.append(r.get("language"))
            vals.append(v)
    chart = bar("c_pros", labels, vals, horizontal=True, color="@s5",
                x_title="F0 intra-std (Hz) — pitch dynamism") if labels else ""
    cols = [("language", "Lang"), ("f0_mean_hz", "F0 mean (Hz)"),
            ("f0_std_intra_hz", "F0 intra-std (Hz)"),
            ("f0_range_intra_hz", "F0 range (Hz)"), ("npvi", "nPVI"),
            ("pause_ratio", "Pause ratio"), ("wpm_mean", "WPM mean"),
            ("wpm_std", "WPM std"), ("n_sentences", "N")]
    note = notice(
        "Reference-free prosody: pitch dynamism (F0 std/range), rhythm (acoustic "
        "nPVI), pause ratio and speaking rate. Intrusive F0-vs-reference metrics "
        "need a same-speaker ground truth, which a fixed synthetic voice lacks "
        "(see 2.4).", "muted")
    return views(note + chart, note + table(display, cols))


def render_signal(rows):
    reason = rows[0].get("reason") if rows else None
    msg = reason or (
        "MCD/PESQ/STOI are full-reference (intrusive) metrics requiring a same-"
        "speaker, same-text ground-truth recording. None exists for a fixed "
        "synthetic voice, so any score measures speaker mismatch, not quality. "
        "Use 2.1 (UTMOS/DNSMOS) and 2.2 (round-trip WER) instead; a valid "
        "intrusive run needs a zero-shot voice-cloning deployment.")
    return notice("N/A for this model — " + msg, "warn")


def render_tts_latency(rows):
    chart = ""
    if rows:
        labels = [r.get("bucket", "") for r in rows]
        vals = []
        for r in rows:
            d = maybe_literal(r.get("ttfb"))
            vals.append(fval(d.get("p50"), 3) if isinstance(d, dict) else fval(r.get("ttfb"), 3))
        if any(v is not None for v in vals):
            chart = bar("c_ttslat", labels, vals, color="@s1",
                        x_title="Text-length bucket", y_title="TTFB P50 (s)")
    return views(chart, render_percentile_table(
        rows, "bucket", "Bucket",
        [("ttfb", "TTFB (s)"), ("rtf", "RTF"), ("elapsed", "Elapsed (s)")]))


def render_tts_concurrency(rows):
    chart = ""
    if rows:
        rps_pts = [{"x": fval(r.get("concurrency"), 0), "y": fval(r.get("rps"), 3)}
                   for r in rows if fval(r.get("concurrency"), 0) is not None]
        lat_pts = [{"x": fval(r.get("concurrency"), 0), "y": fval(r.get("latency_p50"), 3)}
                   for r in rows if fval(r.get("concurrency"), 0) is not None]
        chart = (lines("c_ttsrps", [{"name": "Requests/sec", "points": rps_pts}],
                       x_title="Concurrency", y_title="RPS")
                 + lines("c_ttsconclat",
                         [{"name": "P50 latency (s)", "points": lat_pts}],
                         x_title="Concurrency", y_title="P50 latency (s)"))
    for r in rows:
        r["error_rate"] = pct(r.get("error_rate"))
        r["latency_p50"] = fnum(r.get("latency_p50"), 3)
        r["latency_p99"] = fnum(r.get("latency_p99"), 3)
    cols = [("interface", "Interface"), ("concurrency", "N"), ("n_success", "OK"),
            ("error_rate", "Err rate"), ("rps", "RPS"), ("latency_p50", "P50 (s)"),
            ("latency_p99", "P99 (s)")]
    return views(chart, table(rows, cols))


def render_edge(rows):
    if not rows:
        return table(rows, [])
    labels = [r.get("category", "") for r in rows]
    vals = [None if fval(r.get("pass_rate"), 4) is None
            else round(fval(r.get("pass_rate"), 4) * 100, 1) for r in rows]
    colors = ["@good" if (v or 0) >= 90 else "@serious" if (v or 0) >= 50 else "@crit"
              for v in vals]
    chart = bar("c_edge", labels, vals, horizontal=True, colors=colors,
                x_title="Pass rate (%) — higher is better")
    for r in rows:
        r["pass_rate"] = pct(r.get("pass_rate"))
    cols = [("category", "Category"), ("total", "Total"), ("pass", "Pass"),
            ("fail", "Fail"), ("degraded", "Degraded"), ("pass_rate", "Pass rate")]
    return views(chart, table(rows, cols))


def render_tts_longform(rows):
    chart = ""
    if rows:
        labels = [str(r.get("title", ""))[:22] for r in rows]
        sim = [fval(r.get("mean_speaker_sim"), 3) for r in rows]
        chart = bar("c_ttslf", labels, sim, horizontal=True, color="@s3",
                    x_title="Mean speaker similarity (0–1) — higher is more consistent")
    for r in rows:
        r["mean_speaker_sim"] = fnum(r.get("mean_speaker_sim"), 3)
        r["f0_drift_hz"] = fnum(r.get("f0_drift_hz"), 1)
        r["wpm_drift"] = fnum(r.get("wpm_drift"), 1)
    cols = [("title", "Passage"), ("n_paragraphs", "Paras"),
            ("mean_speaker_sim", "Speaker sim"), ("f0_drift_hz", "F0 drift (Hz)"),
            ("wpm_drift", "WPM drift")]
    return views(chart, table(rows, cols))


def render_tts_acceptance(rows):
    if not rows:
        return notice(
            "No TTS language-acceptance matrix found. Run Test 2.9 with "
            "tts.multilingual_languages set.")
    ok = [r for r in rows if r.get("status") == "ok"]
    cards = ('<div class="cards">'
             + card(f"{len(ok)}/{len(rows)}", "languages synthesise + round-trip",
                    "pass" if len(ok) == len(rows) else "warn")
             + "</div>")
    labels = [r.get("label", r.get("language", "")) for r in rows]
    vals = [None if fval(r.get("round_trip_wer"), 4) is None
            else round(fval(r.get("round_trip_wer"), 4) * 100, 2) for r in rows]
    colors = ["@good" if r.get("status") == "ok" else "@serious" for r in rows]
    chart = bar("c_ttsacc", labels, vals, horizontal=True, colors=colors,
                x_title="Round-trip WER (%) — lower is better") if any(v is not None for v in vals) else ""
    gap = ""
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        names = ", ".join(f'<b>{esc(r.get("label"))}</b> ({esc(r.get("status"))})'
                          for r in bad)
        gap = notice(f"Serving gap: {len(bad)} configured language(s) did not pass "
                     f"— {names}.", "warn")

    def row_style(r):
        return {"ok": "", "wrong_language": "background:rgba(236,131,90,.10)",
                "empty": "background:rgba(208,59,59,.08)",
                "error": "background:rgba(208,59,59,.08)",
                "load_failed": "background:rgba(236,131,90,.10)"}.get(r.get("status"), "")

    disp = []
    for r in rows:
        disp.append({
            "label": esc(r.get("label")), "voice": esc(r.get("voice")),
            "status_b": badge(r.get("status", "?"),
                              {"ok": "pass", "wrong_language": "warn", "empty": "fail",
                               "error": "fail", "load_failed": "warn"}.get(r.get("status"), "muted")),
            "detected": esc(r.get("detected_language")),
            "wer": pct(r.get("round_trip_wer")) if r.get("round_trip_wer") not in (None, "", "None") else "—",
            "sample": esc((r.get("sample_hyp") or "")[:60]),
            "status": r.get("status"),
        })
    cols = [("label", "Language"), ("voice", "Voice"), ("status_b", "Status"),
            ("detected", "Whisper detected"), ("wer", "Round-trip WER"),
            ("sample", "Sample transcript")]
    return views(cards + gap + chart,
                 cards + gap + table(disp, cols, highlight=row_style))


# ----------------------------------------------------------------------------
# multilingual acceptance matrix (1.13) — the centrepiece
# ----------------------------------------------------------------------------

STATUS_KIND = {"ok": "pass", "empty": "fail", "error": "fail", "load_failed": "warn"}
JOULE_LABEL = {"ga": "GA", "preview": "Preview", "": "—", None: "—"}


def render_acceptance(matrix: list[dict]) -> str:
    if not matrix:
        return notice(
            "No language-acceptance matrix found. Run Test 1.13 "
            "(python -m eval.run stt --test language_acceptance).")

    def group(pred):
        return [r for r in matrix if pred(r)]

    ga = group(lambda r: r.get("joule") == "ga")
    prev = group(lambda r: r.get("joule") == "preview")
    non = group(lambda r: not r.get("joule"))

    total = len(matrix)
    ok = sum(1 for r in matrix if r.get("status") == "ok")
    empty = [r for r in matrix if r.get("status") == "empty"]
    joule = ga + prev
    joule_ok = sum(1 for r in joule if r.get("status") == "ok")

    cards = (f'<div class="cards">'
             f'{card(f"{ok}/{total}", "languages transcribe", "pass")}'
             f'{card(f"{joule_ok}/{len(joule)}", "SAP Joule languages transcribe", "pass" if joule_ok == len(joule) else "warn")}'
             f'{card(str(len(empty)), "return blank output", "fail")}'
             f'</div>')

    # stacked ok/empty summary bar across the three groups
    def counts(rows):
        o = sum(1 for r in rows if r.get("status") == "ok")
        return o, len(rows) - o
    g_labels = [f"Joule GA ({len(ga)})", f"Joule Preview ({len(prev)})",
                f"Outside scope ({len(non)})"]
    ok_vals = [counts(g)[0] for g in (ga, prev, non)]
    bad_vals = [counts(g)[1] for g in (ga, prev, non)]
    sum_chart = add_chart(
        "c_accept", "bar",
        {"labels": g_labels,
         "datasets": [{"label": "Transcribes", "data": ok_vals,
                       "backgroundColor": "@good", "borderColor": "@good",
                       "stack": "s", "borderRadius": 4},
                      {"label": "Blank / error", "data": bad_vals,
                       "backgroundColor": "@crit", "borderColor": "@crit",
                       "stack": "s", "borderRadius": 4}]},
        {"indexAxis": "y",
         "plugins": {"legend": {"display": True, "position": "top"}},
         "scales": {"x": {"stacked": True, "title": {"display": True, "text": "Languages"}},
                    "y": {"stacked": True}}}, height=200)

    gap = ""
    joule_blank = [r for r in empty if r.get("joule")]
    if joule_blank:
        names = ", ".join(f'<b>{esc(r["label"])}</b>' for r in joule_blank)
        gap = (f'<p class="notice fail"><b>Serving gap:</b> {len(joule_blank)} '
               f'SAP Joule GA language(s) — {names} — return empty transcripts on '
               f'this endpoint despite being GA on the Nemotron-3.5 model card. '
               f'A deployment/gateway issue, not a model capability gap.</p>')

    def row_style(r):
        k = STATUS_KIND.get(r.get("status"), "")
        return {"pass": "", "fail": "background:rgba(208,59,59,.08)",
                "warn": "background:rgba(236,131,90,.10)"}.get(k, "")

    def prep(rows):
        out = []
        for r in rows:
            rr = dict(r)
            rr["label"] = esc(r.get("label"))
            rr["key"] = esc(r.get("key"))
            rr["joule_h"] = JOULE_LABEL.get(r.get("joule"), "—")
            rr["quality"] = esc(r.get("quality"))
            rr["status_b"] = badge(r.get("status", "?"),
                                   STATUS_KIND.get(r.get("status"), "muted"))
            metric = (r.get("metric") or "").upper()
            sv = r.get("sample_primary")
            rr["sample"] = f"{metric} {fnum(sv, 3)}" if sv not in (None, "", "None") else "—"
            rr["sample_ref"] = esc((r.get("sample_ref") or "")[:70])
            cfg = r.get("fleurs") or ""
            rr["fleurs_link"] = (
                '<a href="https://huggingface.co/datasets/google/fleurs/viewer/'
                f'{esc(cfg)}/test" target="_blank" rel="noopener">{esc(cfg)}</a>'
                if cfg else "—")
            out.append(rr)
        return out

    cols = [("label", "Language"), ("key", "Key"), ("joule_h", "Joule"),
            ("quality", "Model tier"), ("fleurs_link", "FLEURS clips"),
            ("status_b", "Status"),
            ("sample", "Sample WER/CER"), ("sample_ref", "Reference sample")]

    blanks = ", ".join(f'{esc(r["label"])} ({esc(r["key"])})' for r in empty)
    blocks = [cards, sum_chart, gap]
    blocks.append(f"<h4>SAP Joule — GA ({len(ga)})</h4>")
    blocks.append(table(prep(ga), cols, highlight=row_style))
    blocks.append(f"<h4>SAP Joule — Preview ({len(prev)})</h4>")
    blocks.append(table(prep(prev), cols, highlight=row_style))
    blocks.append(f"<h4>Model-supported, outside Joule scope ({len(non)})</h4>")
    blocks.append(table(prep(non), cols, highlight=row_style))
    if blanks:
        blocks.append('<p class="notice muted">Blank languages: ' + esc(blanks)
                      + '. Each is recorded as a graceful <code>no_output</code> '
                      'row rather than a fabricated score.</p>')
    return "".join(blocks)

# ----------------------------------------------------------------------------
# phase 0 + page assembly
# ----------------------------------------------------------------------------


def render_phase0(data: dict) -> str:
    if not data:
        return notice("Phase 0 discovery.json not found in the selected run.")
    rows = []
    for name, r in data.items():
        if name == "cold_start":
            continue
        if not isinstance(r, dict):
            rows.append({"check": name, "status": badge("info", "muted"),
                         "detail": esc(r)})
            continue
        if r.get("error"):
            rows.append({"check": name, "status": badge("error", "fail"),
                         "detail": esc(r.get("error"))[:120]})
        elif "by_status" in r:
            rows.append({"check": name, "status": badge("ok", "pass"),
                         "detail": f"acceptance matrix: {esc(r.get('by_status'))}; "
                                   f"Joule ok {esc(r.get('joule_ok'))}/{esc(r.get('joule_total'))}"})
        else:
            rows.append({"check": name, "status": badge("ok", "pass"),
                         "detail": esc(json.dumps(r, default=str))[:140]})
    return table(rows, [("check", "Sub-phase"), ("status", "Status"),
                        ("detail", "Detail")])


def chip(rows_present: bool, na: bool = False) -> str:
    if na:
        return badge("N/A", "warn")
    return badge("live data", "pass") if rows_present else badge("not run", "muted")


# (anchor, test-id, title)
STT_TESTS = [
    ("t1_1", "1.1", "Test 1.1 — Clean-Speech Accuracy"),
    ("t1_12", "1.12", "Test 1.12 — Multilingual Accuracy"),
    ("t1_2", "1.2", "Test 1.2 — Batch Performance"),
    ("t1_3", "1.3", "Test 1.3 — Streaming Performance"),
    ("t1_4", "1.4", "Test 1.4 — REST vs gRPC"),
    ("t1_5", "1.5", "Test 1.5 — Noise Robustness"),
    ("t1_6", "1.6", "Test 1.6 — Accent Robustness"),
    ("t1_7", "1.7", "Test 1.7 — Long-Form Audio"),
    ("t1_8", "1.8", "Test 1.8 — Domain Vocabulary"),
    ("t1_9", "1.9", "Test 1.9 — Output Quality"),
    ("t1_10", "1.10", "Test 1.10 — Confidence Calibration"),
    ("t1_11", "1.11", "Test 1.11 — Format Robustness"),
    ("t1_13", "1.13", "Test 1.13 — Language Acceptance (34 languages)"),
]
TTS_TESTS = [
    ("t2_1", "2.1", "Test 2.1 — Naturalness"),
    ("t2_2", "2.2", "Test 2.2 — Intelligibility"),
    ("t2_3", "2.3", "Test 2.3 — Prosody"),
    ("t2_4", "2.4", "Test 2.4 — Signal Quality"),
    ("t2_5", "2.5", "Test 2.5 — Streaming Latency"),
    ("t2_6", "2.6", "Test 2.6 — Throughput & Concurrency"),
    ("t2_7", "2.7", "Test 2.7 — Edge Cases"),
    ("t2_8", "2.8", "Test 2.8 — Long-Form Consistency"),
    ("t2_9", "2.9", "Test 2.9 — Language Acceptance"),
]


def build(run_dir: Path, acceptance_csv: Path, out: Path, model: str,
          endpoint: str, dual_run_link: str | None = None,
          tts_model: str = "Magpie TTS Multilingual") -> Path:
    _CHARTS.clear()
    stt = run_dir / "stt"
    tts = run_dir / "tts"
    matrix = read_csv(acceptance_csv)

    # ---- load STT ----
    acc = read_csv(stt / "accuracy" / "summary.csv")
    perf = read_csv(stt / "performance" / "summary.csv")
    conc = read_csv(stt / "performance" / "concurrency_summary.csv")
    stream = read_csv(stt / "streaming" / "summary.csv")
    rest = read_csv(stt / "rest_vs_grpc" / "summary.csv")
    noise = read_csv(stt / "noise_robustness" / "summary.csv")
    accent = read_csv(stt / "accent" / "summary.csv")
    longf = read_csv(stt / "long_form" / "summary.csv")
    domain = read_csv(stt / "domain" / "summary.csv")
    outq = read_csv(stt / "output_quality" / "summary.csv")
    conf = read_csv(stt / "confidence" / "summary.csv")
    fmt = read_csv(stt / "format_robustness" / "summary.csv")
    mling = read_csv(stt / "multilingual" / "summary.csv")

    # ---- load TTS ----
    nat = read_csv(tts / "naturalness" / "summary.csv")
    intel = read_csv(tts / "intelligibility" / "summary.csv")
    pros = read_csv(tts / "prosody" / "summary.csv")
    sig = read_csv(tts / "signal_quality" / "summary.csv")
    tlat = read_csv(tts / "latency" / "summary.csv")
    tconc = read_csv(tts / "concurrency" / "summary.csv")
    edge = read_csv(tts / "edge_cases" / "summary.csv")
    tlong = read_csv(tts / "long_form" / "summary.csv")
    tacc = read_csv(tts / "language_acceptance" / "summary.csv")

    # ---- exec summary ----
    clean_wer = next((pct(r["wer"]) for r in acc
                      if "clean" in (r.get("dataset") or "").lower()), "—")
    utmos = "—"
    if nat:
        d = maybe_literal(nat[0].get("utmos"))
        if isinstance(d, dict):
            utmos = fnum(d.get("mean"), 2)
    ok = sum(1 for r in matrix if r.get("status") == "ok")
    joule = [r for r in matrix if r.get("joule")]
    joule_ok = sum(1 for r in joule if r.get("status") == "ok")

    exec_cards = (
        '<div class="cards">'
        + card(clean_wer, "STT clean WER (LibriSpeech)", "pass")
        + card(f"{joule_ok}/{len(joule) or '—'}", "SAP Joule languages transcribe",
               "pass" if joule and joule_ok == len(joule) else "warn")
        + card(f"{ok}/{len(matrix) or '—'}", "model languages transcribe", "")
        + card(utmos, "TTS naturalness (UTMOS, en)", "pass")
        + "</div>")

    # ---- STT cards ----
    stt_bodies = {
        "1.1": render_accuracy(acc),
        "1.2": render_performance(perf, conc),
        "1.3": render_streaming(stream),
        "1.4": render_rest_grpc(rest),
        "1.5": render_noise(noise),
        "1.6": render_accent(accent),
        "1.7": render_longform(longf),
        "1.8": render_domain(domain),
        "1.9": render_output_quality(outq),
        "1.10": render_confidence(conf),
        "1.11": render_format(fmt),
        "1.12": render_multilingual(mling),
        "1.13": render_acceptance(matrix),
    }
    stt_present = {"1.1": acc, "1.2": perf or conc, "1.3": stream, "1.4": rest,
                   "1.5": noise, "1.6": accent, "1.7": longf, "1.8": domain,
                   "1.9": outq, "1.10": conf, "1.11": fmt, "1.12": mling,
                   "1.13": matrix}

    stt_cards = []
    for anchor, tid, title in STT_TESTS:
        na = tid == "1.4"
        c = chip(bool(stt_present.get(tid)), na=na)
        stt_cards.append(test_card(anchor, title, TEST_META.get(tid, {}),
                                   c, stt_bodies[tid]))

    # ---- TTS cards ----
    tts_bodies = {
        "2.1": render_naturalness(nat, _per_lang_rows(tts / "naturalness")),
        "2.2": render_intelligibility(intel, _per_lang_rows(tts / "intelligibility")),
        "2.3": render_prosody(pros, _per_lang_rows(tts / "prosody")),
        "2.4": render_signal(sig),
        "2.5": render_tts_latency(tlat),
        "2.6": render_tts_concurrency(tconc),
        "2.7": render_edge(edge),
        "2.8": render_tts_longform(tlong),
        "2.9": render_tts_acceptance(tacc),
    }
    tts_present = {"2.1": nat, "2.2": intel, "2.3": pros, "2.4": sig, "2.5": tlat,
                   "2.6": tconc, "2.7": edge, "2.8": tlong, "2.9": tacc}
    tts_cards = []
    for anchor, tid, title in TTS_TESTS:
        na = tid == "2.4"
        c = chip(bool(tts_present.get(tid)), na=na)
        tts_cards.append(test_card(anchor, title, TEST_META.get(tid, {}),
                                   c, tts_bodies[tid]))

    phase0_card = test_card("phase0", "Phase 0 — Discovery",
                            TEST_META.get("phase0", {}),
                            chip(bool(read_json(run_dir / "phase0" / "discovery.json"))),
                            render_phase0(read_json(run_dir / "phase0" / "discovery.json")))

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    _mi = ('M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z|'
           'M19 10v2a7 7 0 0 1-14 0v-2|M12 19v3')
    _sp = ('P11 5 6 9 2 9 2 15 6 15 11 19 11 5|'
           'M15.5 8.5a5 5 0 0 1 0 7|M19 5a10 10 0 0 1 0 14')
    _ck = ('C12 12 9|P12 7 12 12 15 14')

    def _svg(spec):
        parts = []
        for seg in spec.split("|"):
            k, rest = seg[0], seg[1:]
            if k == "M":
                parts.append(f'<path d="{seg}"/>')
            elif k == "P":
                parts.append(f'<polyline points="{rest}"/>')
            elif k == "R":
                x, y, w, h, r = rest.split()
                parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}"/>')
            elif k == "C":
                cx, cy, rr = rest.split()
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="{rr}"/>')
            elif k == "L":
                x1, y1, x2, y2 = rest.split()
                parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
        return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                + "".join(parts) + "</svg>")

    def _meta(icon, label, value):
        return (f'<div class="mi">{_svg(icon)}<div>'
                f'<span class="ml">{esc(label)}</span>'
                f'<span class="mv">{esc(value)}</span></div></div>')

    hero_meta = (
        '<div class="herometa">'
        + _meta(_mi, "STT model", model)
        + _meta(_sp, "TTS model", tts_model)
        + _meta(_ck, "Generated", now)
        + '</div>')

    summary_card = (
        '<section class="card hero" id="summary">'
        '<p class="kicker"><span class="lv" aria-hidden="true"></span>Evaluation Report</p>'
        '<h2 class="herotitle">Executive Summary</h2>'
        '<p class="summ">One report covering Phase 0 discovery, STT tests 1.1–1.13 '
        '(including the full 34-language multilingual matrix) and TTS tests 2.1–2.9. '
        'All metrics are computed client-side. Every test leads with a plain-language '
        'explanation you can expand; any test with no data on disk says so rather '
        'than showing a fabricated number.</p>'
        + hero_meta + exec_cards + "</section>")

    # ---- sidebar nav ----
    def nav_links(items):
        return "".join(f'<a href="#{a}">{esc(t.split("—")[0].strip())}'
                       f'<span style="color:var(--muted)"> {esc(t.split("—",1)[1].strip())}</span></a>'
                       if "—" in t else f'<a href="#{a}">{esc(t)}</a>'
                       for a, _, t in items)
    nav = (
        '<div class="navbackdrop" id="navBackdrop"></div>'
        '<nav class="side" id="sidenav" aria-label="Report sections">'
        '<button class="navclose" id="navClose" type="button" aria-label="Close navigation"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="6" y1="6" x2="18" y2="18"/><line x1="6" y1="18" x2="18" y2="6"/></svg></button>'
        '<a href="#summary">Executive Summary</a>'
        '<a href="#phase0">Phase 0 — Discovery</a>'
        '<div class="grp">Speech-to-Text · 13 tests</div>'
        + nav_links(STT_TESTS)
        + '<div class="grp">Text-to-Speech · 9 tests</div>'
        + nav_links(TTS_TESTS)
        + "</nav>")

    stt_band = band("stt", "Speech-to-Text (STT) — 13 tests",
                    "Transcription quality, speed, robustness and multilingual "
                    "coverage for the Nemotron-3.5 ASR model.")
    tts_band = band("tts", "Text-to-Speech (TTS) — 9 tests",
                    "Naturalness, intelligibility, prosody, latency and language "
                    "coverage for the Magpie TTS Multilingual voice.")

    main_html = (summary_card + phase0_card
                 + stt_band + "".join(stt_cards)
                 + tts_band + "".join(tts_cards))

    dual_btn = (f'<a class="hbtn" href="{esc(dual_run_link)}" aria-label="Dual-run comparison report">'
                '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>'
                '<span>Dual-run comparison</span></a>'
                if dual_run_link else "")
    specs_json = json.dumps(_CHARTS)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech Model Evaluation — Full Report</title>
<script>(function(){{try{{var t=localStorage.getItem('report-theme');
if(t)document.documentElement.setAttribute('data-theme',t);}}catch(e){{}}}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap">
<style>{CSS}</style></head>
<body>
<header class="top">
<button class="hamburger" id="navToggle" type="button" aria-label="Open navigation" aria-controls="sidenav" aria-expanded="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg></button>
<p class="brand"><span class="dot" aria-hidden="true"></span>Speech Model Evaluation</p>
<div class="spacer"></div>
{dual_btn}
<button class="hbtn themeBtn" id="themeBtn" type="button" aria-label="Toggle dark mode"><svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg><svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg><span id="themeTxt">Dark</span></button>
</header>
<div class="layout">{nav}<main>{main_html}</main></div>
<footer>Generated by eval/report/generate_full_report.py — stdlib only, no fabricated data. Charts by Chart.js.</footer>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script>
const CHART_SPECS = {specs_json};
let CHARTS = [];
function cssVar(n){{return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}}
const TOKEN = {{'@s1':'--s1','@s2':'--s2','@s3':'--s3','@s4':'--s4','@s5':'--s5',
'@s6':'--s6','@s7':'--s7','@s8':'--s8','@good':'--good','@warn':'--warn',
'@serious':'--serious','@crit':'--crit','@muted':'--muted','@ink':'--ink',
'@ink2':'--ink2','@grid':'--grid','@axis':'--axis'}};
function resolve(v){{
  if(typeof v==='string'&&v[0]==='@')return cssVar(TOKEN[v]||'--ink')||'#888';
  if(Array.isArray(v))return v.map(resolve);
  if(v&&typeof v==='object'){{const o={{}};for(const k in v)o[k]=resolve(v[k]);return o;}}
  return v;
}}
// Value labels for single-series horizontal bars: a true 0 (e.g. a perfect 0%
// WER) renders a zero-length bar that looks empty — the label makes the number
// explicit. Anchored past the bar end, or inside it (white) when it would overflow.
const barValueLabels={{
  id:'barvaluelabels',
  afterDatasetsDraw(chart){{
    if(chart.config.type!=='bar') return;
    if((chart.options.indexAxis||'x')!=='y') return;
    if(chart.data.datasets.length!==1) return;
    const meta=chart.getDatasetMeta(0);const ds=chart.data.datasets[0];
    const {{ctx,chartArea}}=chart;ctx.save();
    ctx.font="600 12px "+Chart.defaults.font.family;ctx.textBaseline='middle';
    const ink=cssVar('--ink2')||'#52514e';
    meta.data.forEach((el,i)=>{{
      const v=ds.data[i];if(v==null||typeof v!=='number'&&isNaN(parseFloat(v)))return;
      const txt=Number(v).toFixed(1);const w=ctx.measureText(txt).width;
      if(el.x+6+w<=chartArea.right){{ctx.fillStyle=ink;ctx.textAlign='left';ctx.fillText(txt,el.x+6,el.y);}}
      else{{ctx.fillStyle='#ffffff';ctx.textAlign='right';ctx.fillText(txt,el.x-6,el.y);}}
    }});
    ctx.restore();
  }}
}};
Chart.register(barValueLabels);
function toggleView(btn,mode){{
  const card=btn.closest('.card');if(!card)return;
  const v=card.querySelector('.views');if(v)v.setAttribute('data-view',mode);
  card.querySelectorAll('.vtoggle button').forEach(b=>{{
    const on=b===btn;b.classList.toggle('active',on);
    b.setAttribute('aria-pressed',on?'true':'false');
  }});
}}
function buildCharts(){{
  CHARTS.forEach(c=>c.destroy());CHARTS=[];
  Chart.defaults.font.family="'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif";
  Chart.defaults.color=cssVar('--ink2');
  Chart.defaults.borderColor=cssVar('--grid');
  Chart.defaults.plugins.legend.labels.boxWidth=12;
  Chart.defaults.plugins.legend.labels.usePointStyle=true;
  CHART_SPECS.forEach(spec=>{{
    const el=document.getElementById(spec.id);if(!el)return;
    const cfg={{type:spec.type,data:resolve(spec.data),
      options:Object.assign({{responsive:true,maintainAspectRatio:false,
        interaction:{{mode:'nearest',intersect:false}}}},resolve(spec.options))}};
    CHARTS.push(new Chart(el,cfg));
  }});
}}
function setTheme(t){{
  document.documentElement.setAttribute('data-theme',t);
  try{{localStorage.setItem('report-theme',t);}}catch(e){{}}
  const dark=t==='dark';
  var tt=document.getElementById('themeTxt');if(tt)tt.textContent=dark?'Light':'Dark';
  buildCharts();
}}
(function(){{
  const cur=document.documentElement.getAttribute('data-theme')||
    (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  const dark=cur==='dark';
  var tt0=document.getElementById('themeTxt');if(tt0)tt0.textContent=dark?'Light':'Dark';
  document.getElementById('themeBtn').addEventListener('click',()=>{{
    const now=(document.documentElement.getAttribute('data-theme')==='dark')?'light':'dark';
    setTheme(now);
  }});
  buildCharts();
  // scrollspy
  const links=[...document.querySelectorAll('nav.side a')];
  const map={{}};links.forEach(a=>map[a.getAttribute('href').slice(1)]=a);
  const obs=new IntersectionObserver(es=>{{
    es.forEach(e=>{{if(e.isIntersecting){{
      links.forEach(a=>a.classList.remove('active'));
      const a=map[e.target.id];if(a)a.classList.add('active');
    }}}});
  }},{{rootMargin:'-20% 0px -70% 0px'}});
  document.querySelectorAll('section.card[id]').forEach(s=>obs.observe(s));
  // mobile nav drawer
  const nav=document.getElementById('sidenav');
  const navToggle=document.getElementById('navToggle');
  const navBackdrop=document.getElementById('navBackdrop');
  const navClose=document.getElementById('navClose');
  let lastFocus=null;
  function openNav(){{
    lastFocus=document.activeElement;
    nav.classList.add('open');navBackdrop.classList.add('show');
    document.body.classList.add('navopen');
    navToggle.setAttribute('aria-expanded','true');
    if(navClose)navClose.focus();
  }}
  function closeNav(){{
    nav.classList.remove('open');navBackdrop.classList.remove('show');
    document.body.classList.remove('navopen');
    navToggle.setAttribute('aria-expanded','false');
    if(lastFocus&&lastFocus.focus)lastFocus.focus();
  }}
  if(navToggle)navToggle.addEventListener('click',openNav);
  if(navClose)navClose.addEventListener('click',closeNav);
  if(navBackdrop)navBackdrop.addEventListener('click',closeNav);
  links.forEach(a=>a.addEventListener('click',()=>{{if(nav.classList.contains('open'))closeNav();}}));
  document.addEventListener('keydown',e=>{{if(e.key==='Escape'&&nav.classList.contains('open'))closeNav();}});
}})();
</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Comprehensive single-file HTML report")
    ap.add_argument("--run", default="results/run-04-11-26/riva",
                    help="Run directory containing stt/ tts/ phase0/")
    ap.add_argument("--acceptance",
                    default="results/run-04-11-26/riva/stt/language_acceptance/language_acceptance.csv",
                    help="Language-acceptance matrix CSV (Test 1.13)")
    ap.add_argument("--out", default="docs/full-report.html")
    ap.add_argument("--model", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    ap.add_argument("--tts-model", default="Magpie TTS Multilingual",
                    help="TTS model name shown in the header")
    ap.add_argument("--endpoint", default="SAP AI Core gateway (gRPC StreamingRecognize)")
    ap.add_argument("--dual-run-link", default=None,
                    help="If set, inject a 'Dual-run comparison report' header button")
    args = ap.parse_args(argv)
    out = build(Path(args.run), Path(args.acceptance), Path(args.out),
                args.model, args.endpoint, args.dual_run_link, args.tts_model)
    print(f"Report written to: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
