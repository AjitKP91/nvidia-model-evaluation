"""Comprehensive single-file HTML report — a beginner-friendly website.

Covers Phase 0 discovery, STT 1.1-1.13 (including the full 40-language
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
        "how": "eval/stt/accuracy.py streams each clip over gRPC, then scores it "
               "with eval.utils.score_pair: text is normalised with Whisper's "
               "EnglishTextNormalizer and WER/CER/MER are computed with jiwer "
               "(edit distance over words/characters). Per-dataset means are "
               "written to accuracy/summary.csv with a PASS/warn verdict.",
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
        "how": "eval/stt/performance.py times each gRPC call, computing RTF = "
               "elapsed ÷ audio-duration and RTFx = its inverse, bucketed by clip "
               "length. A concurrency sweep replays the workload at increasing "
               "parallelism and records P50/P99 latency and mean RTF per level "
               "(concurrency_summary.csv).",
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
        "how": "eval/stt/streaming.py drives StreamingRecognize and records "
               "time-to-first-word, finalisation latency, partial latency and a "
               "stability rate (how often partials survive unchanged), plus a "
               "batch-vs-stream WER comparison. Percentiles are stored as dicts "
               "in summary.csv. English-only by design.",
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
        "how": "eval/stt/rest_vs_grpc.py expects both interfaces; when REST is "
               "unavailable it writes no fabricated numbers. The report shows the "
               "N/A rationale instead of a chart.",
        "datasets": [],
        "metrics": [],
    },
    "1.5": {
        "summary": "Noise robustness — how accuracy degrades as background noise "
                   "gets louder, across four noise types.",
        "what": "Mixes clean speech with real-world noise at controlled signal-"
                "to-noise ratios and tracks WER as conditions worsen — a "
                "stress test for noisy rooms, cafés, streets and babble.",
        "how": "eval/stt/noise_robustness.py adds MUSAN/DEMAND noise to "
               "LibriSpeech at set SNRs, transcribes, and scores WER per "
               "(noise_type, SNR). It also reports delta_rel, the relative WER "
               "increase versus the clean baseline (snr=∞).",
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
        "how": "eval/stt/accent.py transcribes VCTK and L2-ARCTIC clips grouped "
               "by accent, computes per-accent WER, and derives fairness_delta = "
               "each group's WER minus the best group's WER.",
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
        "how": "eval/stt/long_form.py transcribes each long file whole and in "
               "chunks, scoring full_wer and chunked_wer, plus a hallucination "
               "rate and repetition rate that flag runaway or looping output.",
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
        "how": "eval/stt/domain.py computes domain_wer, a key-term recall rate "
               "(KRR — how many domain terms were caught) and NE-WER (error rate "
               "restricted to named entities), and surfaces the most-"
               "misrecognised terms.",
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
        "how": "eval/stt/output_quality.py compares formatted output against "
               "references and reports per-aspect scores (e.g. F1 for "
               "punctuation marks, capitalisation accuracy, numeric formatting) "
               "in output_quality/summary.csv.",
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
        "how": "eval/stt/confidence.py bins words by confidence and compares each "
               "bin's mean confidence to its measured accuracy, reporting ECE "
               "(calibration gap), AUROC (confidence as an error detector) and "
               "the Brier score, plus the per-bin reliability curve.",
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
        "how": "eval/stt/format_robustness.py submits the same utterance in many "
               "encodings/rates/layouts and records pass/fail per variant with "
               "any error returned. English-only serving-stack property.",
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
        "how": "eval/stt/multilingual.py transcribes FLEURS audio per language "
               "and scores with eval.utils.score_pair, routing CJK languages to "
               "CER and the rest to WER. Opt-in: populated when the multilingual "
               "language lists are configured.",
        "datasets": [
            _ds("FLEURS (102-language read speech)",
                "https://huggingface.co/datasets/google/fleurs"),
        ],
        "metrics": ["WER", "CER"],
    },
    "1.13": {
        "summary": "Language acceptance — a live probe of all 40 model languages: "
                   "does each one actually return a transcript on this endpoint?",
        "what": "The multilingual centrepiece. Rather than trusting the model "
                "card, it sends one real clip per language and records whether "
                "the deployment returns usable text — rendered SAP-Joule-first.",
        "how": "eval/stt/language_acceptance.py probes each language once, "
               "classifies the result as ok / empty / error / load_failed, and "
               "captures a sample reference/hypothesis and a sample WER/CER. "
               "Languages that return nothing are logged as graceful no_output "
               "rows, never scored as fabricated numbers.",
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
        "how": "eval/tts/naturalness.py synthesizes sentences and scores them "
               "with UTMOS (English only) and DNSMOS (language-agnostic P.835: "
               "SIG/BAK/OVRL). Per-language runs report DNSMOS only; UTMOS is "
               "omitted for non-English because it was validated on English.",
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
        "how": "eval/tts/intelligibility.py synthesizes each sentence, "
               "transcribes it with Whisper large-v3 (language-hinted), and "
               "computes round-trip WER/CER after normalisation — by content "
               "category and, for multilingual runs, per language via FLEURS.",
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
        "how": "eval/tts/prosody.py extracts reference-free prosody features: F0 "
               "mean/std/range (pitch dynamism), an acoustic nPVI approximation "
               "(rhythm), pause ratio and words-per-minute. Intrusive F0-vs-"
               "reference metrics are not computable for a fixed synthetic voice "
               "(see 2.4).",
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
        "how": "eval/tts/signal_quality.py records the N/A rationale rather than "
               "emitting misleading intrusive scores. Use 2.1 and 2.2 instead; a "
               "valid intrusive run needs a zero-shot voice-cloning deployment.",
        "datasets": [],
        "metrics": [],
    },
    "2.5": {
        "summary": "Streaming latency — how quickly audio starts and streams, by "
                   "text length.",
        "what": "Measures responsiveness for live use: how soon the first audio "
                "chunk arrives (TTFB) and how fast speech is produced relative to "
                "its duration.",
        "how": "eval/tts/latency.py times gRPC streaming synthesis per text-"
               "length bucket, storing TTFB, RTF and elapsed as P50/P99 "
               "percentile dicts in summary.csv.",
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
        "how": "eval/tts/concurrency.py replays synthesis at rising concurrency "
               "levels and records success count, error rate, RPS and P50/P99 "
               "latency. Shown as two single-axis charts (throughput and latency "
               "vs concurrency) — never a dual-axis plot.",
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
        "how": "eval/tts/edge_cases.py runs each category and classifies every "
               "item as pass / fail / degraded, reporting a per-category "
               "pass_rate in summary.csv.",
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
        "how": "eval/tts/long_form.py synthesizes multi-paragraph passages and "
               "measures speaker similarity (ECAPA-TDNN embeddings) between "
               "paragraphs plus F0 and WPM drift across the passage.",
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
        "how": "eval/tts/language_acceptance.py synthesizes one probe per "
               "language, transcribes it with Whisper (language detection), and "
               "classifies ok / wrong_language / empty / error, recording the "
               "round-trip WER and a sample transcript.",
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
                "request/response schemas match, parameters behave, smoke tests "
                "pass and cold-start behaviour is recorded.",
        "how": "eval/run writes phase0/discovery.json with sub-checks for "
               "connectivity, STT/streaming/TTS schema, parameters, smoke tests, "
               "cold start and the language-acceptance summary.",
        "datasets": [],
        "metrics": [],
    },
})


# ----------------------------------------------------------------------------
# CSS — dataviz palette, light default + dark toggle, sidebar layout
# ----------------------------------------------------------------------------

CSS = """
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --card:#ffffff;
  --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --line:rgba(11,11,11,.10); --axis:#c3c2b7;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --crit:#d03b3b;
  --goodtext:#006300;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100;
  --s5:#e87ba4; --s6:#008300; --s7:#6250d6; --s8:#e34948;
  --accent:#2a78d6; --accent-ink:#1c5cab;
  --shadow:0 1px 2px rgba(11,11,11,.04),0 2px 8px rgba(11,11,11,.05);
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  --surface:#1a1a19; --plane:#0d0d0d; --card:#201f1e;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --line:rgba(255,255,255,.10); --axis:#383835;
  --goodtext:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --accent:#3987e5; --accent-ink:#86b6ef;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  --surface:#1a1a19; --plane:#0d0d0d; --card:#201f1e;
  --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --line:rgba(255,255,255,.10); --axis:#383835;
  --goodtext:#0ca30c;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500;
  --s5:#d55181; --s6:#008300; --s7:#9085e9; --s8:#e66767;
  --accent:#3987e5; --accent-ink:#86b6ef;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 2px 10px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);
  margin:0;background:var(--plane);-webkit-font-smoothing:antialiased}
a{color:var(--accent-ink)}
header.top{background:var(--card);border-bottom:1px solid var(--line);
  padding:22px 32px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}
header.top h1{margin:0;font-size:20px;font-weight:700;letter-spacing:-.01em}
header.top .meta{color:var(--muted);font-size:12.5px;line-height:1.5}
header.top .spacer{flex:1}
.hbtn{font:inherit;font-size:13px;font-weight:600;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);
  padding:8px 14px;border-radius:9px;text-decoration:none;display:inline-flex;
  align-items:center;gap:7px}
.hbtn:hover{border-color:var(--accent)}
.layout{display:flex;align-items:flex-start;max-width:1320px;margin:0 auto}
nav.side{position:sticky;top:0;align-self:flex-start;width:264px;flex:none;
  height:100vh;overflow-y:auto;padding:22px 14px 40px;border-right:1px solid var(--line)}
nav.side .grp{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.06em;color:var(--muted);margin:18px 10px 6px}
nav.side a{display:block;color:var(--ink2);text-decoration:none;font-size:13px;
  padding:6px 10px;border-radius:7px;border-left:2px solid transparent}
nav.side a:hover{background:var(--surface);color:var(--ink)}
nav.side a.active{color:var(--accent-ink);border-left-color:var(--accent);
  background:var(--surface);font-weight:600}
main{flex:1;min-width:0;padding:28px 40px 100px;max-width:980px}
.bandhead{margin:36px 0 10px;padding-bottom:10px;border-bottom:2px solid var(--accent)}
.bandhead:first-child{margin-top:4px}
.bandhead h2{margin:0;font-size:24px;font-weight:800;letter-spacing:-.02em}
.bandhead p{margin:6px 0 0;color:var(--ink2);font-size:14px;max-width:70ch}
section.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:22px 24px;margin:0 0 22px;box-shadow:var(--shadow);scroll-margin-top:16px}
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
.mdef{font-size:12.5px;color:var(--ink2);margin:2px 0}
.mdef b{color:var(--ink)}
.chartbox{position:relative;margin:4px 0 16px;width:100%}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0 4px;
  font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--grid);padding:7px 10px;text-align:left}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.03em;border-bottom:1px solid var(--axis)}
tbody tr:hover td{background:var(--surface)}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 14px}
.stat{flex:1;min-width:140px;background:var(--surface);border:1px solid var(--line);
  border-radius:11px;padding:14px 16px}
.stat .big{font-size:26px;font-weight:800;letter-spacing:-.01em}
.stat .lbl{color:var(--muted);font-size:12px;margin-top:3px;line-height:1.35}
.stat.pass .big{color:var(--goodtext)} .stat.fail .big{color:var(--crit)}
.stat.warn .big{color:var(--serious)}
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
  .layout{flex-direction:column}
  nav.side{position:static;width:100%;height:auto;border-right:none;
    border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:2px}
  nav.side .grp{width:100%;margin:8px 10px 2px}
  nav.side a{border-left:none;border-bottom:2px solid transparent}
  main{padding:20px 18px 80px;max-width:100%}
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
    mk = [m for m in (meta.get("metrics") or []) if m in GLOSSARY]
    if mk:
        parts.append("<h4>Metrics in this test</h4>")
        for m in mk:
            parts.append(f'<p class="mdef"><b>{esc(m)}</b> — {esc(GLOSSARY[m])}</p>')
    parts.append("</div></details>")
    return "".join(parts)


def test_card(anchor: str, title: str, meta: dict, chip_html: str,
              body: str) -> str:
    summ = (f'<p class="summ">{esc(meta.get("summary", ""))}</p>'
            if meta.get("summary") else "")
    return (f'<section class="card" id="{anchor}"><h3>{esc(title)}{chip_html}</h3>'
            f'{summ}{deepdive(meta)}{body}</section>')


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
    return chart + table(rows, cols)


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
    out = chart + "<h4>Throughput by audio duration</h4>" + table(rows, cols)
    if conc:
        for r in conc:
            r["rtf_mean"] = fnum(r.get("rtf_mean"), 4)
            r["p50"] = fnum(r.get("p50"), 3)
            r["p99"] = fnum(r.get("p99"), 3)
        cc = [("concurrency", "Concurrency"), ("n_calls", "N"), ("p50", "P50 (s)"),
              ("p99", "P99 (s)"), ("rtf_mean", "RTF")]
        out += "<h4>Concurrency</h4>" + table(conc, cc)
    return out


def render_streaming(rows):
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    r = rows[0]
    parts = []
    pbar_lbl, pbar_val = [], []
    for key, lbl in [("ttfw", "Time-to-first-word"),
                     ("finalization_latency", "Finalization latency"),
                     ("streaming_rtf", "Streaming RTF"),
                     ("partial_latency", "Partial latency")]:
        d = maybe_literal(r.get(key))
        if isinstance(d, dict):
            parts.append(card(f"{fnum(d.get('p50'), 3)}", f"{lbl} P50", ""))
            pbar_lbl.append(lbl)
            pbar_val.append(fval(d.get("p50"), 3))
    parts.append(card(pct(r.get("stability_rate_mean")), "Stability rate", "pass"))
    parts.append(card(pct(r.get("batch_wer")), "Batch WER", ""))
    parts.append(card(pct(r.get("stream_wer")), "Stream WER", ""))
    chart = bar("c_stream", pbar_lbl, pbar_val, horizontal=True, color="@s1",
                x_title="P50 latency (s)") if pbar_lbl else ""
    return '<div class="cards">' + "".join(parts) + "</div>" + chart


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
    return chart + table(rows, cols)


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
    return chart + table(rows, cols)


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
    return chart + tbl


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
    return tiles + table(rows, cols) + toplist


def render_output_quality(rows):
    if not rows:
        return table(rows, [])
    labels = [r.get("metric", "") for r in rows]
    vals = []
    for r in rows:
        v = fval(r.get("value"), 4)
        vals.append(None if v is None else (round(v * 100, 2) if 0 <= (v or 0) <= 1 else v))
    chart = bar("c_outq", labels, vals, horizontal=True, color="@s7",
                x_title="Score")
    cols = [("metric", "Metric"), ("value", "Value"), ("n_samples", "N")]
    return chart + table(rows, cols)


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
    return cards_html + chart + btbl


def render_format(rows):
    if not rows:
        return table(rows, [])
    for r in rows:
        r["status"] = badge(r.get("status", "?"),
                            "pass" if r.get("status") == "pass" else "fail")
    cols = [("test", "Test"), ("value", "Value"), ("status", "Status"),
            ("error", "Error")]
    return table(rows, cols)


def render_multilingual(rows):
    if not rows:
        return notice(
            "Full per-language accuracy (1.12) is opt-in and was not run here. "
            "The live 40-language acceptance probe (1.13) is shown below.", "muted")
    labels = [r.get("language", r.get("key", "")) for r in rows]
    vals = [None if fval(r.get("primary_value"), 4) is None
            else round(fval(r.get("primary_value"), 4) * 100, 2) for r in rows]
    chart = bar("c_ml", labels, vals, horizontal=True, color="@s1",
                x_title="WER / CER (%)")
    disp = []
    for r in rows:
        disp.append({"language": esc(r.get("language")),
                     "metric": esc((r.get("metric") or "").upper()),
                     "primary_value": pct(r.get("primary_value")),
                     "n": esc(r.get("n_utterances"))})
    cols = [("language", "Language"), ("metric", "Metric"),
            ("primary_value", "Score"), ("n", "N")]
    return chart + table(disp, cols)

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
    return (base + "<h4>Per-language (multilingual)</h4>" + chart + caveat
            + table(out, cols))


def render_intelligibility(rows, lang_rows=None):
    chart = ""
    if rows:
        labels = [r.get("category", "") for r in rows]
        vals = [None if fval(r.get("round_trip_wer"), 4) is None
                else round(fval(r.get("round_trip_wer"), 4) * 100, 2) for r in rows]
        chart = bar("c_intel", labels, vals, color="@s1",
                    x_title="Category", y_title="Round-trip WER (%)")
    for r in rows:
        r["round_trip_wer"] = pct(r.get("round_trip_wer"))
        r["round_trip_cer"] = pct(r.get("round_trip_cer"))
    cols = [("category", "Category"), ("n_sentences", "N"),
            ("round_trip_wer", "Round-trip WER"), ("round_trip_cer", "CER"),
            ("normalizer", "Normalizer")]
    base = chart + table(rows, cols)

    if not lang_rows:
        return base

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
                 x_title="Round-trip WER (%)") if labels else ""
    cols2 = [("language", "Lang"), ("n_sentences", "N"),
             ("round_trip_wer", "Round-trip WER")]
    note = notice("FLEURS text synthesised per language, transcribed with "
                  "Whisper large-v3 using the language hint.", "muted")
    return (base + "<h4>Per-language (FLEURS round-trip)</h4>" + lchart + note
            + table(out, cols2))


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
    return note + chart + table(display, cols)


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
    return chart + render_percentile_table(
        rows, "bucket", "Bucket",
        [("ttfb", "TTFB (s)"), ("rtf", "RTF"), ("elapsed", "Elapsed (s)")])


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
    return chart + table(rows, cols)


def render_edge(rows):
    if not rows:
        return table(rows, [])
    labels = [r.get("category", "") for r in rows]
    vals = [None if fval(r.get("pass_rate"), 4) is None
            else round(fval(r.get("pass_rate"), 4) * 100, 1) for r in rows]
    colors = ["@good" if (v or 0) >= 90 else "@serious" if (v or 0) >= 50 else "@crit"
              for v in vals]
    chart = bar("c_edge", labels, vals, horizontal=True, colors=colors,
                x_title="Pass rate (%)")
    for r in rows:
        r["pass_rate"] = pct(r.get("pass_rate"))
    cols = [("category", "Category"), ("total", "Total"), ("pass", "Pass"),
            ("fail", "Fail"), ("degraded", "Degraded"), ("pass_rate", "Pass rate")]
    return chart + table(rows, cols)


def render_tts_longform(rows):
    chart = ""
    if rows:
        labels = [str(r.get("title", ""))[:22] for r in rows]
        sim = [fval(r.get("mean_speaker_sim"), 3) for r in rows]
        chart = bar("c_ttslf", labels, sim, horizontal=True, color="@s3",
                    x_title="Mean speaker similarity (higher = more consistent)")
    for r in rows:
        r["mean_speaker_sim"] = fnum(r.get("mean_speaker_sim"), 3)
        r["f0_drift_hz"] = fnum(r.get("f0_drift_hz"), 1)
        r["wpm_drift"] = fnum(r.get("wpm_drift"), 1)
    cols = [("title", "Passage"), ("n_paragraphs", "Paras"),
            ("mean_speaker_sim", "Speaker sim"), ("f0_drift_hz", "F0 drift (Hz)"),
            ("wpm_drift", "WPM drift")]
    return chart + table(rows, cols)


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
                x_title="Round-trip WER (%)") if any(v is not None for v in vals) else ""
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
    return cards + gap + chart + table(disp, cols, highlight=row_style)


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
            out.append(rr)
        return out

    cols = [("label", "Language"), ("key", "Key"), ("joule_h", "Joule"),
            ("quality", "Model tier"), ("status_b", "Status"),
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
        if not isinstance(r, dict):
            rows.append({"check": name, "status": badge("info", "muted"),
                         "detail": esc(r)})
            continue
        if r.get("error"):
            rows.append({"check": name, "status": badge("error", "fail"),
                         "detail": esc(r.get("error"))[:120]})
        elif "by_status" in r:
            rows.append({"check": name, "status": badge("ok", "pass"),
                         "detail": f"40-lang matrix: {esc(r.get('by_status'))}; "
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
    ("t1_12", "1.12", "Test 1.12 — Multilingual Accuracy"),
    ("t1_13", "1.13", "Test 1.13 — Language Acceptance (40 languages)"),
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
        + card(f"{ok}/{len(matrix) or '—'}", "of 40 model languages transcribe", "")
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

    gaps = (
        "<ul>"
        "<li><b>th / id / ar blank on this endpoint</b> — 3 SAP Joule GA languages "
        "return empty transcripts. A serving/gateway gap, not a model gap.</li>"
        "<li><b>1.6 dialect &amp; 1.8 domain</b> — multilingual coverage gaps; need "
        "per-locale / multilingual domain corpora. Deliberately not faked.</li>"
        "<li><b>1.3 / 1.4 / 1.11</b> — English-only by design (serving-stack / "
        "audio-frontend properties; 1.4 N/A — gRPC-only deployment).</li>"
        "<li><b>Full 1.12 accuracy + per-language depth</b> — opt-in; run with the "
        "multilingual config lists populated to replace the acceptance-only view.</li>"
        "</ul>")

    phase0_card = test_card("phase0", "Phase 0 — Discovery",
                            TEST_META.get("phase0", {}),
                            chip(bool(read_json(run_dir / "phase0" / "discovery.json"))),
                            render_phase0(read_json(run_dir / "phase0" / "discovery.json")))

    summary_card = (
        '<section class="card" id="summary"><h3>Executive Summary</h3>'
        '<p class="summ">One report covering Phase 0 discovery, STT tests 1.1–1.13 '
        '(including the full 40-language multilingual matrix) and TTS tests 2.1–2.9. '
        'All metrics are computed client-side. Every test leads with a plain-language '
        'explanation you can expand; any test with no data on disk says so rather '
        'than showing a fabricated number.</p>'
        + exec_cards + "</section>")

    gaps_card = ('<section class="card" id="gaps"><h3>Known Gaps &amp; Honest Scope</h3>'
                 '<p class="summ">What this run does <i>not</i> claim, stated plainly.</p>'
                 + gaps + "</section>")

    # ---- sidebar nav ----
    def nav_links(items):
        return "".join(f'<a href="#{a}">{esc(t.split("—")[0].strip())}'
                       f'<span style="color:var(--muted)"> {esc(t.split("—",1)[1].strip())}</span></a>'
                       if "—" in t else f'<a href="#{a}">{esc(t)}</a>'
                       for a, _, t in items)
    nav = (
        '<nav class="side" aria-label="Report sections">'
        '<a href="#summary">Executive Summary</a>'
        '<div class="grp">Speech-to-Text · 13 tests</div>'
        + nav_links(STT_TESTS)
        + '<div class="grp">Text-to-Speech · 9 tests</div>'
        + nav_links(TTS_TESTS)
        + '<div class="grp">More</div>'
        + '<a href="#phase0">Phase 0 — Discovery</a>'
        + '<a href="#gaps">Known Gaps &amp; Scope</a>'
        + "</nav>")

    stt_band = band("stt", "Speech-to-Text (STT) — 13 tests",
                    "Transcription quality, speed, robustness and multilingual "
                    "coverage for the Nemotron-3.5 ASR model.")
    tts_band = band("tts", "Text-to-Speech (TTS) — 9 tests",
                    "Naturalness, intelligibility, prosody, latency and language "
                    "coverage for the Magpie TTS Multilingual voice.")

    main_html = (summary_card + stt_band + "".join(stt_cards)
                 + tts_band + "".join(tts_cards)
                 + '<div class="bandhead" id="more"><h2>Discovery &amp; Scope</h2>'
                   '<p>Pre-flight checks and an honest statement of what is out of scope.</p></div>'
                 + phase0_card + gaps_card)

    dual_btn = (f'<a class="hbtn" href="{esc(dual_run_link)}">📊 Dual-run comparison report</a>'
                if dual_run_link else "")
    specs_json = json.dumps(_CHARTS)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech Model Evaluation — Full Report</title>
<script>(function(){{try{{var t=localStorage.getItem('report-theme');
if(t)document.documentElement.setAttribute('data-theme',t);}}catch(e){{}}}})();</script>
<style>{CSS}</style></head>
<body>
<header class="top">
<div><h1>Speech Model Evaluation</h1>
<div class="meta"><b>STT</b> · {esc(model)} &nbsp;·&nbsp; <b>TTS</b> · {esc(tts_model)}<br>
{esc(endpoint)} &nbsp;·&nbsp; Run <code>{esc(run_dir.name)}</code> &nbsp;·&nbsp; generated {esc(now)}</div></div>
<div class="spacer"></div>
{dual_btn}
<button class="hbtn" id="themeBtn" type="button" aria-label="Toggle dark mode">
<span id="themeIcon">🌙</span><span id="themeTxt">Dark</span></button>
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
function buildCharts(){{
  CHARTS.forEach(c=>c.destroy());CHARTS=[];
  Chart.defaults.font.family="system-ui,-apple-system,'Segoe UI',sans-serif";
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
  document.getElementById('themeIcon').textContent=dark?'☀️':'🌙';
  document.getElementById('themeTxt').textContent=dark?'Light':'Dark';
  buildCharts();
}}
(function(){{
  const cur=document.documentElement.getAttribute('data-theme')||
    (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
  const dark=cur==='dark';
  document.getElementById('themeIcon').textContent=dark?'☀️':'🌙';
  document.getElementById('themeTxt').textContent=dark?'Light':'Dark';
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
                    help="40-language acceptance matrix CSV (Test 1.13)")
    ap.add_argument("--out", default="docs/full-report.html")
    ap.add_argument("--model", default="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
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
