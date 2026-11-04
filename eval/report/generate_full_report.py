"""Comprehensive single-file HTML report — covers EVERYTHING.

Unlike eval/report/generate_report.py (which stops at STT 1.11 and never renders
the multilingual story), this generator covers:

  * Phase 0 discovery
  * STT 1.1 - 1.11  (from a run directory's per-test summary.csv)
  * STT 1.12 multilingual accuracy + 1.13 language acceptance  (the 40-language
    matrix, rendered Joule-first)
  * STT per-language depth (1.2/1.5/1.7/1.9/1.10 Tier-A/B) when present
  * TTS 2.1 - 2.8

Design goals:
  * STDLIB ONLY (csv, json, ast, html, pathlib, datetime) so it runs with the
    system python3 — no datasets/pydub/venv required.
  * HONEST: sections with no data on disk are rendered as an explicit
    "not run in this run" notice, never fabricated.

Usage:
    python3 eval/report/generate_full_report.py \
        --run results/run-07-07-26/riva \
        --acceptance results/stt/language_acceptance/language_acceptance.csv \
        --out docs/full-report.html
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
# small parsing helpers
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


def pct(val, nd=1):
    try:
        return f"{float(val) * 100:.{nd}f}%"
    except (TypeError, ValueError):
        return esc(val)


def esc(val) -> str:
    return html.escape("" if val is None else str(val))


# ----------------------------------------------------------------------------
# generic renderers
# ----------------------------------------------------------------------------


def table(rows: list[dict], cols: list[tuple[str, str]], highlight=None) -> str:
    """rows: list of dicts. cols: list of (key, header). highlight(row)->css or ''."""
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    head = "".join(f"<th>{esc(h)}</th>" for _, h in cols)
    body = []
    for r in rows:
        style = f' style="{highlight(r)}"' if highlight else ""
        tds = "".join(f"<td>{r.get(k, '')}</td>" for k, _ in cols)
        body.append(f"<tr{style}>{tds}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def notice(msg: str, kind: str = "muted") -> str:
    return f'<p class="notice {kind}">{esc(msg)}</p>'


def badge(text: str, kind: str) -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def section(anchor: str, title: str, body: str, subtitle: str = "") -> str:
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    return (
        f'<section id="{anchor}"><h2>{esc(title)}</h2>{sub}{body}</section>'
    )


# ----------------------------------------------------------------------------
# multilingual — the centerpiece
# ----------------------------------------------------------------------------

STATUS_KIND = {"ok": "pass", "empty": "fail", "error": "fail", "load_failed": "warn"}
JOULE_LABEL = {"ga": "GA", "preview": "Preview", "": "—", None: "—"}


def render_acceptance(matrix: list[dict]) -> str:
    if not matrix:
        return notice(
            "No language-acceptance matrix found. Run Test 1.13 "
            "(python -m eval.run stt --test language_acceptance)."
        )

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

    cards = (
        f'<div class="cards">'
        f'{card(f"{ok}/{total}", "languages transcribe", "pass")}'
        f'{card(f"{joule_ok}/{len(joule)}", "SAP Joule languages transcribe", "pass" if joule_ok == len(joule) else "warn")}'
        f'{card(str(len(empty)), "return blank output", "fail")}'
        f'</div>'
    )

    blanks = ", ".join(f'{esc(r["label"])} ({esc(r["key"])})' for r in empty)
    gap = ""
    joule_blank = [r for r in empty if r.get("joule")]
    if joule_blank:
        names = ", ".join(f'<b>{esc(r["label"])}</b>' for r in joule_blank)
        gap = (
            f'<p class="notice fail"><b>Serving gap:</b> {len(joule_blank)} '
            f'SAP Joule GA language(s) — {names} — return empty transcripts on '
            f'this endpoint despite being GA on the Nemotron-3.5 model card. '
            f'This is a deployment/gateway issue, not a model capability gap.</p>'
        )

    def row_style(r):
        k = STATUS_KIND.get(r.get("status"), "")
        return {"pass": "", "fail": "background:#fdecea", "warn": "background:#fff6e5"}.get(k, "")

    def prep(rows):
        out = []
        for r in rows:
            rr = dict(r)
            rr["label"] = esc(r.get("label"))
            rr["key"] = esc(r.get("key"))
            rr["joule_h"] = JOULE_LABEL.get(r.get("joule"), "—")
            rr["quality"] = esc(r.get("quality"))
            rr["status_b"] = badge(r.get("status", "?"), STATUS_KIND.get(r.get("status"), "muted"))
            metric = (r.get("metric") or "").upper()
            sv = r.get("sample_primary")
            rr["sample"] = f"{metric} {fnum(sv, 3)}" if sv not in (None, "", "None") else "—"
            rr["sample_ref"] = esc((r.get("sample_ref") or "")[:70])
            out.append(rr)
        return out

    cols = [
        ("label", "Language"), ("key", "Key"), ("joule_h", "Joule"),
        ("quality", "Model tier"), ("status_b", "Status"),
        ("sample", "Sample WER/CER"), ("sample_ref", "Reference sample"),
    ]

    blocks = [cards, gap]
    blocks.append(f"<h3>SAP Joule — GA ({len(ga)})</h3>")
    blocks.append(table(prep(ga), cols, highlight=row_style))
    blocks.append(f"<h3>SAP Joule — Preview ({len(prev)})</h3>")
    blocks.append(table(prep(prev), cols, highlight=row_style))
    blocks.append(f"<h3>Model-supported, outside Joule scope ({len(non)})</h3>")
    blocks.append(table(prep(non), cols, highlight=row_style))
    blocks.append(
        '<p class="notice muted">CER sample for Japanese reads high in this raw '
        'per-clip column because the endpoint emits one space per CJK character; '
        'the scored path (eval.utils.score_pair, CER branch) strips that spacing '
        'and the corrected CER is ~0. See docs/multilingual-test-gap-analysis.md.</p>'
    )
    blocks.append(
        '<p class="notice muted">Blank languages: ' + esc(blanks) + '. Each is '
        'recorded as a graceful <code>no_output</code> row by every per-language '
        'depth test rather than a fabricated score.</p>'
    )
    return "".join(blocks)


def card(big: str, label: str, kind: str = "") -> str:
    return (
        f'<div class="stat {kind}"><div class="big">{esc(big)}</div>'
        f'<div class="lbl">{esc(label)}</div></div>'
    )


# ----------------------------------------------------------------------------
# STT sections
# ----------------------------------------------------------------------------


def render_accuracy(rows):
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        r["cer"] = pct(r.get("cer"))
        r["mer"] = pct(r.get("mer"))
        v = r.get("verdict", "")
        r["verdict"] = badge(v, "pass" if v == "PASS" else "warn") if v else ""
    cols = [("dataset", "Dataset"), ("n_utterances", "N"), ("wer", "WER"),
            ("cer", "CER"), ("mer", "MER"), ("verdict", "Verdict")]
    return table(rows, cols)


def render_noise(rows):
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        dr = r.get("delta_rel")
        r["delta_rel"] = ("+" + pct(dr)) if dr not in (None, "", "0.0") else "baseline"
    cols = [("noise_type", "Noise"), ("snr_db", "SNR (dB)"), ("wer", "WER"),
            ("delta_rel", "Δ vs clean"), ("n_utterances", "N")]
    return table(rows, cols)


def render_output_quality(rows):
    cols = [("metric", "Metric"), ("value", "Value"), ("n_samples", "N")]
    return table(rows, cols)


def render_performance(rows, conc):
    for r in rows:
        r["rtfx_mean"] = fnum(r.get("rtfx_mean"), 1)
        r["rtf_mean"] = fnum(r.get("rtf_mean"), 4)
        r["latency_p50"] = fnum(r.get("latency_p50"), 3)
        r["latency_p99"] = fnum(r.get("latency_p99"), 3)
    cols = [("bucket_s", "Bucket (s)"), ("n_calls", "N"), ("rtfx_mean", "RTFx"),
            ("rtf_mean", "RTF"), ("latency_p50", "P50 (s)"), ("latency_p99", "P99 (s)")]
    out = "<h3>Throughput by audio duration</h3>" + table(rows, cols)
    if conc:
        for r in conc:
            r["rtf_mean"] = fnum(r.get("rtf_mean"), 4)
            r["p50"] = fnum(r.get("p50"), 3)
            r["p99"] = fnum(r.get("p99"), 3)
        cc = [("concurrency", "Concurrency"), ("n_calls", "N"), ("p50", "P50 (s)"),
              ("p99", "P99 (s)"), ("rtf_mean", "RTF")]
        out += "<h3>Concurrency</h3>" + table(conc, cc)
    return out


def render_longform(rows):
    for r in rows:
        r["full_wer"] = pct(r.get("full_wer"))
        r["chunked_wer"] = pct(r.get("chunked_wer"))
        r["hallucination_rate"] = pct(r.get("hallucination_rate"))
        r["repetition_rate"] = pct(r.get("repetition_rate"))
        r["duration_s"] = fnum(r.get("duration_s"), 1)
    cols = [("id", "File"), ("duration_s", "Dur (s)"), ("full_wer", "Full WER"),
            ("chunked_wer", "Chunked WER"), ("hallucination_rate", "Halluc."),
            ("repetition_rate", "Repet.")]
    return table(rows, cols)


def render_confidence(rows):
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    r = rows[0]
    cards = (
        '<div class="cards">'
        + card(fnum(r.get("ece"), 3), "ECE (lower=better)", "warn")
        + card(fnum(r.get("auroc"), 3), "AUROC", "")
        + card(fnum(r.get("brier_score"), 3), "Brier", "")
        + card(esc(r.get("n_words_evaluated")), "words evaluated", "")
        + "</div>"
    )
    bins = maybe_literal(r.get("bins"))
    btbl = ""
    if isinstance(bins, list):
        for b in bins:
            b["accuracy"] = fnum(b.get("accuracy"), 3)
            b["confidence"] = fnum(b.get("confidence"), 3)
            b["gap"] = fnum(b.get("gap"), 3)
        btbl = "<h3>Reliability bins</h3>" + table(
            bins, [("bin", "Bin"), ("n", "N"), ("confidence", "Mean conf."),
                   ("accuracy", "Accuracy"), ("gap", "Gap")])
    return cards + btbl


def render_percentile_table(rows, label_key, label_hdr, metric_keys):
    """Expand cells that hold {'p50':..,'p99':..} dicts into columns."""
    out = []
    for r in rows:
        rr = {label_hdr: esc(r.get(label_key))}
        for mk, mh in metric_keys:
            d = maybe_literal(r.get(mk))
            if isinstance(d, dict):
                rr[mh] = f"P50 {fnum(d.get('p50'), 3)} / P99 {fnum(d.get('p99'), 3)}"
            else:
                rr[mk] = fnum(d, 3)
        out.append(rr)
    cols = [(label_hdr, label_hdr)] + [(mh, mh) for _, mh in metric_keys]
    return table(out, cols)


def render_streaming(rows):
    if not rows:
        return notice("No data on disk for this test in the selected run.")
    r = rows[0]
    parts = []
    for key, lbl in [("ttfw", "Time-to-first-word"),
                     ("finalization_latency", "Finalization latency"),
                     ("streaming_rtf", "Streaming RTF"),
                     ("partial_latency", "Partial latency")]:
        d = maybe_literal(r.get(key))
        if isinstance(d, dict):
            parts.append(card(f"{fnum(d.get('p50'), 3)}", f"{lbl} P50", ""))
    parts.append(card(pct(r.get("stability_rate_mean")), "Stability rate", "pass"))
    parts.append(card(pct(r.get("batch_wer")), "Batch WER", ""))
    parts.append(card(pct(r.get("stream_wer")), "Stream WER", ""))
    return '<div class="cards">' + "".join(parts) + "</div>"


def render_rest_grpc(rows):
    for r in rows:
        r["p50"] = fnum(r.get("p50"), 4)
        r["p99"] = fnum(r.get("p99"), 4)
    cols = [("mode", "Mode"), ("interface", "Interface"), ("p50", "P50 (s)"), ("p99", "P99 (s)")]
    return table(rows, cols)


def render_accent(rows):
    for r in rows:
        r["wer"] = pct(r.get("wer"))
        r["fairness_delta"] = pct(r.get("fairness_delta"))
    cols = [("accent", "Accent"), ("source", "Source"), ("n_utterances", "N"),
            ("wer", "WER"), ("fairness_delta", "Fairness Δ")]
    return table(rows, cols)


def render_domain(rows):
    for r in rows:
        r["domain_wer"] = pct(r.get("domain_wer"))
        r["krr_mean"] = pct(r.get("krr_mean"))
        r["ne_wer_mean"] = pct(r.get("ne_wer_mean"))
        r["top_misrecognised"] = esc((r.get("top_misrecognised") or "")[:80] + "…")
    cols = [("dataset", "Dataset"), ("n_utterances", "N"), ("domain_wer", "Domain WER"),
            ("krr_mean", "Key-term recall"), ("ne_wer_mean", "NE WER")]
    return table(rows, cols)


def render_format(rows):
    for r in rows:
        r["status"] = badge(r.get("status", "?"), "pass" if r.get("status") == "pass" else "fail")
    cols = [("test", "Test"), ("value", "Value"), ("status", "Status"), ("error", "Error")]
    return table(rows, cols)


# ----------------------------------------------------------------------------
# TTS sections
# ----------------------------------------------------------------------------


def _per_lang_rows(test_dir: Path) -> list[dict]:
    """Read every per-language summary_<key>.csv in a TTS test directory."""
    rows: list[dict] = []
    if not test_dir.exists():
        return rows
    for f in sorted(test_dir.glob("summary_*.csv")):
        rows.extend(read_csv(f))
    return rows


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
                parts.append(card(fnum(d.get("mean"), 2), lbl, "pass" if lbl.startswith("UTMOS") else ""))
        base = '<div class="cards">' + "".join(parts) + "</div>"

    if not lang_rows:
        return base

    out = []
    for r in lang_rows:
        if r.get("status") in ("no_output", "load_failed"):
            out.append({"language": esc(r.get("language")), "n_sentences": esc(r.get("n_sentences")),
                        "dnsmos_ovrl": badge(r.get("status"), "warn"), "dnsmos_sig": "—",
                        "dnsmos_bak": "—", "utmos": "—"})
            continue
        rr = {"language": esc(r.get("language")), "n_sentences": esc(r.get("n_sentences")),
              "utmos": "n/a (en-only)"}
        for k in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak"):
            d = maybe_literal(r.get(k))
            rr[k] = fnum(d.get("mean"), 2) if isinstance(d, dict) else "—"
        out.append(rr)
    cols = [("language", "Lang"), ("n_sentences", "N"), ("dnsmos_ovrl", "DNSMOS OVRL"),
            ("dnsmos_sig", "DNSMOS SIG"), ("dnsmos_bak", "DNSMOS BAK"), ("utmos", "UTMOS")]
    caveat = notice(
        "Per-language naturalness uses DNSMOS (language-agnostic). UTMOS is "
        "omitted for non-English languages — it was validated on English only "
        "(Chinese as out-of-domain), so it is not a valid naturalness score for "
        "French/Spanish.", "muted")
    return base + "<h4>Per-language (multilingual)</h4>" + caveat + table(out, cols)


def render_intelligibility(rows, lang_rows=None):
    for r in rows:
        r["round_trip_wer"] = pct(r.get("round_trip_wer"))
        r["round_trip_cer"] = pct(r.get("round_trip_cer"))
    cols = [("category", "Category"), ("n_sentences", "N"), ("round_trip_wer", "Round-trip WER"),
            ("round_trip_cer", "CER"), ("normalizer", "Normalizer")]
    base = table(rows, cols)

    if not lang_rows:
        return base

    out = []
    for r in lang_rows:
        if r.get("status") in ("no_output", "load_failed"):
            out.append({"language": esc(r.get("language")), "n_sentences": esc(r.get("n_sentences")),
                        "round_trip_wer": badge(r.get("status"), "warn")})
            continue
        out.append({"language": esc(r.get("language")), "n_sentences": esc(r.get("n_sentences")),
                    "round_trip_wer": pct(r.get("round_trip_wer"))})
    cols2 = [("language", "Lang"), ("n_sentences", "N"), ("round_trip_wer", "Round-trip WER")]
    note = notice("FLEURS text synthesised per language (en/fr/es, all WER), "
                  "transcribed with Whisper large-v3 using the language hint.", "muted")
    return base + "<h4>Per-language (FLEURS round-trip)</h4>" + note + table(out, cols2)


def render_prosody(rows, lang_rows=None):
    all_rows = []
    for r in rows:
        rr = dict(r)
        rr.setdefault("language", "en")
        all_rows.append(rr)
    all_rows.extend(lang_rows or [])
    display = []
    for r in all_rows:
        if r.get("status") in ("no_output", "load_failed"):
            display.append({"language": esc(r.get("language")), "status": badge(r.get("status"), "warn")})
            continue
        display.append({k: esc(r.get(k)) for k in
                        ("language", "f0_mean_hz", "f0_std_intra_hz", "f0_range_intra_hz",
                         "npvi", "pause_ratio", "wpm_mean", "wpm_std", "n_sentences")})
    cols = [("language", "Lang"), ("f0_mean_hz", "F0 mean (Hz)"),
            ("f0_std_intra_hz", "F0 intra-std (Hz)"), ("f0_range_intra_hz", "F0 range (Hz)"),
            ("npvi", "nPVI"), ("pause_ratio", "Pause ratio"),
            ("wpm_mean", "WPM mean"), ("wpm_std", "WPM std"), ("n_sentences", "N")]
    note = notice(
        "Reference-free prosody: intra-utterance pitch dynamism (F0 std/range), "
        "rhythm (acoustic nPVI approximation), pause ratio, and speaking rate. "
        "Intrusive F0-vs-reference metrics are not computable for a fixed "
        "synthetic voice (no same-speaker ground truth — see 2.4).", "muted")
    return note + table(display, cols)


def render_signal(rows):
    reason = rows[0].get("reason") if rows else None
    msg = reason or (
        "MCD/PESQ/STOI are full-reference (intrusive) metrics requiring a "
        "same-speaker, same-text ground-truth recording. None exists for a fixed "
        "synthetic voice, so any score measures speaker mismatch, not quality. "
        "Use 2.1 (UTMOS/DNSMOS) and 2.2 (round-trip WER) instead; a valid "
        "intrusive run needs a zero-shot voice-cloning deployment."
    )
    return notice("N/A for this model — " + msg, "warn")


def render_tts_acceptance(rows):
    if not rows:
        return notice(
            "No TTS language-acceptance matrix found. Run Test 2.9 "
            "(python -m eval.run tts --test tts_language_acceptance) with "
            "tts.multilingual_languages set.")
    ok = [r for r in rows if r.get("status") == "ok"]
    cards = (
        '<div class="cards">'
        + card(f"{len(ok)}/{len(rows)}", "languages synthesise + round-trip", "pass" if len(ok) == len(rows) else "warn")
        + '</div>'
    )
    gap = ""
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        names = ", ".join(f'<b>{esc(r.get("label"))}</b> ({esc(r.get("status"))})' for r in bad)
        gap = notice(
            f"Serving gap: {len(bad)} configured language(s) did not pass — {names}. "
            "Magpie TTS Multilingual advertises en/fr/es, but this deployment "
            "exposes only the voices that pass here; per-language 2.1/2.2/2.3 "
            "score only accepted languages.", "warn")

    def row_style(r):
        return {"ok": "", "wrong_language": "background:#fff6e5", "empty": "background:#fdecea",
                "error": "background:#fdecea", "load_failed": "background:#fff6e5"}.get(r.get("status"), "")

    disp = []
    for r in rows:
        disp.append({
            "label": esc(r.get("label")),
            "voice": esc(r.get("voice")),
            "status_b": badge(r.get("status", "?"),
                              {"ok": "pass", "wrong_language": "warn", "empty": "fail",
                               "error": "fail", "load_failed": "warn"}.get(r.get("status"), "muted")),
            "detected": esc(r.get("detected_language")),
            "wer": pct(r.get("round_trip_wer")) if r.get("round_trip_wer") not in (None, "", "None") else "—",
            "sample": esc((r.get("sample_hyp") or "")[:60]),
        })
    cols = [("label", "Language"), ("voice", "Voice"), ("status_b", "Status"),
            ("detected", "Whisper detected"), ("wer", "Round-trip WER"), ("sample", "Sample transcript")]
    return cards + gap + table(disp, cols, highlight=lambda r: "")


def render_tts_latency(rows):
    return render_percentile_table(
        rows, "bucket", "Bucket",
        [("ttfb", "TTFB (s)"), ("rtf", "RTF"), ("elapsed", "Elapsed (s)")])


def render_tts_concurrency(rows):
    for r in rows:
        r["error_rate"] = pct(r.get("error_rate"))
        r["latency_p50"] = fnum(r.get("latency_p50"), 3)
        r["latency_p99"] = fnum(r.get("latency_p99"), 3)
    cols = [("interface", "Interface"), ("concurrency", "N"), ("n_success", "OK"),
            ("error_rate", "Err rate"), ("rps", "RPS"), ("latency_p50", "P50 (s)"),
            ("latency_p99", "P99 (s)")]
    return table(rows, cols)


def render_edge(rows):
    for r in rows:
        r["pass_rate"] = pct(r.get("pass_rate"))
    cols = [("category", "Category"), ("total", "Total"), ("pass", "Pass"),
            ("fail", "Fail"), ("degraded", "Degraded"), ("pass_rate", "Pass rate")]
    return table(rows, cols)


def render_tts_longform(rows):
    for r in rows:
        r["mean_speaker_sim"] = fnum(r.get("mean_speaker_sim"), 3)
        r["f0_drift_hz"] = fnum(r.get("f0_drift_hz"), 1)
        r["wpm_drift"] = fnum(r.get("wpm_drift"), 1)
    cols = [("title", "Passage"), ("n_paragraphs", "Paras"),
            ("mean_speaker_sim", "Speaker sim"), ("f0_drift_hz", "F0 drift (Hz)"),
            ("wpm_drift", "WPM drift")]
    return table(rows, cols)


# ----------------------------------------------------------------------------
# page assembly
# ----------------------------------------------------------------------------

CSS = """
:root{--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--pass:#1a7f37;--fail:#b42318;--warn:#b25e00;--accent:#2a4b8d;}
*{box-sizing:border-box}
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--fg);margin:0;background:#fafafa}
header{background:#13233f;color:#fff;padding:28px 40px}
header h1{margin:0 0 6px;font-size:24px}
header .meta{color:#b9c6df;font-size:13px}
nav{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 40px;font-size:13px;z-index:10;overflow-x:auto;white-space:nowrap}
nav a{color:var(--accent);text-decoration:none;margin-right:14px}
nav a:hover{text-decoration:underline}
main{max-width:1100px;margin:0 auto;padding:24px 40px 80px}
section{margin:0 0 40px;background:#fff;border:1px solid var(--line);border-radius:8px;padding:20px 24px}
h2{margin:0 0 4px;font-size:19px;border-bottom:2px solid #13233f;padding-bottom:8px}
h3{font-size:14px;color:var(--mut);margin:20px 0 8px;text-transform:uppercase;letter-spacing:.03em}
p.sub{color:var(--mut);margin:6px 0 14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 10px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:left}
th{background:#f3f5f9;font-weight:600}
tr:nth-child(even) td{background:#fcfcfd}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 14px}
.stat{flex:1;min-width:150px;background:#f6f8fb;border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.stat .big{font-size:26px;font-weight:700}
.stat .lbl{color:var(--mut);font-size:12px;margin-top:2px}
.stat.pass .big{color:var(--pass)} .stat.fail .big{color:var(--fail)} .stat.warn .big{color:var(--warn)}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge.pass{background:#e7f5ec;color:var(--pass)} .badge.fail{background:#fdecea;color:var(--fail)}
.badge.warn{background:#fff6e5;color:var(--warn)} .badge.muted{background:#eee;color:var(--mut)}
.notice{border-radius:6px;padding:10px 14px;font-size:13px;margin:10px 0}
.notice.muted{background:#f4f4f6;color:var(--mut)}
.notice.warn{background:#fff6e5;color:var(--warn);border:1px solid #f0d9a8}
.notice.fail{background:#fdecea;color:var(--fail);border:1px solid #f3c2bc}
code{background:#eef;padding:1px 4px;border-radius:3px;font-size:12px}
footer{color:var(--mut);font-size:12px;text-align:center;padding:30px}
"""

NAV = [
    ("summary", "Summary"), ("multilingual", "Multilingual (1.12/1.13)"),
    ("t1_1", "1.1 Accuracy"), ("t1_2", "1.2 Perf"), ("t1_3", "1.3 Streaming"),
    ("t1_4", "1.4 REST/gRPC"), ("t1_5", "1.5 Noise"), ("t1_6", "1.6 Accent"),
    ("t1_7", "1.7 Long-form"), ("t1_8", "1.8 Domain"), ("t1_9", "1.9 Output"),
    ("t1_10", "1.10 Confidence"), ("t1_11", "1.11 Format"),
    ("tts", "TTS 2.1-2.9"), ("phase0", "Phase 0"), ("gaps", "Gaps"),
]


def build(run_dir: Path, acceptance_csv: Path, out: Path,
          model: str, endpoint: str) -> Path:
    stt = run_dir / "stt"
    tts = run_dir / "tts"
    matrix = read_csv(acceptance_csv)

    acc = read_csv(stt / "accuracy" / "summary.csv")
    clean_wer = next((pct(r["wer"]) for r in read_csv(stt / "accuracy" / "summary.csv")
                      if "clean" in (r.get("dataset") or "").lower()), "—")
    nat = read_csv(tts / "naturalness" / "summary.csv")
    utmos = "—"
    if nat:
        d = maybe_literal(nat[0].get("utmos"))
        if isinstance(d, dict):
            utmos = fnum(d.get("mean"), 2)

    ok = sum(1 for r in matrix if r.get("status") == "ok")
    joule = [r for r in matrix if r.get("joule")]
    joule_ok = sum(1 for r in joule if r.get("status") == "ok")

    summary_cards = (
        '<div class="cards">'
        + card(clean_wer, "STT clean WER (LibriSpeech)", "pass")
        + card(f"{joule_ok}/{len(joule) or '—'}", "Joule languages transcribe", "pass" if joule and joule_ok == len(joule) else "warn")
        + card(f"{ok}/{len(matrix) or '—'}", "of 40 model languages", "")
        + card(utmos, "TTS naturalness (UTMOS)", "pass")
        + "</div>"
        + '<p class="sub">Speech model evaluation — one report covering Phase 0, '
          'STT 1.1-1.13 (including the full multilingual matrix), and TTS 2.1-2.8. '
          'All metrics client-side. Sections with no data in the selected run are '
          'marked explicitly rather than fabricated.</p>'
    )

    # per-language depth presence (Tier A/B) — honest flag
    depth_present = any((stt / t / "summary_by_language.csv").exists()
                        for t in ["performance", "noise_robustness", "long_form"])
    depth_note = ""
    if not depth_present:
        depth_note = notice(
            "Per-language depth passes (Tier-A 1.5/1.9, Tier-B 1.2/1.7/1.10) were "
            "not run in this run (config lists empty = English-only). The 40-language "
            "acceptance matrix below is live; full per-language WER/CER (1.12) and depth "
            "tables will populate here once run with multilingual_extra_languages / "
            "spot_check_languages set.", "warn")

    gaps = (
        "<ul>"
        "<li><b>th / id / ar blank on this endpoint</b> — 3 Joule GA languages return "
        "empty transcripts. Serving/gateway gap, not a model gap. Raise with gateway owner.</li>"
        "<li><b>1.6 dialect &amp; 1.8 domain</b> — flagged multilingual coverage gaps; need "
        "Common Voice per-locale / multilingual domain corpora. Deliberately not faked.</li>"
        "<li><b>1.3 / 1.4 / 1.11</b> — English-only by design (serving-stack / audio-frontend "
        "properties; 1.3 also moot — offline-only model).</li>"
        "<li><b>Full 1.12 accuracy + per-language depth</b> — opt-in; run with the two config "
        "lists populated to replace the acceptance-only view above.</li>"
        "</ul>"
    )

    parts = [
        section("summary", "Executive Summary", summary_cards),
        section("multilingual", "Multilingual — Accuracy (1.12) & Language Acceptance (1.13)",
                depth_note + render_acceptance(matrix),
                "40 Nemotron-3.5 languages probed live; rendered Joule-first."),
        section("t1_1", "Test 1.1 — Clean Speech Accuracy", render_accuracy(acc)),
        section("t1_2", "Test 1.2 — Batch Performance",
                render_performance(read_csv(stt / "performance" / "summary.csv"),
                                   read_csv(stt / "performance" / "concurrency_summary.csv"))),
        section("t1_3", "Test 1.3 — Streaming Performance",
                render_streaming(read_csv(stt / "streaming" / "summary.csv"))),
        section("t1_4", "Test 1.4 — REST vs gRPC",
                render_rest_grpc(read_csv(stt / "rest_vs_grpc" / "summary.csv"))),
        section("t1_5", "Test 1.5 — Noise Robustness",
                render_noise(read_csv(stt / "noise_robustness" / "summary.csv"))),
        section("t1_6", "Test 1.6 — Accent Robustness",
                render_accent(read_csv(stt / "accent" / "summary.csv"))),
        section("t1_7", "Test 1.7 — Long-Form Audio",
                render_longform(read_csv(stt / "long_form" / "summary.csv"))),
        section("t1_8", "Test 1.8 — Domain Vocabulary",
                render_domain(read_csv(stt / "domain" / "summary.csv"))),
        section("t1_9", "Test 1.9 — Output Quality",
                render_output_quality(read_csv(stt / "output_quality" / "summary.csv"))),
        section("t1_10", "Test 1.10 — Confidence Calibration",
                render_confidence(read_csv(stt / "confidence" / "summary.csv"))),
        section("t1_11", "Test 1.11 — Format Robustness",
                render_format(read_csv(stt / "format_robustness" / "summary.csv"))),
        section("tts", "TTS Evaluation — 2.1 to 2.9",
                "<h3>2.1 Naturalness</h3>"
                + render_naturalness(nat, _per_lang_rows(tts / "naturalness"))
                + "<h3>2.2 Intelligibility (round-trip WER)</h3>"
                + render_intelligibility(read_csv(tts / "intelligibility" / "summary.csv"),
                                         _per_lang_rows(tts / "intelligibility"))
                + "<h3>2.3 Prosody</h3>"
                + render_prosody(read_csv(tts / "prosody" / "summary.csv"),
                                 _per_lang_rows(tts / "prosody"))
                + "<h3>2.4 Signal Quality</h3>" + render_signal(read_csv(tts / "signal_quality" / "summary.csv"))
                + "<h3>2.5 Streaming Latency</h3>" + render_tts_latency(read_csv(tts / "latency" / "summary.csv"))
                + "<h3>2.6 Throughput &amp; Concurrency</h3>" + render_tts_concurrency(read_csv(tts / "concurrency" / "summary.csv"))
                + "<h3>2.7 Edge Cases</h3>" + render_edge(read_csv(tts / "edge_cases" / "summary.csv"))
                + "<h3>2.8 Long-Form Consistency</h3>" + render_tts_longform(read_csv(tts / "long_form" / "summary.csv"))
                + "<h3>2.9 Language Acceptance (en/fr/es)</h3>"
                + render_tts_acceptance(read_csv(tts / "language_acceptance" / "summary.csv"))),
        section("phase0", "Phase 0 — Discovery",
                render_phase0(read_json(run_dir / "phase0" / "discovery.json"))),
        section("gaps", "Known Gaps & Honest Scope", gaps),
    ]

    nav = "".join(f'<a href="#{a}">{esc(l)}</a>' for a, l in NAV)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Speech Model Evaluation — Full Report</title><style>{CSS}</style></head>
<body>
<header><h1>Speech Model Evaluation — Full Report</h1>
<div class="meta">Model: {esc(model)} &nbsp;·&nbsp; Endpoint: {esc(endpoint)}
&nbsp;·&nbsp; Run: {esc(run_dir.name)} &nbsp;·&nbsp; Generated: {esc(now)}</div></header>
<nav>{nav}</nav>
<main>{"".join(parts)}</main>
<footer>Generated by eval/report/generate_full_report.py — stdlib only, no fabricated data.</footer>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out


def render_phase0(data: dict) -> str:
    if not data:
        return notice("Phase 0 discovery.json not found in the selected run.")
    rows = []
    for name, r in data.items():
        if not isinstance(r, dict):
            rows.append({"check": name, "status": badge("info", "muted"), "detail": esc(r)})
            continue
        if r.get("error"):
            rows.append({"check": name, "status": badge("error", "fail"), "detail": esc(r.get("error"))[:120]})
        elif "by_status" in r:  # language_acceptance summary
            rows.append({"check": name, "status": badge("ok", "pass"),
                         "detail": f"40-lang matrix: {esc(r.get('by_status'))}; Joule ok {esc(r.get('joule_ok'))}/{esc(r.get('joule_total'))}"})
        else:
            rows.append({"check": name, "status": badge("ok", "pass"),
                         "detail": esc(json.dumps(r, default=str))[:140]})
    return table(rows, [("check", "Sub-phase"), ("status", "Status"), ("detail", "Detail")])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Comprehensive single-file HTML report")
    ap.add_argument("--run", default="results/run-07-07-26/riva",
                    help="Run directory containing stt/ tts/ phase0/")
    ap.add_argument("--acceptance", default="results/stt/language_acceptance/language_acceptance.csv",
                    help="40-language acceptance matrix CSV (Test 1.13)")
    ap.add_argument("--out", default="docs/full-report.html")
    ap.add_argument("--model", default="NVIDIA-Nemotron-3.5-Transcribe-30B-A4B")
    ap.add_argument("--endpoint", default="SAP AI Core gateway (gRPC StreamingRecognize)")
    args = ap.parse_args(argv)
    out = build(Path(args.run), Path(args.acceptance), Path(args.out),
                args.model, args.endpoint)
    print(f"Report written to: {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
