# Multilingual Test-Gap Analysis — STT Tests 1.2–1.11

**Question asked:** now that we evaluate 40 languages for *accuracy* (Test 1.12) and
*acceptance* (Test 1.13), do the **other** STT tests (1.2–1.11) also need per-language
versions, and must their **datasets be enhanced** to measure each language properly?

**Short answer:** No, not uniformly. Each test's language-sensitivity depends on whether it
measures a property of the **model's linguistic ability** (varies by language — needs
per-language data) or a property of the **serving stack / audio frontend** (language-neutral —
one representative language is enough). Below, every test is classified, with the concrete
dataset change needed where it matters.

> **Gating principle first.** This is a single SAP AI Core gateway deployment that implements
> only `StreamingRecognize`. The Phase-0 acceptance probe (1.13) shows the endpoint
> returns **empty transcripts for Thai, Arabic and Indonesian** even though all three are
> Nemotron-3.5 GA languages. So the order of operations is: **1.13 acceptance → 1.12 accuracy on
> accepted languages → per-language depth tests only on the subset that actually produces
> output.** There is no value in building a per-language noise test for a language the endpoint
> transcribes as blank — so every per-language test records a graceful `no_output` row for
> those languages instead of a fabricated score.

---

## Classification of tests 1.2–1.11

| Test | What it measures | Current data (language) | Language-sensitive? | Recommendation |
|------|------------------|-------------------------|---------------------|----------------|
| **1.2 Performance (RTF/RTFx)** | Batch throughput / speed | LibriSpeech clean (EN) | **Partially.** This is an autoregressive LLM-ASR; decode time scales with *output token count*, which differs by script (CJK dense, morphologically-rich langs longer). Speed is otherwise language-neutral. | **Spot-check** 4–5 representative langs (en, de, ja, th, ar). Reuse FLEURS audio. Also report output-tokens-per-audio-second to expose decoder-cost variance. Not all 40. |
| **1.3 Streaming (TTFW/stability)** | Partial/final latency, stability | LibriSpeech + VoxPopuli (EN) | **Low / N/A.** Nemotron-3.5-Transcribe is **offline-only** (30 s internal chunking); real-time streaming is a *different* model (`nemotron-3.5-asr-streaming-0.6b`). TTFW here is the gateway's fallback behaviour, a serving property. | **Keep English-only.** Add a note that streaming semantics are stack-level, not model-linguistic. |
| **1.4 REST vs gRPC** | Transport parity | LibriSpeech clean (EN) | **No.** Pure transport comparison; identical bytes regardless of language. (And this gateway only does gRPC streaming anyway.) | **Keep English-only.** |
| **1.5 Noise robustness** | WER degradation vs SNR | LibriSpeech clean + synthetic noise (EN) | **Yes.** Noise sensitivity varies by language (phoneme density, per-language training-data volume). A model robust in EN can be fragile in a lower-resource language. | **Per-language (Joule subset).** Cheap: mix the same SNR noise onto **FLEURS clips per language** — no new corpus needed, reuse the existing SNR mixer. |
| **1.6 Accent & dialect** | WER across accents | L2-Arctic (L2 *English* accents) | **Yes, but reframed.** The multilingual analog is **regional dialects within a language** — exactly where the `es_419` vs `es-ES` and `ar_eg` vs MSA mismatches bite. FLEURS gives one dialect per language, so it *cannot* measure this. | **New data needed** (Common Voice has per-locale accents). **Flag as a coverage gap**; do not fake it with FLEURS. Lower priority than 1.5/1.9. |
| **1.7 Long-form** | Drift/hallucination on long audio | VoxPopuli EN + concatenated LibriSpeech | **Yes.** Repetition/hallucination under long audio is typically worse for lower-resource languages. | **Spot-check** 3–5 langs. Reuse the concatenation trick on **FLEURS clips per language** (concat ~10 clips → ~5 min). Actionable, no new corpus. |
| **1.8 Domain vocabulary** | WER on domain terms (medical, etc.) | English domain datasets | **Yes, inherently.** Domain vocabulary is language-specific; an English medical-term test says nothing about German medical transcription. | **Flag as a gap.** Per-language domain corpora barely exist publicly. Document the limitation rather than produce a misleading English-only verdict labelled "domain". |
| **1.9 Output quality (punct/caps/numbers)** | Formatting correctness | English datasets, English rules | **Highest language-sensitivity of all.** Capitalisation (German capitalises every noun; CJK/Thai have no case), number formatting (decimal comma vs point), punctuation (Spanish `¿¡`, CJK full-width `。、`, Arabic `؟`). The current logic is English-centric and would emit **wrong verdicts** in other languages. | **Per-language (Joule subset), with per-language rules.** **Dataset change required:** use FLEURS **`raw_transcription`** (keeps original casing/punctuation/numerals) as reference — the default `transcription` field is pre-lowercased and useless here. |
| **1.10 Confidence calibration** | Reliability of confidence scores | LibriSpeech + GigaSpeech (EN) | **Moderate.** Calibration can drift per language (over/under-confident on lower-resource langs), but this is second-order. | **Spot-check** 3–5 langs on FLEURS. Not all 40. |
| **1.11 Format robustness** | Container/codec/sample-rate handling | Fixed clip, format-mutated | **No.** Entirely in the audio frontend (decode → resample). Language-neutral. | **Keep English-only.** |

---

## Does the dataset need enhancing? — Yes, in three specific ways

1. **FLEURS is read speech, and that is its ceiling.** FLEURS covers all 40 languages for a
   clean WER/CER baseline (Test 1.12), but it is scripted, clean, single-dialect audio. It
   **cannot** exercise noise (1.5), dialects (1.6), long-form drift (1.7), domain vocabulary
   (1.8), or spontaneous speech. For the language-sensitive tests we either *derive* harder
   conditions from FLEURS (add noise, concatenate for length — cheap, in-repo) or bring new
   corpora (dialects, domain — expensive, flagged as gaps).

2. **Use the right FLEURS reference field per test.** `transcription` is lowercased/normalised
   (fine for WER); `raw_transcription` preserves case, punctuation and numerals (required for
   the 1.9 output-quality test). Picking the wrong field silently invalidates the measurement.

3. **Measurement validity is partly a *normaliser* problem, not only a data problem.** Scoring
   must use language-correct normalisation or the numbers are meaningless regardless of how
   good the dataset is. Concrete example already found and fixed: this endpoint emits
   **one space per CJK character** (`イ ン タ ー ネ ッ ト`); without stripping whitespace
   before CER the Japanese score was a spurious **0.97**, versus a correct **~0.0** after the
   fix (`eval/utils.py::score_pair`, CER branch). Equivalent per-language rules are needed for
   1.9 (caps/number/punctuation conventions).

---

## Recommended tiering (pragmatic, given data scarcity + single-endpoint reality)

- **Tier A — run per-language on the Joule subset (18) that passes acceptance:**
  1.12 accuracy (done), **1.9 output-quality** (needs `raw_transcription` + per-language rules),
  **1.5 noise robustness** (FLEURS + existing SNR mixer).
- **Tier B — spot-check 4–5 representative languages (en, de, ja, th, ar):**
  1.2 performance, 1.7 long-form, 1.10 confidence. Representative, not exhaustive.
- **Tier C — keep English-only (language-neutral):**
  1.3 streaming (also N/A: offline model), 1.4 REST-vs-gRPC, 1.11 format robustness.
- **Flagged data gaps (document, don't fake):**
  1.6 dialect coverage (needs Common Voice per-locale), 1.8 domain vocabulary (no multilingual
  corpus). The `es_419`/`es-ES` and `ar_eg`/MSA mismatches are instances of the 1.6 gap.

This keeps the harness honest: it adds per-language depth exactly where the *model's* behaviour
varies, spot-checks where variance is plausible but secondary, and refuses to dress up an
English-only or single-dialect test as multilingual coverage it does not have.

---

## Implementation status (what was built)

Implemented at scope **Tier A + Tier B + gap docs**, additively. **Existing English tests run
byte-for-byte unchanged** — every new language is opt-in via config and defaults to empty.

**Config (opt-in, both default `[]` in `config.yaml` / `STTConfig`):**
- `stt.multilingual_extra_languages` — drives the Tier-A per-language passes (1.5, 1.9).
- `stt.spot_check_languages` — drives the Tier-B spot-checks (1.2, 1.7, 1.10).
- Empty list ⇒ only the original English path runs ⇒ today's behaviour, same datasets, same
  output paths (`calls.jsonl`, `summary.csv`), same record shapes.

**Shared code:**
- `eval/stt/fleurs_common.py` — single FLEURS code path: `resolve_languages`, `fleurs_load`,
  `get_audio_and_ref` (`field="raw_transcription"` for 1.9), `pcm16_bytes`. `multilingual.py`
  (1.12) was refactored onto these helpers (behaviour-preserving).
- `eval/stt/text_quality_rules.py` — per-language `LANG_RULES` for 1.9: which punctuation marks
  to score (Latin `. , ? !`, Spanish `¿¡`, CJK full-width `。、？！`, Arabic `؟،`, Greek `·;`),
  `has_case` (False ⇒ capitalisation reported **N/A**, never a false 0), and `ascii_digits`
  (False ⇒ number accuracy **N/A** for native-digit scripts).

**Tier A (per-language on the opt-in Joule subset):**
- **1.5 noise robustness** — same noise×SNR grid on FLEURS audio (the SNR injectors are reused
  verbatim), scored with `score_pair` so CJK/Thai get CER. Per-language `<key>.jsonl`; rows
  tagged `language`.
- **1.9 output quality** — uses FLEURS `raw_transcription`; language-aware punctuation (char-
  level for CER scripts), capitalisation N/A for unicameral scripts, number accuracy N/A for
  native-digit scripts.

**Tier B (spot-check via `spot_check_languages`):**
- **1.2 performance** — RTF/RTFx per duration bucket on FLEURS, plus **output cost**
  (transcript chars and tokens per audio-second) to expose decoder-cost variance across scripts.
- **1.7 long-form** — concatenates ~10 FLEURS clips per language; hallucination/repetition
  detectors switch to character n-grams and are flagged `detector_approximate: true` for CER
  languages rather than reporting a bogus word-based rate.
- **1.10 confidence** — ECE/AUROC/Brier per language (logic factored into one shared helper);
  language-tagged item IDs and per-language JSONL so resume never collides; results under
  `results["per_language"]`.

**Honest gate everywhere:** any language that returns empty transcripts (Thai, Arabic,
Indonesian on this endpoint) records a `no_output` row/status instead of a fabricated score.

**Flagged gaps (documented, deliberately not faked):** 1.6 dialect fairness (needs Common Voice
per-locale; `es-ES`/`es-419` and MSA/`ar-EG` live here) and 1.8 domain vocabulary (no
multilingual domain corpus). Module docstrings in `accent.py`/`domain.py` state this.
`streaming.py` (1.3), `rest_vs_grpc.py` (1.4), `format_robustness.py` (1.11) carry docstrings
noting they are intentionally English-only (serving-stack / audio-frontend properties).

---

## TTS multilingual (Tests 2.1 / 2.2 / 2.3 / 2.9)

The TTS model under test — **Magpie TTS Multilingual** — is a different situation from the STT
model, but the deployment turns out to serve a broad language set.

**Twelve languages, each with its own voice.** The SAP AI Core Magpie deployment serves **12
languages** — CONFIRMED live 2026-11-05 with one `GetRivaSynthesisConfig` call (the authoritative
Riva RPC that enumerates served models/voices): it advertises
`en-US, es-US, fr-FR, de-DE, zh-CN, vi-VN, it-IT, hi-IN, ja-JP, ko-KR, ar-XA, pt-BR` with **139
subvoices**, and a synthesis smoke test rendered audio for **11/12 immediately** (only `ar-XA`
rejected the *Latin* probe sentence — its grapheme-to-phoneme front-end needs Arabic script, and
synthesises normally from FLEURS Arabic). This corrects an earlier wrong assumption that the model
served only en/fr/es: the real cause of the first multilingual run's non-English failures was
**fabricated voice strings** (`…FR-FR.Female.Female-1`, `…ES-US.Female.Female-1`) that do not
exist on the server — not a serving gap. Each language has its *own* voice; the server accepts the
format `Magpie-Multilingual.<LOCALE>.<Name>` (e.g. `Magpie-Multilingual.FR-FR.Aria`), so driving
non-English synthesis requires setting both a per-language **voice** and **locale** on the request,
not just a language hint. Most languages are space-delimited and scored with **WER**; **Chinese
and Japanese** have no word boundaries and are scored with **CER** (Korean stays WER — it is
space-delimited at the eojeol level). The language table lives in `eval/tts/tts_languages.py`
(re-probe any time with `scripts/probe_tts_voices.py --synth`); voice names are overridable per
deployment via `tts.voice_overrides`.

**What industry measures for multilingual TTS, and what we adopted:**
- **Intelligibility = ASR round-trip WER/CER** (Whisper large-v3) is the primary, language-
  portable metric. → Test **2.2** synthesizes FLEURS text per language and round-trips it (WER
  for most languages, CER for Chinese/Japanese — the routing in `eval/utils.py` is shared with STT).
- **Naturalness = DNSMOS** (language-agnostic signal-quality MOS predictor). → Test **2.1**
  reports DNSMOS for every language. **UTMOS is English-biased** — it was validated on English
  with Chinese only as an out-of-domain check — so we **do not** report it as a naturalness
  score for the non-English languages (recorded as `utmos: null` with a note). This is the single
  most important honesty gate on the TTS side.
- **Language-ID as an acceptance gate.** → Test **2.9** (TTS analogue of 1.13) synthesizes a few
  FLEURS sentences per configured language and checks, via Whisper's detected language, that the
  deployment actually produces that language. The live probe already showed all 12 configured
  languages render; 2.9 is the per-run, per-utterance gate that records `ok | wrong_language |
  empty | error` per language, so any future regression (or a script-mismatch case like feeding
  the Arabic voice non-Arabic text) surfaces as an explicit status, not a faked score.
- **Speaker-similarity** (WavLM/ECAPA) is only meaningful for voice *cloning*, which this model
  does not do, so it is not used for multilingual scoring here (2.8 already tracks intra-voice
  drift for long-form English).

**Why 2.3 Prosody is reference-free (and 2.4 is N/A).** Both MCD/PESQ/STOI (2.4) and
F0-RMSE/F0-correlation "vs. reference" (the old 2.3) are **intrusive** full-reference metrics:
they need a recording of the *same speaker* uttering the *same text*, aligned to the output. A
fixed synthetic persona ("Aria") has no such human reference, so these are impossible here and
were being computed against a mismatched speaker (meaningless). 2.4 is therefore marked **N/A**,
and 2.3 was rewritten to the **reference-free** prosody signals that *are* valid and useful —
intra-utterance F0 std/range (pitch dynamism), acoustic nPVI (rhythm), pause ratio, WPM — which
flag monotone/robotic output and are language-neutral. The only legitimate way to run the
intrusive metrics on this kind of model is zero-shot voice cloning against held-out real-speaker
audio, which needs a different NIM deployment (kept behind a guard in `signal_quality.py`).

**Opt-in, English unchanged.** As with STT, `tts.multilingual_languages` defaults to `[]`; empty
⇒ only the original English path runs (2.1/2.2 byte-for-byte). 2.3 and 2.4 English output changed
*on purpose* (2.3's old metric was acknowledged-wrong; 2.4 became N/A). Per-language results are
written to `results/tts/<test>/<key>.jsonl` + `summary_<key>.csv`, tagged with `language`.
