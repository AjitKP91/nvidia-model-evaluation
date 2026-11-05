#!/usr/bin/env python3
"""Probe the live Magpie TTS deployment on SAP AI Core for its real voice list.

Why this exists
---------------
The run showed every non-English synthesis failing with
``INVALID_ARGUMENT "Model is not available on server: subvoice requested not
found"``. Rather than guess 12 languages' worth of voice-name strings and fire
blind synthesis calls (each would fail on a name mismatch regardless of whether
the language is served), we ask the server directly.

Riva TTS exposes ``GetRivaSynthesisConfig`` — ONE gRPC call that returns every
model/voice/subvoice the deployment actually serves. That is the authoritative
answer to "which of Magpie's 12 languages does this AI Core deployment expose".

Credentials (same as the harness; values live in a git-ignored .env or the shell):
  export AICORE_BEARER_TOKEN=<jwt>                 # static, ~4h
  # OR client-credentials (auto-mint):
  export AICORE_CLIENT_ID=... AICORE_CLIENT_SECRET=... AICORE_AUTH_URL=...

Usage:
  .venv-probe/bin/python scripts/probe_tts_voices.py                 # list voices (1 call)
  .venv-probe/bin/python scripts/probe_tts_voices.py --synth         # + 1 synth per subvoice
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import certifi
import requests
import yaml

import riva.client
import riva.client.proto.riva_tts_pb2 as riva_tts

REPO = Path(__file__).resolve().parent.parent


def load_env(dotenv: Path) -> None:
    """Minimal .env loader (real shell vars win), mirroring config._load_dotenv."""
    if not dotenv.exists():
        return
    for raw in dotenv.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ and v:
            os.environ[k] = v


def mint_token() -> str:
    static = os.environ.get("AICORE_BEARER_TOKEN", "")
    if static:
        print("auth: using static AICORE_BEARER_TOKEN", file=sys.stderr)
        return static
    cid = os.environ.get("AICORE_CLIENT_ID", "")
    csec = os.environ.get("AICORE_CLIENT_SECRET", "")
    aurl = os.environ.get("AICORE_AUTH_URL", "")
    if not (cid and csec and aurl):
        sys.exit(
            "ERROR: no credentials. Set AICORE_BEARER_TOKEN, OR "
            "AICORE_CLIENT_ID / AICORE_CLIENT_SECRET / AICORE_AUTH_URL "
            "(in .env or the shell)."
        )
    print("auth: minting token via client-credentials", file=sys.stderr)
    resp = requests.post(
        aurl.rstrip("/") + "/oauth/token",
        data={"grant_type": "client_credentials"},
        auth=(cid, csec),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true",
                    help="after listing, try one short synth per discovered subvoice")
    args = ap.parse_args()

    load_env(REPO / ".env")
    cfg = yaml.safe_load((REPO / "eval" / "config.yaml").read_text())
    tts = cfg["tts"]
    riva_cfg = cfg["riva"]
    uri = tts["grpc_uri"]
    deployment_id = tts.get("deployment_id", "")
    resource_group = riva_cfg.get("resource_group", "default")

    token = mint_token()
    auth = riva.client.Auth(uri=uri, use_ssl=True, ssl_root_cert=certifi.where())
    md = [("ai-resource-group", resource_group)]
    if deployment_id:
        md.append(("deployment-id", deployment_id))
    md.append(("authorization", f"Bearer {token}"))
    auth.metadata = md
    service = riva.client.SpeechSynthesisService(auth)

    print(f"\n=== GetRivaSynthesisConfig  (uri={uri}  deployment={deployment_id}) ===")
    try:
        req = riva_tts.RivaSynthesisConfigRequest()
        resp = service.stub.GetRivaSynthesisConfig(req, metadata=auth.metadata)
    except Exception as exc:  # noqa: BLE001
        print(f"GetRivaSynthesisConfig FAILED: {type(exc).__name__}: {exc}")
        sys.exit(2)

    # voice_name to send on a SynthesizeSpeechRequest is "<prefix>.<SUBVOICE>"
    # (e.g. the known-good English "Magpie-Multilingual.EN-US.Aria"), and
    # language_code is a SINGLE locale (e.g. "fr-FR"), NOT the comma-joined list
    # the server reports in its config. We derive both from each subvoice entry
    # "LOCALE.Name[.Emotion]:index" -> locale "FR-FR", full voice
    # "<prefix>.FR-FR.Name".
    def locale_to_lang(loc: str) -> str:  # "FR-FR" -> "fr-FR"
        parts = loc.split("-")
        return parts[0].lower() + ("-" + "-".join(parts[1:]) if len(parts) > 1 else "")

    # per-locale: list of full voice strings (base voices first, emotions after)
    by_locale: dict[str, list[str]] = {}
    prefix = "Magpie-Multilingual"
    n_models = len(resp.model_config)
    print(f"server returned {n_models} model config entry(ies)\n")
    for mc in resp.model_config:
        params = dict(mc.parameters)
        prefix = params.get("voice_name", prefix)
        subvoices = params.get("subvoices", "")
        print(f"- model_name : {mc.model_name}")
        print(f"  language_code (served): {params.get('language_code', '?')}")
        print(f"  voice_name prefix     : {prefix}")
        print(f"  subvoice count        : {len(subvoices.split(',')) if subvoices else 0}")
        print()
        for sv in subvoices.split(","):
            sv = sv.strip()
            if not sv:
                continue
            base = sv.split(":")[0]                 # "FR-FR.Aria.Calm"
            loc = base.split(".")[0]                # "FR-FR"
            full = f"{prefix}.{base}"               # "Magpie-Multilingual.FR-FR.Aria.Calm"
            by_locale.setdefault(loc, []).append(full)

    locales = list(by_locale.keys())
    total = sum(len(v) for v in by_locale.values())
    print(f"=== SUMMARY: {total} subvoice(s) across {len(locales)} locale(s) ===")
    print(f"locales served: {locales}\n")
    for loc in locales:
        voices = by_locale[loc]
        # show base (emotionless) voices = those with exactly one dot after locale
        base_voices = [v for v in voices if v.count(".") == 2]
        print(f"  {loc:6s} ({len(voices):2d} subvoices)  base: {[v.split('.')[-1] for v in base_voices]}")

    if args.synth and by_locale:
        print("\n=== synth probe: one representative voice per LANGUAGE (correct format) ===")
        ok, err = [], []
        for loc in locales:
            # pick first base (emotionless) voice, else first subvoice
            base_voices = [v for v in by_locale[loc] if v.count(".") == 2]
            voice = (base_voices or by_locale[loc])[0]
            lang = locale_to_lang(loc)
            try:
                r = service.synthesize(
                    "This is a test of the text to speech system.",
                    voice_name=voice, language_code=lang,
                    sample_rate_hz=int(tts.get("sample_rate", 22050)),
                )
                print(f"  OK    {lang:7s} {voice:38s} ({len(r.audio)} bytes)")
                ok.append(lang)
            except Exception as exc:  # noqa: BLE001
                detail = getattr(exc, "details", lambda: str(exc))()
                print(f"  ERROR {lang:7s} {voice:38s} -> {detail}")
                err.append(lang)
        print(f"\n  synth OK ({len(ok)}): {ok}")
        if err:
            print(f"  synth ERROR ({len(err)}): {err}")


if __name__ == "__main__":
    main()
