from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml


@dataclass
class RivaConfig:
    use_ssl: bool = True
    # Static token (optional). If set, it is used verbatim.
    auth_token_env: str = "AICORE_BEARER_TOKEN"
    # SAP AI Core gRPC gateway routing
    resource_group: str = "default"
    # Client-credentials auto-refresh (used when the static token env is unset).
    # These name the ENV VARS that hold the actual secrets — never put secrets
    # in this file or in config.yaml.
    client_id_env: str = "AICORE_CLIENT_ID"
    client_secret_env: str = "AICORE_CLIENT_SECRET"
    auth_url_env: str = "AICORE_AUTH_URL"
    # Seconds before expiry at which we proactively refresh a minted token.
    token_refresh_margin_s: int = 300

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._cached_token: str | None = None
        self._cached_exp: float = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _jwt_exp(token: str) -> float:
        """Best-effort parse of a JWT `exp` claim (no signature check)."""
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return float(payload.get("exp", 0.0))
        except Exception:
            return 0.0

    def _mint_token(self) -> str:
        client_id = os.environ.get(self.client_id_env, "")
        client_secret = os.environ.get(self.client_secret_env, "")
        auth_url = os.environ.get(self.auth_url_env, "")
        if not (client_id and client_secret and auth_url):
            raise EnvironmentError(
                f"Set ${self.auth_token_env} with a Bearer token, OR set "
                f"${self.client_id_env}, ${self.client_secret_env} and "
                f"${self.auth_url_env} for client-credentials auto-refresh."
            )
        token_url = auth_url.rstrip("/") + "/oauth/token"
        resp = requests.post(
            token_url,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    @property
    def auth_token(self) -> str:
        """Return a currently-valid Bearer token.

        Priority:
          1. Static token in ``$auth_token_env`` (used verbatim, no refresh).
          2. Client-credentials flow, cached and refreshed ~5 min before expiry.
        """
        static = os.environ.get(self.auth_token_env, "")
        if static:
            return static

        with self._lock:
            now = time.time()
            if (
                self._cached_token is not None
                and now < self._cached_exp - self.token_refresh_margin_s
            ):
                return self._cached_token
            token = self._mint_token()
            exp = self._jwt_exp(token)
            # Fall back to a 1-hour lifetime if exp couldn't be parsed.
            self._cached_exp = exp if exp > now else now + 3600
            self._cached_token = token
            return token


@dataclass
class STTConfig:
    grpc_uri: str
    model_name: str
    rest_endpoint: str
    deployment_id: str = ""          # SAP AI Core gRPC routing (empty = direct NIM host)
    language_code: str = "en-US"
    auth_header: str = "Authorization"
    request_timeout_s: int = 120
    # Offline/batch Recognize is UNIMPLEMENTED on the AI Core gateway — only
    # StreamingRecognize exists. When False, recognize_batch() transparently
    # drives the streaming RPC instead.
    supports_batch: bool = True
    # REST inference proxy returns 500 on the gateway; disable REST tests there.
    supports_rest: bool = True
    # Optional allow-list of language keys (e.g. ["en","de","ja"]) for the
    # multilingual test (1.12). Empty = run every language in its table.
    multilingual_languages: list[str] = field(default_factory=list)
    # Optional languages to ADD (beyond the unchanged English pass) to the
    # Tier-A per-language depth tests — 1.5 noise robustness and 1.9 output
    # quality. Keys from eval/stt/languages.py (e.g. ["de","ja","es"]).
    # Empty = English-only = original behaviour (nothing extra runs).
    multilingual_extra_languages: list[str] = field(default_factory=list)
    # Optional representative languages to spot-check in the Tier-B tests —
    # 1.2 performance, 1.7 long-form, 1.10 confidence (e.g. ["en","de","ja"]).
    # These are added alongside the unchanged English pass. Empty = English-only.
    spot_check_languages: list[str] = field(default_factory=list)


@dataclass
class TTSConfig:
    grpc_uri: str
    model_name: str
    voice_name: str
    rest_endpoint: str
    deployment_id: str = ""          # SAP AI Core gRPC routing (empty = direct NIM host)
    language_code: str = "en-US"
    auth_header: str = "Authorization"
    request_timeout_s: int = 60
    sample_rate: int = 22050
    max_sequence_tokens: int = 400
    supports_rest: bool = True
    # Optional language keys (from en|fr|es — the languages Magpie TTS
    # Multilingual supports) to run the per-language TTS passes: 2.1
    # naturalness, 2.2 intelligibility, 2.3 prosody, 2.9 acceptance.
    # Empty = English-only = original behaviour (nothing extra runs).
    multilingual_languages: list[str] = field(default_factory=list)
    # Optional per-deployment voice-name overrides, mapping a language key to
    # the exact voice string this endpoint serves (when the probe finds the
    # deployment uses different voice strings than the documented defaults).
    voice_overrides: dict = field(default_factory=dict)


@dataclass
class EvalConfig:
    stt_concurrency_levels: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    tts_concurrency_levels: list[int] = field(
        default_factory=lambda: [1, 5, 10, 20, 50]
    )
    bootstrap_n: int = 1000
    results_dir: str = "results/"
    log_level: str = "INFO"
    data_dir: str = "eval/data/"


@dataclass
class Config:
    riva: RivaConfig
    stt: STTConfig
    tts: TTSConfig
    evaluation: EvalConfig

    @property
    def results_path(self) -> Path:
        return Path(self.evaluation.results_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.evaluation.data_dir)


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a project-root ``.env`` into os.environ.

    Zero-dependency (no python-dotenv). Looks for ``.env`` next to the repo root
    (parent of this package) and in the current working directory. Values already
    present in the real environment WIN — exporting a var in your shell overrides
    the file, so CI/secret-manager injection keeps working. Supports an optional
    ``export `` prefix, ``#`` comments, blank lines, and single/double quotes.

    This is where SAP AI Core credentials live for a full harness run:
    AICORE_CLIENT_ID / AICORE_CLIENT_SECRET / AICORE_AUTH_URL (or a static
    AICORE_BEARER_TOKEN) plus optional HF_TOKEN. The file is git-ignored
    (.gitignore: ``.env`` / ``*.env``) so secrets never get committed — never put
    the secret VALUES in config.yaml or config.py, only in .env or the shell.
    """
    candidates = [
        Path(__file__).resolve().parent.parent / ".env",  # repo root
        Path.cwd() / ".env",
    ]
    seen: set[Path] = set()
    for env_path in candidates:
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        try:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except OSError:
            pass


def load_config(path: str | Path | None = None) -> Config:
    _load_dotenv()
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        riva=RivaConfig(**raw["riva"]),
        stt=STTConfig(**raw["stt"]),
        tts=TTSConfig(**raw["tts"]),
        evaluation=EvalConfig(**raw.get("evaluation", {})),
    )
