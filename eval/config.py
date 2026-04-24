from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RivaConfig:
    grpc_uri: str
    use_ssl: bool = True
    auth_token_env: str = "AICORE_BEARER_TOKEN"

    @property
    def auth_token(self) -> str:
        token = os.environ.get(self.auth_token_env, "")
        if not token:
            raise EnvironmentError(
                f"Set ${self.auth_token_env} with a valid Bearer token"
            )
        return token


@dataclass
class STTConfig:
    model_name: str
    rest_endpoint: str
    language_code: str = "en-US"
    auth_header: str = "Authorization"
    request_timeout_s: int = 120


@dataclass
class TTSConfig:
    model_name: str
    voice_name: str
    rest_endpoint: str
    language_code: str = "en-US"
    auth_header: str = "Authorization"
    request_timeout_s: int = 60
    sample_rate: int = 22050


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


def load_config(path: str | Path | None = None) -> Config:
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
