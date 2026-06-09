"""Download all evaluation datasets from HuggingFace and other sources."""
from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger("eval.data")

DATASETS = {
    # ---- STT datasets (HuggingFace) ----
    "librispeech_clean": {
        "hf_path": "openslr/librispeech_asr",
        "hf_name": "clean",
        "split": "test",
        "description": "LibriSpeech test-clean (5.4 h narrated audiobooks)",
    },
    "librispeech_other": {
        "hf_path": "openslr/librispeech_asr",
        "hf_name": "other",
        "split": "test",
        "description": "LibriSpeech test-other (5.4 h challenging narration)",
    },
    "tedlium": {
        "hf_path": "facebook/voxpopuli",
        "hf_name": "en",
        "split": "test",
        "description": "VoxPopuli EN test (European Parliament prepared speech — replaces LIUM/tedlium)",
    },
    "gigaspeech": {
        "hf_path": "speechcolab/gigaspeech",
        "hf_name": "xs",
        "split": "test",
        "description": "GigaSpeech test xs (mixed broadcast/web)",
    },
    "spgispeech": {
        "hf_path": "kensho/spgispeech",
        "hf_name": None,
        "split": "validation",
        "description": "SPGISpeech validation (financial calls)",
    },
    # revdotcom/earnings22 removed — builder downloads from dead third-party URLs
    # edinburghcst/ami removed — dataset not on HF Hub
    # common_voice_en removed — Mozilla migrated all CV datasets to Mozilla Data
    #   Collective in October 2025; the HF repo (common_voice_17_0) is now empty
    # ---- TTS reference datasets ----
    "ljspeech": {
        "hf_path": "keithito/lj_speech",
        "hf_name": None,
        "split": "train",
        "description": "LJSpeech (single female speaker, TTS reference)",
    },
}


def download_dataset(name: str, cache_dir: str | Path | None = None) -> object:
    info = DATASETS[name]
    logger.info("Downloading %s: %s", name, info["description"])

    kwargs = {
        "path": info["hf_path"],
        "split": info["split"],
        "token": True,
    }
    if info.get("hf_name"):
        kwargs["name"] = info["hf_name"]
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**kwargs)
    logger.info("Downloaded %s: %d examples", name, len(ds))
    return ds


def download_all(cache_dir: str | Path | None = None) -> dict:
    results = {}
    for name in DATASETS:
        try:
            results[name] = download_dataset(name, cache_dir)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            results[name] = None
    return results


def download_musan(target_dir: str | Path) -> None:
    """Download MUSAN noise corpus for noise robustness tests."""
    target_dir = Path(target_dir)
    if (target_dir / "musan").exists():
        logger.info("MUSAN already downloaded at %s", target_dir / "musan")
        return

    logger.info("Downloading MUSAN noise corpus...")
    import subprocess

    subprocess.run(
        [
            "wget",
            "-qO-",
            "https://www.openslr.org/resources/17/musan.tar.gz",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    logger.info(
        "MUSAN download: use `wget https://www.openslr.org/resources/17/musan.tar.gz` "
        "and extract to %s/musan/",
        target_dir,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download evaluation datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.cache_dir)
    else:
        download_dataset(args.dataset, args.cache_dir)
