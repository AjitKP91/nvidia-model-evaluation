"""Download all evaluation datasets from HuggingFace and other sources."""
from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger("eval.data")

DATASETS = {
    # ---- STT datasets (HuggingFace) ----
    "librispeech_clean": {
        "hf_path": "librispeech_asr",
        "hf_name": "clean",
        "split": "test.clean",
        "description": "LibriSpeech test-clean (5.4 h narrated audiobooks)",
    },
    "librispeech_other": {
        "hf_path": "librispeech_asr",
        "hf_name": "other",
        "split": "test.other",
        "description": "LibriSpeech test-other (5.4 h challenging narration)",
    },
    "tedlium": {
        "hf_path": "LIUM/tedlium",
        "hf_name": "release3",
        "split": "test",
        "description": "TED-LIUM v3 test (3 h prepared oratory)",
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
        "split": "val",
        "description": "SPGISpeech val (financial calls)",
    },
    "earnings22": {
        "hf_path": "revdotcom/earnings22",
        "hf_name": None,
        "split": "test",
        "description": "Earnings-22 test (spontaneous business)",
    },
    "ami": {
        "hf_path": "edinburghcst/ami",
        "hf_name": "ihm",
        "split": "test",
        "description": "AMI IHM test (meeting conversations)",
    },
    "common_voice_en": {
        "hf_path": "mozilla-foundation/common_voice_17_0",
        "hf_name": "en",
        "split": "test",
        "description": "Common Voice EN test (accented speech)",
    },
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
    # Only Common Voice uses a custom loading script that requires trust_remote_code
    if info["hf_path"].startswith("mozilla-foundation/"):
        kwargs["trust_remote_code"] = True
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
