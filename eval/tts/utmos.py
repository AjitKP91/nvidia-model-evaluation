"""Shared UTMOS22 scorer — lazy-initialised, one model per process."""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import shutil
import zipfile
from pathlib import Path

import soundfile as sf

logger = logging.getLogger("eval.tts.utmos")

_utmos_scorer = None  # ("pypi"|"hub", model)
_utmos_available = None


def _preload_cuda_runtime() -> None:
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if torch_lib not in current:
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{current}"
        for name in sorted(os.listdir(torch_lib)):
            if name.startswith("libcudart"):
                try:
                    ctypes.CDLL(os.path.join(torch_lib, name))
                except Exception:
                    pass
    except Exception:
        pass


_preload_cuda_runtime()


def _install_utmos22(hub_dir: Path) -> Path | None:
    """Download UTMOS22 via GitHub archive and install into hub_dir.

    Uses /archive/refs/heads/master.zip which extracts to UTMOS22-master/,
    a predictable name unlike the zipball which uses a commit hash.
    Returns the installed directory path, or None on failure.
    """
    import urllib.request

    zip_url = "https://github.com/sarulab-speech/UTMOS22/archive/refs/heads/master.zip"
    zip_path = hub_dir / "utmos22_master.zip"
    target_dir = hub_dir / "sarulab-speech_UTMOS22_master"

    try:
        logger.info("Downloading UTMOS22 from GitHub archive")
        urllib.request.urlretrieve(zip_url, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(hub_dir)
        zip_path.unlink(missing_ok=True)

        # GitHub /archive/ always extracts to {repo}-{branch}/ = UTMOS22-master/
        extracted = hub_dir / "UTMOS22-master"
        if extracted.is_dir():
            shutil.rmtree(target_dir, ignore_errors=True)
            extracted.rename(target_dir)
            logger.info("Installed UTMOS22 to %s", target_dir)
            return target_dir

        logger.warning("Expected %s after extraction but it was not found", extracted)
        return None
    except Exception as e:
        logger.warning("UTMOS22 installation failed: %s", e)
        zip_path.unlink(missing_ok=True)
        return None


def get_utmos_scorer():
    global _utmos_scorer, _utmos_available
    if _utmos_available is False:
        return None
    if _utmos_scorer is not None:
        return _utmos_scorer

    # Strategy 1: PyPI utmos package
    try:
        from utmos import UTMOSScore
        _utmos_scorer = ("pypi", UTMOSScore())
        _utmos_available = True
        logger.info("UTMOS scorer initialised (utmos PyPI package)")
        return _utmos_scorer
    except Exception as e:
        logger.warning("UTMOS PyPI strategy failed: %s", e)

    # Strategy 2: UTMOS22 via direct download + torch.hub source=local
    try:
        import torch

        hub_dir = Path(torch.hub.get_dir())
        hub_dir.mkdir(parents=True, exist_ok=True)
        utmos_dir = hub_dir / "sarulab-speech_UTMOS22_master"

        if not (utmos_dir / "hubconf.py").exists():
            # Also check for any previously extracted variant with wrong name
            for d in hub_dir.iterdir():
                if d.is_dir() and d != utmos_dir and (
                    "utmos22" in d.name.lower() or "sarulab" in d.name.lower()
                ) and (d / "hubconf.py").exists():
                    logger.info("Renaming misnamed hub dir %s → %s", d.name, utmos_dir.name)
                    shutil.rmtree(utmos_dir, ignore_errors=True)
                    d.rename(utmos_dir)
                    break

        if not (utmos_dir / "hubconf.py").exists():
            result = _install_utmos22(hub_dir)
            if result is None or not (result / "hubconf.py").exists():
                raise RuntimeError("hubconf.py not found after install")
            utmos_dir = result

        predictor = torch.hub.load(
            str(utmos_dir), "strong",
            source="local", trust_repo=True, verbose=False,
        )
        _utmos_scorer = ("hub", predictor)
        _utmos_available = True
        logger.info("UTMOS scorer initialised (torch.hub UTMOS22, source=local)")
        return _utmos_scorer
    except Exception as e:
        logger.warning("UTMOS not available: %s", e)
        _utmos_available = False
        return None


def score_utmos(audio_path: str) -> float | None:
    result = get_utmos_scorer()
    if result is None:
        return None
    kind, model = result
    try:
        if kind == "pypi":
            return float(model.score(audio_path))
        # torch.hub UTMOS22: expects 1-D float32 tensor at 16 kHz
        import torch
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav_t = torch.tensor(wav).unsqueeze(0)
        return float(model(wav_t, 16000).item())
    except Exception as e:
        logger.warning("UTMOS scoring failed: %s", e)
        return None


def release_utmos() -> None:
    """Free the loaded model and optionally wipe the hub cache dir."""
    global _utmos_scorer, _utmos_available
    if _utmos_scorer is None:
        return
    kind, model = _utmos_scorer
    del model
    _utmos_scorer = None
    _utmos_available = None
    gc.collect()
    if kind == "hub":
        try:
            import torch
            torch.cuda.empty_cache()
            hub_dir = Path(torch.hub.get_dir())
            for entry in hub_dir.iterdir():
                if "utmos" in entry.name.lower() or "sarulab" in entry.name.lower():
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info("Deleted UTMOS hub cache: %s", entry)
        except Exception as e:
            logger.warning("UTMOS cache cleanup failed: %s", e)
