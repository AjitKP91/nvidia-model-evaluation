#!/usr/bin/env bash
# Download all HuggingFace evaluation datasets once and cache them in ~/hf_home.
# Run this after setup.sh and before start_eval.sh.
# Datasets are preserved across clean.sh runs (unless --data flag is used).
# Usage: bash scripts/download_datasets.sh [DATASET_NAME]
#   DATASET_NAME — optional; one of: librispeech_clean, librispeech_other,
#                  tedlium, gigaspeech, spgispeech, earnings22, ami,
#                  common_voice_en, ljspeech
#                  Omit to download all.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
DATASET="${1:-all}"

echo "=========================================="
echo "  NVIDIA Eval — Download Datasets"
echo "  Dataset: $DATASET"
echo "=========================================="

# ── Check venv exists ────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: .venv not found. Run setup first: bash scripts/setup.sh"
    exit 1
fi

source "$REPO_DIR/.venv/bin/activate"
export HF_HOME=~/hf_home
export TORCH_HOME=~/torch_home
export PIP_CACHE_DIR=~/pip_cache

echo ""
echo "HF_HOME: $HF_HOME"
echo ""
echo "Note: Some datasets (GigaSpeech, SPGISpeech) require you to accept"
echo "      their terms on HuggingFace before downloading."
echo "      Visit https://huggingface.co/datasets/<name> and click Accept."
echo ""

# ── Download ─────────────────────────────────────────────────────────────────
python -m eval.data.download_datasets --dataset "$DATASET"

echo ""
echo "=========================================="
echo "  Download complete."
echo "  Datasets cached in: $HF_HOME/datasets"
echo "  Run 'bash scripts/start_eval.sh' to start evaluation."
echo "=========================================="
