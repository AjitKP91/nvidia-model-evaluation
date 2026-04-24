#!/usr/bin/env bash
# One-time setup: system packages, /mnt caches, Python venv, dependencies, HuggingFace login.
# Run this once after cloning the repo.
# Usage: bash scripts/setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "=========================================="
echo "  NVIDIA Eval — One-Time Setup"
echo "=========================================="

# ── 1. System packages ──────────────────────────────────────────────────────
echo ""
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y ffmpeg libavcodec-extra libsndfile1 git python3 python3-pip python3-venv

# libopenh264: ffmpeg needs this for H.264; version on the system may differ
sudo apt-get install -y libopenh264-dev 2>/dev/null || true
# If only .so.6 is available, symlink it as .so.5 (what ffmpeg looks for)
if [ ! -f /usr/lib/x86_64-linux-gnu/libopenh264.so.5 ]; then
    SO6=$(find /usr/lib -name "libopenh264.so.6" 2>/dev/null | head -1)
    if [ -n "$SO6" ]; then
        sudo ln -sf "$SO6" /usr/lib/x86_64-linux-gnu/libopenh264.so.5
        echo "  Symlinked libopenh264.so.6 → libopenh264.so.5"
    fi
fi

# ── 2. /mnt cache directories ───────────────────────────────────────────────
echo ""
echo "[2/6] Setting up /mnt cache directories..."
sudo mkdir -p /mnt/hf_home /mnt/torch_home /mnt/pip_cache
sudo chown -R "$USER" /mnt/hf_home /mnt/torch_home /mnt/pip_cache

# Persist env vars in ~/.bashrc (idempotent)
for line in \
    'export HF_HOME=/mnt/hf_home' \
    'export TORCH_HOME=/mnt/torch_home' \
    'export PIP_CACHE_DIR=/mnt/pip_cache'; do
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done
export HF_HOME=/mnt/hf_home
export TORCH_HOME=/mnt/torch_home
export PIP_CACHE_DIR=/mnt/pip_cache
echo "  Cache dirs: HF_HOME=$HF_HOME  TORCH_HOME=$TORCH_HOME"

# ── 3. Python venv ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] Creating Python virtual environment..."
if [ ! -d "$REPO_DIR/.venv" ]; then
    python3 -m venv "$REPO_DIR/.venv"
fi
source "$REPO_DIR/.venv/bin/activate"

# ── 4. Install Python dependencies ──────────────────────────────────────────
echo ""
echo "[4/6] Installing Python dependencies (this may take 10-20 minutes)..."

# Install uv — much faster resolver, handles complex graphs pip cannot
pip install uv --no-cache-dir -q
echo "  uv installed."

# Pre-fix omegaconf: fairseq/utmos ships 2.0.6 which pip 24+ rejects
echo "  Pre-fixing omegaconf..."
uv pip install "omegaconf>=2.1" --no-cache-dir

# Install all requirements via uv (avoids resolution-too-deep errors)
uv pip install -r "$REPO_DIR/requirements.txt" --no-cache-dir

# Ensure setuptools is present for pyworld
uv pip install setuptools --no-cache-dir

# spaCy language model
echo "  Downloading spaCy model..."
python -m spacy download en_core_web_sm -q

echo "  Dependencies installed."

# ── Set LD_LIBRARY_PATH for PyTorch bundled CUDA (fixes UTMOS libcudart error) ──
TORCH_LIB="$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null)/lib"
if [ -d "$TORCH_LIB" ]; then
    LINE="export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
    grep -qxF "$LINE" ~/.bashrc || echo "$LINE" >> ~/.bashrc
    export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
    echo "  CUDA runtime path set: $TORCH_LIB"
fi

# ── 5. HuggingFace login ─────────────────────────────────────────────────────
echo ""
echo "[5/6] HuggingFace login"
echo "  Get your token at: https://huggingface.co/settings/tokens"
echo "  Make sure you have accepted access to esb/datasets at:"
echo "  https://huggingface.co/datasets/esb/datasets"
echo ""
read -rsp "  Paste your HuggingFace token (input hidden): " HF_TOKEN
echo ""
echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
echo "  HuggingFace login done."

# ── 6. Verify ────────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying setup..."
python -c "import riva.client; print('  riva.client OK')"
python -c "import jiwer; print('  jiwer OK')"
python -c "import soundfile; print('  soundfile OK')"
python -c "import pkg_resources; print('  pkg_resources OK')"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  Next step: run the evaluation with:"
echo "    bash scripts/start_eval.sh"
echo "=========================================="
