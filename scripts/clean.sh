#!/usr/bin/env bash
# Clean script for NVIDIA Eval harness.
#
# Usage:
#   bash scripts/clean.sh              — results/ + stray /tmp/*.wav only
#   bash scripts/clean.sh --models    — also clears ML model weights (~/hf_home/hub, ~/torch_home)
#   bash scripts/clean.sh --data      — also clears HuggingFace dataset cache (forces re-download)
#   bash scripts/clean.sh --full      — also removes .venv (forces full reinstall)
#   Flags can be combined: --models --data --full
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

FULL=false
DATA=false
MODELS=false
for arg in "${@}"; do
    [ "$arg" = "--full" ]   && FULL=true
    [ "$arg" = "--data" ]   && DATA=true
    [ "$arg" = "--models" ] && MODELS=true
done

echo "=========================================="
echo "  NVIDIA Eval — Clean"
echo "  Wipe models : $MODELS"
echo "  Wipe data   : $DATA"
echo "  Full reset  : $FULL"
echo "=========================================="

# ── Show disk usage before cleaning ─────────────────────────────────────────
echo ""
echo "Disk usage BEFORE clean:"
df -h /
echo ""
echo "Top space users in home:"
du -sh "$HOME"/hf_home "$HOME"/torch_home "$HOME"/pip_cache \
        "$REPO_DIR/.venv" "$REPO_DIR/results" 2>/dev/null \
    | sort -rh | head -10 || true
echo ""

# ── Kill tmux eval sessions ──────────────────────────────────────────────────
for session in eval eval_stt eval_tts; do
    if tmux has-session -t "$session" 2>/dev/null; then
        echo "Killing tmux session '$session'..."
        tmux kill-session -t "$session"
    fi
done

# ── Always: stray temp WAV files from crashed runs ───────────────────────────
echo "Removing stray /tmp/*.wav files..."
rm -f /tmp/*.wav 2>/dev/null || true

# ── Remove results ───────────────────────────────────────────────────────────
echo "Removing results/..."
rm -rf "$REPO_DIR/results"
mkdir -p "$REPO_DIR/results"

rm -f "$REPO_DIR/scripts/_tmux_launcher.sh"

# ── Optionally clear ML model weights (Whisper, SpeechBrain, UTMOS, etc.) ───
if [ "$MODELS" = true ]; then
    echo "Clearing ML model weights (~/hf_home/hub and ~/torch_home)..."
    rm -rf "$HOME/hf_home/hub"
    rm -rf "$HOME/torch_home"
    echo "  Models will be re-downloaded on next eval run."
fi

# ── Optionally clear HuggingFace dataset cache ───────────────────────────────
if [ "$DATA" = true ]; then
    echo "Clearing HuggingFace dataset cache ($HOME/hf_home/datasets)..."
    rm -rf "$HOME/hf_home/datasets"
    echo "  Run 'bash scripts/download_datasets.sh' to re-download."
fi

# ── Full clean: remove venv ──────────────────────────────────────────────────
if [ "$FULL" = true ]; then
    echo "Removing .venv (full clean)..."
    rm -rf "$REPO_DIR/.venv"
    echo "Run 'bash scripts/setup.sh' to reinstall dependencies."
fi

# ── Pip cache (always safe to clear) ────────────────────────────────────────
echo "Clearing pip wheel cache..."
rm -rf "$HOME/pip_cache"/* 2>/dev/null || true

# ── Show disk usage after cleaning ──────────────────────────────────────────
echo ""
echo "Disk usage AFTER clean:"
df -h /
echo ""
echo "Top space users in home:"
du -sh "$HOME"/hf_home "$HOME"/torch_home "$HOME"/pip_cache \
        "$REPO_DIR/.venv" "$REPO_DIR/results" 2>/dev/null \
    | sort -rh | head -10 || true

echo ""
echo "=========================================="
echo "  Clean done."
echo ""
echo "  To free MORE space without re-downloading datasets:"
echo "    bash scripts/clean.sh --models    (~3-10 GB — model weights)"
echo ""
echo "  To free MAXIMUM space:"
echo "    bash scripts/clean.sh --data      (~50-100 GB — all datasets)"
echo ""
if [ "$FULL" = false ]; then
    echo "  .venv kept — run 'bash scripts/start_eval.sh' to restart."
else
    echo "  Run 'bash scripts/setup.sh' then 'bash scripts/start_eval.sh'."
fi
echo "=========================================="
