#!/usr/bin/env bash
# Clean script for NVIDIA Eval harness.
#
# Usage:
#   bash scripts/clean.sh          — remove results, model weights, temp files
#                                    (datasets are preserved — no re-download needed)
#   bash scripts/clean.sh --full   — also remove datasets and .venv (full reset)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

FULL=false
for arg in "${@}"; do
    [ "$arg" = "--full" ] && FULL=true
done

echo "=========================================="
echo "  NVIDIA Eval — Clean"
echo "  Full reset (incl. datasets) : $FULL"
echo "=========================================="

echo ""
echo "Disk usage BEFORE clean:"
df -h /
echo ""
echo "Top space users:"
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

# ── Stray temp WAV files from crashed runs ───────────────────────────────────
echo "Removing stray /tmp/*.wav files..."
rm -f /tmp/*.wav 2>/dev/null || true

# ── Results ──────────────────────────────────────────────────────────────────
echo "Removing results/..."
rm -rf "$REPO_DIR/results"
mkdir -p "$REPO_DIR/results"
rm -f "$REPO_DIR/scripts/_tmux_launcher.sh"

# ── Model weights (small models are deleted in-program after use;            ──
#    this clears anything left over from interrupted runs)                    ──
echo "Clearing cached model weights (~/hf_home/hub, ~/torch_home)..."
rm -rf "$HOME/hf_home/hub"
rm -rf "$HOME/torch_home"

# ── Pip wheel cache ──────────────────────────────────────────────────────────
echo "Clearing pip wheel cache..."
rm -rf "$HOME/pip_cache"/* 2>/dev/null || true

# ── Full: also remove datasets and .venv ─────────────────────────────────────
if [ "$FULL" = true ]; then
    echo "Full reset: removing HuggingFace dataset cache..."
    rm -rf "$HOME/hf_home/datasets"
    echo "Full reset: removing .venv..."
    rm -rf "$REPO_DIR/.venv"
    echo "  Run 'bash scripts/setup.sh' to reinstall dependencies."
fi

echo ""
echo "Disk usage AFTER clean:"
df -h /
echo ""
echo "Top space users:"
du -sh "$HOME"/hf_home "$HOME"/torch_home "$HOME"/pip_cache \
        "$REPO_DIR/.venv" "$REPO_DIR/results" 2>/dev/null \
    | sort -rh | head -10 || true

echo ""
echo "=========================================="
echo "  Clean done."
echo ""
if [ "$FULL" = false ]; then
    echo "  Datasets kept — run 'bash scripts/start_eval.sh' to restart."
    echo ""
    echo "  To wipe EVERYTHING (including datasets + venv):"
    echo "    bash scripts/clean.sh --full"
else
    echo "  Run 'bash scripts/setup.sh' then 'bash scripts/start_eval.sh'."
fi
echo "=========================================="
