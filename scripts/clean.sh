#!/usr/bin/env bash
# Clean script for NVIDIA Eval harness.
#
# All dataset loading uses streaming=True, so HuggingFace dataset cache never
# accumulates between runs and is always safe to delete.
#
# Usage:
#   bash scripts/clean.sh          — remove results, model weights, dataset
#                                    cache, and temp files; keeps .venv intact
#   bash scripts/clean.sh --full   — also remove .venv (run setup.sh after)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

FULL=false
for arg in "${@}"; do
    [ "$arg" = "--full" ] && FULL=true
done

echo "=========================================="
echo "  NVIDIA Eval — Clean"
echo "  Wipe .venv (--full) : $FULL"
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

# ── Model weights ────────────────────────────────────────────────────────────
echo "Clearing cached model weights (~/hf_home/hub, ~/torch_home)..."
rm -rf "$HOME/hf_home/hub"
rm -rf "$HOME/torch_home"

# ── HuggingFace dataset cache ─────────────────────────────────────────────────
# All loads use streaming=True so nothing here is needed between runs.
echo "Clearing HuggingFace dataset cache (~/hf_home/datasets)..."
rm -rf "$HOME/hf_home/datasets"

# ── Pip wheel cache ──────────────────────────────────────────────────────────
echo "Clearing pip wheel cache..."
rm -rf "$HOME/pip_cache"/* 2>/dev/null || true

# ── Full: also remove .venv ───────────────────────────────────────────────────
if [ "$FULL" = true ]; then
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
    echo "  .venv kept — run 'bash scripts/start_eval.sh' to restart."
    echo ""
    echo "  To also remove .venv (forces full reinstall):"
    echo "    bash scripts/clean.sh --full"
else
    echo "  Run 'bash scripts/setup.sh' then 'bash scripts/start_eval.sh'."
fi
echo "=========================================="
