#!/usr/bin/env bash
# Clean everything for a fresh restart.
# Usage:
#   bash scripts/clean.sh          — clears results only (keeps .venv and datasets)
#   bash scripts/clean.sh --data   — also clears HuggingFace dataset cache (forces re-download)
#   bash scripts/clean.sh --full   — also removes .venv (forces full reinstall)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
FULL=false
DATA=false
for arg in "${@}"; do
    [ "$arg" = "--full" ] && FULL=true
    [ "$arg" = "--data" ] && DATA=true
done

echo "=========================================="
echo "  NVIDIA Eval — Clean"
echo "  Full mode : $FULL"
echo "  Wipe data : $DATA"
echo "=========================================="

# ── Kill tmux session ────────────────────────────────────────────────────────
if tmux has-session -t eval 2>/dev/null; then
    echo "Killing tmux session 'eval'..."
    tmux kill-session -t eval
fi

# ── Remove results ───────────────────────────────────────────────────────────
echo "Removing results/..."
rm -rf "$REPO_DIR/results"
mkdir -p "$REPO_DIR/results"

# ── Remove tmux launcher ─────────────────────────────────────────────────────
rm -f "$REPO_DIR/scripts/_tmux_launcher.sh"

# ── Optionally clear HuggingFace dataset cache ───────────────────────────────
if [ "$DATA" = true ]; then
    if [ -d "$HOME/hf_home/datasets" ]; then
        echo "Clearing HuggingFace dataset cache (~/$HOME/hf_home/datasets)..."
        rm -rf "$HOME/hf_home/datasets"
        echo "  Run 'bash scripts/download_datasets.sh' to re-download."
    fi
else
    echo "Keeping HuggingFace dataset cache (pass --data to wipe)."
fi

# ── Full clean: remove venv ──────────────────────────────────────────────────
if [ "$FULL" = true ]; then
    echo "Removing .venv (full clean)..."
    rm -rf "$REPO_DIR/.venv"
    echo "Run 'bash scripts/setup.sh' to reinstall dependencies."
fi

# ── Check disk space on / ────────────────────────────────────────────────────
echo ""
echo "Disk usage after clean:"
df -h /
echo ""
echo "Largest items in home:"
du -sh ~/* 2>/dev/null | sort -rh | head -5 || true

echo ""
echo "=========================================="
echo "  Clean done."
if [ "$FULL" = false ]; then
    echo "  .venv kept — run 'bash scripts/start_eval.sh' to restart."
else
    echo "  Run 'bash scripts/setup.sh' then 'bash scripts/start_eval.sh'."
fi
echo "=========================================="
