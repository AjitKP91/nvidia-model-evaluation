#!/usr/bin/env bash
# Start the evaluation in a persistent tmux session.
# Asks for the AICORE bearer token, then launches inside tmux so SSH drops don't kill it.
# Usage: bash scripts/start_eval.sh [all|stt|tts|phase0|report]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
COMMAND="${1:-all}"
SESSION="eval"

echo "=========================================="
echo "  NVIDIA Eval — Start Run ($COMMAND)"
echo "=========================================="

# ── Check venv exists ────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "ERROR: .venv not found. Run setup first: bash scripts/setup.sh"
    exit 1
fi

# ── Pull latest code ─────────────────────────────────────────────────────────
echo ""
echo "Pulling latest code..."
git pull

# ── Ask for AICORE token ─────────────────────────────────────────────────────
echo ""
read -rsp "Paste your AICORE_BEARER_TOKEN (input hidden): " AICORE_TOKEN
echo ""
if [ -z "$AICORE_TOKEN" ]; then
    echo "ERROR: token cannot be empty."
    exit 1
fi

# ── Kill existing tmux session if running ────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Killing existing tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION"
fi

# ── Write a launcher script for inside tmux ──────────────────────────────────
LAUNCHER="$REPO_DIR/scripts/_tmux_launcher.sh"
cat > "$LAUNCHER" <<LAUNCHER_EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_DIR"
source "$REPO_DIR/.venv/bin/activate"
export HF_HOME=/mnt/hf_home
export TORCH_HOME=/mnt/torch_home
export PIP_CACHE_DIR=/mnt/pip_cache
export AICORE_BEARER_TOKEN="$AICORE_TOKEN"
# Make PyTorch bundled CUDA visible (fixes UTMOS libcudart error)
TORCH_LIB="\$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
export LD_LIBRARY_PATH="\$TORCH_LIB:\${LD_LIBRARY_PATH:-}"
mkdir -p results
echo "Starting: python -m eval.run $COMMAND"
echo "Log: results/run.log"
python -m eval.run $COMMAND 2>&1 | tee results/run.log
echo ""
echo "=== Run complete. Press any key to close. ==="
read -n1
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# ── Start tmux session ───────────────────────────────────────────────────────
echo ""
echo "Starting tmux session '$SESSION'..."
tmux new-session -d -s "$SESSION" "bash $LAUNCHER"

echo ""
echo "=========================================="
echo "  Eval is running in tmux session '$SESSION'."
echo ""
echo "  Attach to watch progress:"
echo "    tmux attach -t $SESSION"
echo ""
echo "  Detach without killing:"
echo "    Ctrl+B then D"
echo ""
echo "  Tail the log from outside tmux:"
echo "    tail -f $REPO_DIR/results/run.log"
echo "=========================================="
