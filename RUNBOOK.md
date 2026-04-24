# Evaluation Runbook — NVIDIA Riva on SAP AI Core

## TL;DR

**Run on the Azure VM, not your laptop.**
The VM is in Germany West Central, which is the same region as AI Core (Frankfurt / eu-central-1).
Running locally adds 10–40 ms of network jitter to every single API call, which contaminates all latency and RTF measurements.

---

## 1. Where to Run What

| Task | Where | Why |
|------|-------|-----|
| All 20 eval tests (phase0, stt, tts) | **Azure VM** | Same region as AI Core → clean latency numbers |
| Whisper transcription (local GPU) | **Azure VM** | NC24ads A100 handles large-v3 in real time |
| UTMOS / SpeechBrain scoring | **Azure VM** | GPU-accelerated; takes hours on CPU |
| Viewing the HTML report | Laptop | Copy `results/report.html` back via SCP |
| Editing config / fixing code | Either | Up to you |

---

## 2. One-Time VM Setup

SSH into your Azure VM, then run the following once.

### 2.1 System packages

```bash
sudo apt-get update && sudo apt-get install -y \
    python3 python3-pip python3-venv git ffmpeg libsndfile1
```

### 2.2 Clone the repo

```bash
git clone <your-repo-url> nvidia-model-evaluation
cd nvidia-model-evaluation
```

Or if you've been editing locally, rsync your working directory:

```bash
# Run from your laptop:
rsync -avz --exclude results/ --exclude __pycache__ \
    /Users/I350548/Downloads/nvidia/nvidia-model-evaluation/ \
    <vm-user>@<vm-ip>:~/nvidia-model-evaluation/
```

### 2.3 Redirect caches to the data disk

The OS disk fills quickly. Before installing anything, point all caches to `/mnt` (the larger data disk):

```bash
sudo mkdir -p /mnt/hf_home /mnt/torch_home /mnt/pip_cache
sudo chown -R $USER /mnt/hf_home /mnt/torch_home /mnt/pip_cache

export HF_HOME=/mnt/hf_home
export TORCH_HOME=/mnt/torch_home
export PIP_CACHE_DIR=/mnt/pip_cache
```

Add these exports to `~/.bashrc` so they survive reboots and new tmux sessions:

```bash
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/mnt/hf_home
export TORCH_HOME=/mnt/torch_home
export PIP_CACHE_DIR=/mnt/pip_cache
EOF
source ~/.bashrc
```

### 2.4 Create a Python virtual environment

```bash
cd ~/nvidia-model-evaluation
python3 -m venv .venv
source .venv/bin/activate
```

### 2.5 Install dependencies

`fairseq` (pulled in by `utmos`) ships an old `omegaconf 2.0.6` with an invalid requirement format that pip 24+ rejects. Unpin it first, then install everything:

```bash
pip install --upgrade pip wheel setuptools

# Fix omegaconf before the full install
pip install "omegaconf>=2.1" --no-cache-dir

# Install all requirements
pip install -r requirements.txt --no-cache-dir

# spaCy English model (needed for Test 1.8 domain vocabulary)
python -m spacy download en_core_web_sm
```

> **Note on UTMOS / NISQA:** These packages may require extra steps if the PyPI version is outdated.
> If `pip install utmos` fails, install from source:
> ```bash
> pip install git+https://github.com/sarulab-speech/UTMOS22.git
> ```

> **Note on pyworld:** F0 extraction in TTS Test 2.3 uses pyworld. If import fails with
> `No module named 'pkg_resources'`, install setuptools into the venv:
> ```bash
> .venv/bin/pip install setuptools --no-cache-dir
> ```
> The test will run without F0 metrics if pyworld is unavailable.

---

## 3. Configure Endpoints

Edit `eval/config.yaml` with your actual AI Core values. Note that each service has its own `grpc_uri`:

```yaml
riva:
  use_ssl: true
  auth_token_env: AICORE_BEARER_TOKEN

stt:
  grpc_uri: <parakeet-host>:443        # gRPC host:port (no https://)
  model_name: parakeet-1.1b-en-US-asr-offline
  rest_endpoint: https://<parakeet-host>/v1/audio/transcriptions
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 120

tts:
  grpc_uri: <magpie-host>:443          # gRPC host:port (no https://)
  model_name: Magpie-Multilingual.EN-US.Aria
  voice_name: Magpie-Multilingual.EN-US.Aria
  rest_endpoint: https://<magpie-host>/v1/audio/synthesize_online
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 60
  sample_rate: 22050
```

Then export your bearer token (get it from the AI Core service key / OAuth token endpoint):

```bash
export AICORE_BEARER_TOKEN="eyJ..."
```

To make this permanent across sessions, add it to `~/.bashrc`:

```bash
echo 'export AICORE_BEARER_TOKEN="eyJ..."' >> ~/.bashrc
```

### Quick connectivity check

Before running any tests, verify the endpoints are reachable:

```bash
python -m eval.run phase0
```

This runs all Phase 0 discovery checks (gRPC ping, REST ping, schema discovery, smoke tests, cold-start measurement) and writes results to `results/phase0/discovery.json`. Fix any connectivity issues before proceeding.

---

## 4. Running the Evaluation

All commands are run from the repo root with the venv active.

### Run everything end-to-end

```bash
python -m eval.run all
```

This runs Phase 0 → all 11 STT tests → all 8 TTS tests → generates `results/report.html`.
Expected runtime on the A100 VM: **4–8 hours** depending on API throughput.

---

### Run one phase at a time (recommended)

```bash
# Phase 0: connectivity + discovery
python -m eval.run phase0

# All STT tests
python -m eval.run stt

# All TTS tests
python -m eval.run tts

# Generate report from whatever results exist so far
python -m eval.run report
```

---

### Run a single test

```bash
# By name:
python -m eval.run stt --test accuracy
python -m eval.run stt --test streaming
python -m eval.run tts --test naturalness
python -m eval.run tts --test edge_cases

# Available STT test names:
#   accuracy, performance, streaming, rest_vs_grpc, noise_robustness,
#   accent, long_form, domain, output_quality, confidence, format_robustness

# Available TTS test names:
#   naturalness, intelligibility, prosody, signal_quality,
#   latency, concurrency, edge_cases, long_form
```

---

### Dry run (check config without calling the API)

```bash
python -m eval.run all --dry-run
```

Prints resolved endpoints and exits. Useful for verifying config before a long run.

---

## 5. Resuming After Interruption

Every API call result is written to a `.jsonl` file immediately. If a test is interrupted, re-running the same command will **skip already-completed items** automatically — no data is lost and no duplicate API calls are made.

```bash
# Safe to re-run at any time:
python -m eval.run stt --test accuracy
```

---

## 6. Outputs

```
results/
├── phase0/
│   └── discovery.json            # Phase 0 findings
├── stt/
│   ├── accuracy/
│   │   ├── calls.jsonl           # Per-utterance results
│   │   └── summary.csv           # Dataset-level WER/CER/etc.
│   ├── streaming/
│   │   ├── calls.jsonl
│   │   └── summary.csv
│   └── ...                       # One folder per test
├── tts/
│   ├── naturalness/
│   │   ├── calls.jsonl
│   │   ├── summary.csv
│   │   └── synth_0001.wav        # Synthesized audio files
│   └── ...
└── report.html                   # Full evaluation report (open in browser)
```

After the run, copy the report back to your laptop:

```bash
# From your laptop:
scp <vm-user>@<vm-ip>:~/nvidia-model-evaluation/results/report.html ~/Desktop/
```

---

## 7. Keeping the VM Run Alive

SSH sessions that disconnect will kill the process. Use `tmux` or `nohup`:

### Option A — tmux (recommended)

```bash
# On the VM, start a tmux session:
tmux new -s eval

# Inside tmux, activate venv and run:
source .venv/bin/activate
python -m eval.run all 2>&1 | tee results/run.log

# Detach from tmux (the process keeps running):
# Press Ctrl+B, then D

# Re-attach later:
tmux attach -t eval
```

### Option B — nohup

```bash
nohup python -m eval.run all > results/run.log 2>&1 &
echo "PID: $!"

# Monitor progress:
tail -f results/run.log
```

---

## 8. Log Level & Debugging

```bash
# Verbose output:
python -m eval.run stt --test accuracy --log-level DEBUG

# Suppress most output:
python -m eval.run tts --log-level WARNING
```

---

## 9. Test Runtime Estimates (A100 VM, Frankfurt → AI Core)

| Test | Calls | Estimated Time |
|------|-------|----------------|
| Phase 0 | ~20 | 2–5 min |
| 1.1 Accuracy | ~5,000 utterances | 60–90 min |
| 1.2 Performance | 100 + concurrency | 10–15 min |
| 1.3 Streaming | 300 utterances | 20–30 min |
| 1.4 REST vs gRPC | 200 calls | 10 min |
| 1.5 Noise Robustness | ~3,000 calls | 45–60 min |
| 1.6 Accent | ~800 utterances | 20–30 min |
| 1.7 Long Form | ~30 files | 15–20 min |
| 1.8 Domain | ~4,000 utterances | 60 min |
| 1.9 Output Quality | ~2,000 utterances | 30 min |
| 1.10 Confidence | ~1,000 utterances | 20 min |
| 1.11 Format Robustness | ~200 calls | 10 min |
| 2.1 Naturalness | 200 synths + UTMOS | 30–45 min |
| 2.2 Intelligibility | 350 synths + Whisper | 45–60 min |
| 2.3 Prosody | 50 synths | 10 min |
| 2.4 Signal Quality | 50 synths | 10 min |
| 2.5 Latency | 500 calls | 20 min |
| 2.6 Concurrency | 500 calls | 15 min |
| 2.7 Edge Cases | ~100 calls | 15 min |
| 2.8 Long Form TTS | ~30 synths + ECAPA | 20 min |
| **Total** | | **~7 hours** |

---

## 10. Troubleshooting

### `$AICORE_BEARER_TOKEN not set`
```bash
export AICORE_BEARER_TOKEN="eyJ..."
```
Tokens from AI Core expire. If you get 401 errors mid-run, fetch a new token and re-export — the harness retries automatically.

### `grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE`
The gRPC endpoint is unreachable. Check:
1. The `grpc_uri` in `config.yaml` is correct (host:port, no `https://`)
2. Port 443 is open in your VM's Network Security Group (NSG)
3. Run `python -m eval.run phase0` to see detailed connectivity output

### `ModuleNotFoundError: No module named 'riva'`
```bash
source .venv/bin/activate
pip install nvidia-riva-client
```

### `ModuleNotFoundError: No module named 'pkg_resources'` (pyworld)
pyworld depends on setuptools which may be missing from the venv:
```bash
.venv/bin/pip install setuptools --no-cache-dir
```
Always use `.venv/bin/pip` (not plain `pip`) to target the venv.

### `omegaconf` / `invalid-installed-package` error during pip install
pip 24+ rejects `omegaconf 2.0.6` (pulled in by `fairseq`/`utmos`) due to an invalid requirement format. Fix:
```bash
.venv/bin/pip uninstall -y omegaconf
.venv/bin/pip install "omegaconf>=2.1" --no-cache-dir
.venv/bin/pip install -r requirements.txt --no-cache-dir
```

### `[Errno 28] No space left on device`
The `/mnt` data disk is full. Find what's using space:
```bash
du -sh /mnt/* 2>/dev/null | sort -rh | head -10
```
Common culprits:
```bash
# HuggingFace model/dataset cache
rm -rf /mnt/hf_cache

# pip wheel build cache
rm -rf /mnt/pip_cache/*
```
Check that caches are redirected to `/mnt` and not filling the OS disk:
```bash
df -h / /mnt
```

### `[Errno 13] Permission denied: '/mnt/hf_home'`
Fix ownership of the cache directories:
```bash
sudo chown -R $USER /mnt/hf_home /mnt/torch_home /mnt/pip_cache
```

### `Unauthorized` / 403 on HuggingFace datasets (esb/datasets)
Several STT datasets (LibriSpeech, TED-LIUM, GigaSpeech, etc.) are loaded via the gated `esb/datasets` collection. To access them:
1. Request access at https://huggingface.co/datasets/esb/datasets
2. Once approved, log in on the VM:
   ```bash
   huggingface-cli login   # paste your HF token when prompted
   ```
Tests will skip gracefully if datasets are unavailable — this is not a fatal error.

### UTMOS / SpeechBrain downloads fail
These models download weights on first use (~hundreds of MB). Ensure the VM has internet access and enough disk space (50 GB recommended on `/mnt`).

### `ffmpeg not found` (pydub / librosa errors)
```bash
sudo apt-get install -y ffmpeg
```

### Out of disk space in results/
TTS audio files can reach 20–50 GB. The `.jsonl` and `.csv` files are small; only `.wav` files are large:
```bash
rm -rf results/tts/naturalness/*.wav
```
