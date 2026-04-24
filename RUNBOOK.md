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

### 2.3 Create a Python virtual environment

```bash
cd ~/nvidia-model-evaluation
python3 -m venv .venv
source .venv/bin/activate
```

### 2.4 Install dependencies

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt

# spaCy English model (needed for Test 1.8 domain vocabulary)
python -m spacy download en_core_web_sm

# Whisper (downloads model weights on first use — ~3 GB for large-v3)
# No extra install needed; openai-whisper is in requirements.txt
```

> **Note on UTMOS / NISQA:** These packages may require extra steps if the PyPI version is outdated.
> If `pip install utmos` fails, install from source:
> ```bash
> pip install git+https://github.com/sarulab-speech/UTMOS22.git
> ```

---

## 3. Configure Endpoints

Edit `eval/config.yaml` with your actual AI Core values:

```yaml
riva:
  grpc_uri: <aicore-grpc-host>:<port>    # e.g. grpc-host.eu-central-1.aws.ml.hana.ondemand.com:443
  use_ssl: true
  auth_token_env: AICORE_BEARER_TOKEN    # name of the env var holding your token

stt:
  model_name: parakeet-1.1b-en-US-asr-offline
  rest_endpoint: https://<aicore-host>/v1/deployments/<stt-deployment-id>/predictions

tts:
  model_name: Magpie-Multilingual.EN-US.Aria
  voice_name: Magpie-Multilingual.EN-US.Aria
  rest_endpoint: https://<aicore-host>/v1/deployments/<tts-deployment-id>/predictions
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

### UTMOS / SpeechBrain downloads fail
These models download weights on first use (~hundreds of MB). Ensure the VM has internet access and enough disk space (50 GB recommended).

### `ffmpeg not found` (pydub / librosa errors)
```bash
sudo apt-get install -y ffmpeg
```

### Out of disk space
Results with audio files can reach 20–50 GB. Check with `df -h` and clear old runs:
```bash
rm -rf results/tts/naturalness/*.wav
```
The `.jsonl` and `.csv` files are small; only the `.wav` files are large.
