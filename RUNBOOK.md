# Evaluation Runbook — NVIDIA Riva on SAP AI Core

## TL;DR

**Run on the Azure VM, not your laptop.**
The VM is in Germany West Central, which is the same region as AI Core (Frankfurt / eu-central-1).
Running locally adds 10–40 ms of network jitter to every single API call, which contaminates all latency and RTF measurements.

---

## 1. Prerequisites (do these before VM setup)

### 1.1 HuggingFace account and dataset access

STT tests load datasets directly from HuggingFace. Several are gated and require you to accept terms before they can be downloaded.

1. Create a free account at https://huggingface.co if you don't have one
2. Generate a read-access token at https://huggingface.co/settings/tokens
3. Accept terms for each gated dataset (click **Agree and access repository**):
   - https://huggingface.co/datasets/speechcolab/gigaspeech
   - https://huggingface.co/datasets/kensho/spgispeech
4. `librispeech_asr`, `LIUM/tedlium`, `revdotcom/earnings22`, `edinburghcst/ami`, and `mozilla-foundation/common_voice_17_0` are public — no approval needed.

> Tests skip gracefully if a dataset is unavailable — it is not a fatal error.

### 1.2 SAP AI Core credentials (`.env`)

The harness authenticates to the AI Core gateway with an OAuth Bearer token. For a full
(~7 hour) run, use the **client-credentials flow** so tokens are minted and **auto-refreshed** —
a static token expires in ~4 h and would kill a long run mid-way.

Credentials live in a git-ignored `.env` at the repo root (`.gitignore` covers `.env` / `*.env`).
**Never put the secret values in `config.yaml` or `config.py`** — only in `.env` or the shell.

1. Copy the template and fill in real values:
   ```bash
   cp .env.example .env
   # then edit .env
   ```
2. Set the client-credentials (recommended — auto-refreshing):
   ```
   AICORE_CLIENT_ID=<service-key clientid>
   AICORE_CLIENT_SECRET=<service-key clientsecret>
   AICORE_AUTH_URL=https://<subdomain>.authentication.<region>.hana.ondemand.com
   ```
   (The harness appends `/oauth/token` and uses `grant_type=client_credentials`.)
3. *Or* set a static token for short/single-test runs only:
   ```
   AICORE_BEARER_TOKEN=eyJ...    # used verbatim, no refresh, expires ~4h
   ```
   If `AICORE_BEARER_TOKEN` is set it **takes priority** over the client-credentials vars.

> **`.env` is git-ignored, so it does not sync via `git clone`/`git pull`.** Copy it to the VM
> separately (`scp .env <vm-user>@<vm-ip>:~/nvidia-model-evaluation/.env`) or recreate it there
> from `.env.example`. `config.py::_load_dotenv()` loads it automatically on every run; real
> shell env vars override the file, so CI/secret-manager injection still works.

---

## 2. Where to Run What

| Task | Where | Why |
|------|-------|-----|
| All eval tests (phase0 + 13 STT + 9 TTS) | **Azure VM** | Same region as AI Core → clean latency numbers |
| Whisper transcription (local GPU) | **Azure VM** | NC24ads A100 handles large-v3 in real time |
| UTMOS / SpeechBrain scoring | **Azure VM** | GPU-accelerated; takes hours on CPU |
| Viewing the HTML report | Laptop | Copy `results/report.html` back via SCP |
| Editing config / fixing code | Either | Up to you |

---

## 3. One-Time VM Setup

SSH into your Azure VM, then run the setup script once. It handles everything automatically.

### 3.1 Clone the repo

```bash
git clone <your-repo-url> nvidia-model-evaluation
cd nvidia-model-evaluation
```

Or sync from your laptop:

```bash
# Run from your laptop:
rsync -avz --exclude results/ --exclude __pycache__ \
    /Users/I350548/Downloads/nvidia/nvidia-model-evaluation/ \
    <vm-user>@<vm-ip>:~/nvidia-model-evaluation/
```

### 3.2 Run setup

```bash
bash scripts/setup.sh
```

This script does everything in order:
1. Installs system packages (`ffmpeg`, `libsndfile1`, `libopenh264`, etc.)
2. Creates cache directories in `~/hf_home`, `~/torch_home`, `~/pip_cache` (on the 512 GB persistent OS disk) and persists them in `~/.bashrc`
3. Creates `.venv` and installs `uv` (faster pip resolver)
4. Pre-installs `omegaconf>=2.1` to work around a pip 24+ incompatibility
5. Installs all Python dependencies via `uv pip install -r requirements.txt`
6. Sets `LD_LIBRARY_PATH` to PyTorch's bundled CUDA libs (needed by UTMOS)
7. Runs `hf auth login` — paste your HF token when prompted

> **Note:** Setup takes 10–20 minutes on first run (downloading PyTorch, SpeechBrain, etc.)

### 3.3 Pre-download datasets (recommended)

Download all evaluation datasets into the local cache **once**, before the first eval run. This means the eval never re-downloads datasets — even after cleaning results.

```bash
bash scripts/download_datasets.sh
```

This downloads into `~/hf_home/datasets` and survives `clean.sh` runs.

You can also download a single dataset:

```bash
bash scripts/download_datasets.sh librispeech_clean
```

Available names: `librispeech_clean`, `librispeech_other`, `tedlium`, `gigaspeech`, `spgispeech`, `earnings22`, `ami`, `common_voice_en`, `ljspeech`.

> **Disk space:** All test splits together are ~20–40 GB. With a 512 GB OS disk this is fine.

---

## 4. Configure Endpoints

Edit `eval/config.yaml` with your actual AI Core values. Each service has its own `grpc_uri`.
**Endpoints and env-var *names* go here; secret *values* go in `.env` (see §1.2).**

```yaml
riva:
  use_ssl: true
  auth_token_env: AICORE_BEARER_TOKEN   # env-var NAME only (value lives in .env)

stt:
  grpc_uri: <parakeet-host>:443        # gRPC host:port (no https://)
  model_name: parakeet-1.1b-en-US-asr-offline
  rest_endpoint: https://<parakeet-host>/v1/audio/transcriptions
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 120
  # Opt-in multilingual depth (default [] = English-only, byte-for-byte unchanged):
  multilingual_languages: []           # FLEURS accuracy/acceptance (1.12/1.13)
  multilingual_extra_languages: []     # Tier-A per-language 1.5 + 1.9
  spot_check_languages: []             # Tier-B spot-checks 1.2 / 1.7 / 1.10

tts:
  grpc_uri: <magpie-host>:443          # gRPC host:port (no https://)
  model_name: Magpie-Multilingual.EN-US.Aria
  voice_name: Magpie-Multilingual.EN-US.Aria
  rest_endpoint: https://<magpie-host>/v1/audio/synthesize_online
  language_code: en-US
  auth_header: Authorization
  request_timeout_s: 60
  sample_rate: 22050
  supports_rest: false                 # this gateway is gRPC-streaming only
  # Opt-in multilingual TTS (default [] = English-only). Magpie supports en/fr/es,
  # each with its own voice. Run Test 2.9 first to see which the deployment exposes:
  multilingual_languages: []           # subset of [en, fr, es]
  voice_overrides: {}                  # lang key → deployed voice string, if different
```

> Auto-detect note: `multilingual_languages` keys are `en|fr|es` for TTS and the FLEURS
> language keys for STT. Leaving every list empty reproduces the English-only behaviour exactly.

---

## 5. Running the Evaluation

Use `start_eval.sh` — it handles everything: pulls latest code, resolves credentials, and runs
the eval inside a persistent tmux session so SSH disconnects don't interrupt it.

**Credentials:** if a `.env` with client-credentials exists (§1.2), the script uses it and
auto-refreshes tokens for the whole run — just press **Enter** at the token prompt. If you have
no `.env`, it prompts for a static `AICORE_BEARER_TOKEN` instead (fine for short runs).

```bash
bash scripts/start_eval.sh           # run everything (phase0 + stt + tts + report)
bash scripts/start_eval.sh stt       # STT tests only
bash scripts/start_eval.sh tts       # TTS tests only
bash scripts/start_eval.sh phase0    # connectivity check only
bash scripts/start_eval.sh report    # regenerate report from existing results
```

Monitor progress from outside tmux:

```bash
tail -f results/run.log
```

Re-attach to the tmux session:

```bash
tmux attach -t eval
# Detach without killing: Ctrl+B then D
```

---

### Run manually (without the start script)

If you prefer to run commands directly (e.g., for a single test):

```bash
source .venv/bin/activate
export HF_HOME=~/hf_home
# Credentials: either rely on .env (client-credentials auto-refresh, recommended)
# or export a static token for this shell:
export AICORE_BEARER_TOKEN="eyJ..."   # optional if .env has client-credentials

# Pre-download datasets (if not done yet):
python -m eval.run download

# Single test:
python -m eval.run stt --test accuracy
python -m eval.run tts --test naturalness
python -m eval.run tts --test tts_language_acceptance   # 2.9 — run first for multilingual

# Full run:
python -m eval.run all

# Dry-run (verify config without API calls):
python -m eval.run all --dry-run
```

Available STT test names:
`accuracy`, `performance`, `streaming`, `rest_vs_grpc`, `noise_robustness`, `accent`,
`long_form`, `domain`, `output_quality`, `confidence`, `format_robustness`, `multilingual`,
`language_acceptance`

Available TTS test names:
`naturalness`, `intelligibility`, `prosody`, `signal_quality`, `latency`, `concurrency`,
`edge_cases`, `long_form`, `tts_language_acceptance`

> **Multilingual tip:** set `tts.multilingual_languages: [en, fr, es]` in `config.yaml`, then run
> `tts_language_acceptance` (2.9) **first** — it probes which languages this deployment actually
> synthesizes (Whisper language-ID). Only run 2.1/2.2/2.3 per-language on whatever passes; a
> language the deployment won't synthesize is recorded as a `wrong_language`/`empty` serving gap,
> not a faked score. Numeric IDs also work: `--test 1.13`, `--test 2.9`.

---

## 6. Resuming After Interruption

Every API call result is written to a `.jsonl` file immediately. Re-running the same command **skips already-completed items** automatically — no data is lost and no duplicate API calls are made.

```bash
# Safe to re-run at any time — resumes from where it stopped:
bash scripts/start_eval.sh stt
```

---

## 7. Outputs

```
results/
├── phase0/
│   └── discovery.json            # Phase 0 findings
├── stt/
│   ├── accuracy/
│   │   ├── calls.jsonl           # Per-utterance results
│   │   └── summary.csv           # Dataset-level WER/CER/etc.
│   ├── multilingual/             # 1.12 — per-language FLEURS accuracy
│   ├── language_acceptance/      # 1.13 — per-language ok/empty/error gate
│   │   ├── <langkey>.jsonl       # per-language calls (opt-in depth tests)
│   │   ├── summary.csv
│   │   └── summary_<langkey>.csv # per-language summaries (1.5/1.9/etc.)
│   └── ...                       # One folder per test
├── tts/
│   ├── naturalness/
│   │   ├── calls.jsonl
│   │   ├── summary.csv
│   │   ├── summary_<langkey>.csv # per-language DNSMOS (fr/es; UTMOS en-only)
│   │   └── synth_0001.wav        # Synthesized audio files
│   ├── language_acceptance/      # 2.9 — per-language synth+Whisper gate
│   └── ...
├── run.log                       # Full run log (tee'd, append mode)
└── report.html                   # Full evaluation report (open in browser)
```

The dated per-run layout is `results/run-<DD-MM-YY>/riva/...`. A standalone, richer HTML report
(`docs/full-report.html`) can be regenerated with stdlib-only Python (no venv needed):

```bash
python3 eval/report/generate_full_report.py \
    --run results/run-<DD-MM-YY>/riva \
    --acceptance results/run-<DD-MM-YY>/riva/stt/language_acceptance/language_acceptance.csv \
    --out docs/full-report.html
```

> The 40-language acceptance matrix (Test 1.13) is produced by the harness itself on every
> `stt`/`all` run, so there's nothing to pre-seed — point `--acceptance` at the dated run dir.

Copy the report to your laptop:

```bash
scp <vm-user>@<vm-ip>:~/nvidia-model-evaluation/results/report.html ~/Desktop/
```

---

## 8. Cleaning Up

```bash
# Clear results only — keeps .venv and downloaded datasets:
bash scripts/clean.sh

# Also wipe the HuggingFace dataset cache (forces re-download):
bash scripts/clean.sh --data

# Full reset — also removes .venv (run setup.sh again afterwards):
bash scripts/clean.sh --full

# Combined:
bash scripts/clean.sh --full --data
```

> After a plain `clean.sh`, datasets are still cached in `~/hf_home/datasets`.
> Run `bash scripts/start_eval.sh` directly — no re-download needed.

---

## 9. Typical Workflow

```
First time on a new VM:
  bash scripts/setup.sh                  # ~15 min
  bash scripts/download_datasets.sh      # ~30-60 min (downloads ~20-40 GB)
  cp .env.example .env                   # fill in AI Core client-credentials (§1.2)
  # edit eval/config.yaml with endpoints

Every eval run:
  bash scripts/start_eval.sh             # Enter at prompt (uses .env) → runs in tmux

After a run (clean results, keep datasets):
  bash scripts/clean.sh
  bash scripts/start_eval.sh             # no re-download, starts immediately
```

---

## 10. Log Level & Debugging

```bash
# Verbose output:
python -m eval.run stt --test accuracy --log-level DEBUG

# Suppress most output:
python -m eval.run tts --log-level WARNING
```

---

## 11. Test Runtime Estimates (A100 VM, Frankfurt → AI Core)

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
| 1.12 Multilingual Accuracy | FLEURS × Joule langs | 30–60 min (per enabled subset) |
| 1.13 Language Acceptance | ~3 clips × 40 langs | 10–15 min |
| 2.1 Naturalness | 200 synths + UTMOS/DNSMOS | 30–45 min |
| 2.2 Intelligibility | 350 synths + Whisper | 45–60 min |
| 2.3 Prosody | 50 synths (reference-free) | 10 min |
| 2.4 Signal Quality | **N/A — no synthesis** | <1 min |
| 2.5 Latency | 500 calls | 20 min |
| 2.6 Concurrency | 500 calls | 15 min |
| 2.7 Edge Cases | ~100 calls | 15 min |
| 2.8 Long Form TTS | ~30 synths + ECAPA | 20 min |
| 2.9 Language Acceptance | ~3 synths × enabled langs + Whisper | 5–10 min |
| **Total** | | **~7–8 hours** (English-only; multilingual subsets add time) |

---

## 12. Troubleshooting

### `$AICORE_BEARER_TOKEN not set`
```bash
export AICORE_BEARER_TOKEN="eyJ..."
```
A static token from AI Core expires in ~4 h. If you get 401s mid-run, the cleanest fix is to use
the **client-credentials flow** instead (see §1.2): put `AICORE_CLIENT_ID` / `AICORE_CLIENT_SECRET`
/ `AICORE_AUTH_URL` in `.env` and leave `AICORE_BEARER_TOKEN` unset — the harness then mints and
auto-refreshes tokens for the whole run. (A static token, if set, always wins over `.env`.)

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
setuptools may be missing from the venv:
```bash
.venv/bin/pip install setuptools --no-cache-dir
```
Always use `.venv/bin/pip` (not plain `pip`) to target the venv.
TTS Test 2.3 (reference-free prosody) uses `librosa`; if pyworld is unavailable the F0 path falls
back gracefully. (Test 2.4 Signal Quality is N/A — it runs no synthesis and needs no audio libs.)

### `omegaconf` / `invalid-installed-package` error during pip install
pip 24+ rejects `omegaconf 2.0.6` (pulled in by `fairseq`/`utmos`). Fix:
```bash
.venv/bin/pip uninstall -y omegaconf
.venv/bin/pip install "omegaconf>=2.1" --no-cache-dir
.venv/bin/pip install -r requirements.txt --no-cache-dir
```
`setup.sh` handles this automatically.

### `resolution-too-deep` error during pip install
pip cannot resolve the dependency graph for complex packages (fairseq, utmos). Use `uv`:
```bash
pip install uv --no-cache-dir
uv pip install -r requirements.txt --no-cache-dir
```
`setup.sh` uses `uv` automatically.

### UTMOS `libcudart.so.13: cannot open shared object file`
PyTorch ships its own CUDA runtime. Add it to `LD_LIBRARY_PATH`:
```bash
TORCH_LIB=$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
```
`setup.sh` and `start_eval.sh` set this automatically.

### `libopenh264.so.5: cannot open shared object file` (ffmpeg)
```bash
sudo apt-get install -y libopenh264-dev
# If only .so.6 is available:
sudo ln -sf /usr/lib/x86_64-linux-gnu/libopenh264.so.6 \
            /usr/lib/x86_64-linux-gnu/libopenh264.so.5
```
`setup.sh` handles this automatically.

### `Unauthorized` / 403 on HuggingFace datasets
1. Make sure you are logged in: `hf auth login`
2. For GigaSpeech and SPGISpeech, accept terms at:
   - https://huggingface.co/datasets/speechcolab/gigaspeech
   - https://huggingface.co/datasets/kensho/spgispeech

### `[Errno 28] No space left on device`
Check disk usage:
```bash
df -h /
du -sh ~/hf_home ~/torch_home ~/pip_cache 2>/dev/null | sort -rh
```
Free up space:
```bash
# Remove pip wheel cache:
rm -rf ~/pip_cache/*

# Remove HuggingFace model weights (re-downloaded on next use):
rm -rf ~/hf_home/hub

# Remove dataset cache (re-download with download_datasets.sh):
bash scripts/clean.sh --data
```

### UTMOS / SpeechBrain model downloads fail
These models download weights on first use (~hundreds of MB). Ensure the VM has internet access and enough disk space. Check `~/hf_home/hub` for partial downloads and delete them before retrying.

### `ffmpeg not found` (pydub / librosa errors)
```bash
sudo apt-get install -y ffmpeg
```

### Out of disk space in results/
TTS audio files can reach 20–50 GB. The `.jsonl` and `.csv` files are small; only `.wav` files are large:
```bash
rm -rf results/tts/naturalness/*.wav
```
