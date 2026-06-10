#!/usr/bin/env bash
# Idempotent setup for civ-sim council LLM on DGX Spark.
# Run once before first use; safe to re-run.
# Usage: bash scripts/setup_llm.sh [--hf-token TOKEN] [--fallback-model]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv"
PRIMARY_MODEL="RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic"
FALLBACK_MODEL="meta-llama/Llama-3.3-70B-Instruct"
HF_TOKEN=""
USE_FALLBACK=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --hf-token) HF_TOKEN="$2"; shift 2 ;;
    --fallback-model) USE_FALLBACK=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODEL_ID="$PRIMARY_MODEL"
[[ "$USE_FALLBACK" -eq 1 ]] && MODEL_ID="$FALLBACK_MODEL"

echo "=== civ-sim Council LLM Setup ==="
echo "Project: $PROJECT_DIR"
echo "Model:   $MODEL_ID"
echo ""

# 1. Check .venv
echo "[1/5] Checking Python virtual environment..."
if [[ ! -f "$VENV/bin/python" ]]; then
  echo "ERROR: .venv not found at $VENV"
  echo "Create it with: python3.12 -m venv .venv && .venv/bin/pip install -e ."
  exit 1
fi
echo "  OK: $("$VENV/bin/python" --version)"

# 2. Check CUDA
echo "[2/5] Checking CUDA..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found. CUDA may not be installed."
  echo "  Continuing — vLLM will fail at runtime without CUDA."
else
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  echo "  OK: $GPU_INFO"
fi

# 3. Install/upgrade vLLM
echo "[3/5] Installing/upgrading vLLM..."
VLLM_VER=$("$VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "")
if [[ -z "$VLLM_VER" ]]; then
  echo "  Installing vLLM (this may take 10-15 minutes)..."
  "$VENV/bin/pip" install "vllm>=0.6.0" --quiet
elif "$VENV/bin/python" -c "from packaging.version import Version; exit(0 if Version('$VLLM_VER') >= Version('0.6.0') else 1)" 2>/dev/null; then
  echo "  OK: vLLM $VLLM_VER already installed"
else
  echo "  Upgrading vLLM from $VLLM_VER to >=0.6.0..."
  "$VENV/bin/pip" install "vllm>=0.6.0" --quiet
fi
"$VENV/bin/pip" install "huggingface_hub>=0.23" --quiet

# 4. HuggingFace login
if [[ -n "$HF_TOKEN" ]]; then
  echo "[4/5] Logging in to HuggingFace..."
  "$VENV/bin/hf" auth login --token "$HF_TOKEN"
  echo "  OK: logged in"
else
  echo "[4/5] HuggingFace login (skipped — pass --hf-token TOKEN to automate)"
  echo "  If model download fails, run: .venv/bin/hf auth login"
fi

# 5. Download model
echo "[5/5] Downloading model: $MODEL_ID"
CACHED=$("$VENV/bin/python" -c "
from huggingface_hub import try_to_load_from_cache
r = try_to_load_from_cache('$MODEL_ID', 'config.json')
print('yes' if r and r != 'not_in_cache_or_no_result' else 'no')
" 2>/dev/null || echo "no")

if [[ "$CACHED" == "yes" ]]; then
  echo "  OK: model already cached, skipping download"
else
  echo "  Downloading (~70 GB fp8 pre-quantized, this will take a while)..."
  if ! "$VENV/bin/hf" download "$MODEL_ID" --repo-type model; then
    echo ""
    echo "  FAILED to download $MODEL_ID"
    if [[ "$MODEL_ID" == "$PRIMARY_MODEL" ]]; then
      echo "  Accept the license at: https://huggingface.co/$PRIMARY_MODEL"
      echo "  Re-run with: bash scripts/setup_llm.sh --fallback-model --hf-token TOKEN"
    else
      echo "  Accept Meta's license at: https://huggingface.co/$FALLBACK_MODEL"
    fi
    exit 1
  fi
  echo "  OK: download complete"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Start vLLM server (terminal 1):"
echo "       bash scripts/start_vllm_council.sh"
echo "  2. Wait for 'Application startup complete' in the server log."
echo "  3. Smoke-test:  curl http://localhost:8000/v1/models"
echo "  4. Run sim:     .venv/bin/python main.py --ticks 200 --no-visualize --config examples/council_dgx.yaml"
