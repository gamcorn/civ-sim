#!/usr/bin/env bash
# Start vLLM server for civ-sim council provider.
# Run in a separate terminal before launching the simulation.
# Usage: bash scripts/start_vllm_council.sh [model_id]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"
MODEL="${1:-nvidia/Llama-3.1-Nemotron-70B-Instruct-HF}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[council-vllm] Detected ${GPU_COUNT} GPU(s) → tensor-parallel-size=${GPU_COUNT}"
echo "[council-vllm] Model: ${MODEL}"
echo "[council-vllm] Endpoint: http://localhost:8000/v1"

"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$GPU_COUNT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8000
