#!/usr/bin/env bash
# Start vLLM server for Nemotron-3-Nano-30B FP8 (MoE, 3.5B active params).
# Much faster inference than 70B — good for high-tick-rate sim runs.
# Run in a separate terminal before launching the simulation.
# Usage: bash scripts/start_vllm_nano.sh [model_id]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"
MODEL="${1:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[nano-vllm] Detected ${GPU_COUNT} GPU(s) → tensor-parallel-size=${GPU_COUNT}"
echo "[nano-vllm] Model: ${MODEL}"
echo "[nano-vllm] Endpoint: http://localhost:8000/v1"
echo "[nano-vllm] Weights: ~16 GB fp8 MoE (3.5B active params/token)"

# Prepend venv bin to PATH so flashinfer JIT can find the ninja build tool
# (FlashInfer calls ninja via subprocess with bare name, not full path)
export PATH="$VENV/bin:$PATH"

# VLLM_USE_FLASHINFER_MOE_FP8: required for fp8 MoE kernel on Blackwell.
# expandable_segments: avoids fragmentation OOMs during model load.
# enforce-eager disables CUDA graph capture (cicc JIT compiler) which OOM-kills
# on this machine when compiling 19 batch-size graphs while model is in memory.
# ~10% throughput cost, irrelevant for simulation workloads.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name model \
  --tensor-parallel-size "$GPU_COUNT" \
  --trust-remote-code \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --host 0.0.0.0 \
  --port 8000
