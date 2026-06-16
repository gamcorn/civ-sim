#!/usr/bin/env bash
# Start vLLM server for civ-sim council provider.
# Run in a separate terminal before launching the simulation.
# Usage: bash scripts/start_vllm_council.sh [model_id]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv"
MODEL="${1:-RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic}"

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
echo "[council-vllm] Detected ${GPU_COUNT} GPU(s) → tensor-parallel-size=${GPU_COUNT}"
echo "[council-vllm] Model: ${MODEL}"
echo "[council-vllm] Endpoint: http://localhost:8000/v1"
echo "[council-vllm] Weights: pre-quantized fp8 (~70 GB on disk, no on-the-fly quantization)"

# Prepend venv bin to PATH so flashinfer JIT can find the ninja build tool
export PATH="$VENV/bin:$PATH"

# expandable_segments avoids fragmentation OOMs during model load.
# enforce-eager disables CUDA graph capture (cicc JIT compiler) which OOM-kills
# on this machine (121 GB unified RAM, no swap) when compiling batch-size graphs
# while 70B model weights are already resident. ~10% throughput cost.
# gpu-memory-utilization 0.70 leaves ~36 GB for OS and Python overhead.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --tensor-parallel-size "$GPU_COUNT" \
  --enforce-eager \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8000
