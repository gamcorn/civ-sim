#!/usr/bin/env bash
# Start a civ-sim tmux session with named windows for each service.
# Usage: bash scripts/tmux-session.sh [session-name]

SESSION="${1:-civ-sim}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists — attaching."
  tmux attach -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n "claude"   # window 1: Claude CLI
tmux new-window  -t "$SESSION" -n "vllm"        # window 2: vLLM server
tmux new-window  -t "$SESSION" -n "sim"         # window 3: simulation runs
tmux new-window  -t "$SESSION" -n "logs"        # window 4: tail logs / duckdb

# Set working directory for all windows
for win in claude vllm sim logs; do
  tmux send-keys -t "$SESSION:$win" "cd /home/arturo/projects/civ-sim" Enter
done

# Helpful hints pre-typed (not executed) in each window
tmux send-keys -t "$SESSION:vllm" "bash scripts/start_vllm_nano.sh" ""
tmux send-keys -t "$SESSION:sim"  ".venv/bin/python main.py --ticks 200 --no-visualize --config examples/council_nano.yaml" ""

# Start on the claude window
tmux select-window -t "$SESSION:claude"
tmux attach -t "$SESSION"
