#!/usr/bin/env bash
# Outputs colored metrics for tmux status-right.
# tmux processes #[fg=...] codes in #() output, so colors work here.

GPU_C='#[fg=#a6e3a1]'   # green
CPU_C='#[fg=#f38ba8]'   # red
RAM_C='#[fg=#fab387]'   # peach
SWP_C='#[fg=#f9e2af]'   # yellow
SEP='#[fg=#585b70] │ '

gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{
    if($2~/N\/A/) printf "GPU %s%%", $1
    else printf "GPU %s%% %sMiB", $1, $2
  }')

# vmstat 1 1: 1-second sample, locale-safe; idle is column 15
cpu=$(vmstat 1 1 2>/dev/null | awk 'NR==3{printf "CPU %d%%", 100-$15}')

# Mem: $2=total $3=used; Échange (swap): NR==3 $2=total $3=used
ram=$(free -h 2>/dev/null | awk '/^Mem/{printf "RAM %s/%s", $3, $2}')
swp=$(free -h 2>/dev/null | awk 'NR==3{if($3!="0B" && $3!="0") printf "SWP %s", $3}')

out="${GPU_C}${gpu}${SEP}${CPU_C}${cpu}${SEP}${RAM_C}${ram}"
[[ -n "$swp" ]] && out+="${SEP}${SWP_C}${swp}"
echo "$out"
