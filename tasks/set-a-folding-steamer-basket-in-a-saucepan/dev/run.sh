#!/bin/bash
# Dev only: one simulator trial of the scripted demo. dev/run.sh <label> [max_steps]; variants and tuning via STEAMER_* env vars.
# Created 11_17_26.
set -u
cd "$(dirname "$0")/.."
source ~/IsaacLab-Arena/yam.env
[[ -e /lib/aarch64-linux-gnu/libgomp.so.1 ]] && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 PYTORCH_JIT=0  # aarch64 hosts only
LOCK=${SIM_LOCK:-$HOME/sim.lock}
OUT=~/yam-eval/steamer-$1
rm -rf "$OUT"; mkdir -p "$OUT"
flock "$LOCK" inspect-robots run --embodiment yam_steamer_basket -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=${CAM_W:-320} -E cam_height=${CAM_H:-176} --policy yam_scripted --no-prompt \
  --instruction "Put the steamer basket in the saucepan ready to steam: standing on its feet with every leaf inside the pan, and its handle pushed down so the lid will fit." \
  --scorer task_score --max-steps "${2:-900}" --seed 0 --store-frames --log-dir "$OUT" > "$OUT/stdout.log" 2>&1
echo "exit $?" >> "$OUT/stdout.log"
grep -aE "task_score|Traceback|Error" "$OUT/stdout.log" | tail -5
