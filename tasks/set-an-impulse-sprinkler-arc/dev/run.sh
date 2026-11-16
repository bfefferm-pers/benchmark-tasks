#!/bin/bash
# Dev only: one simulator trial of the scripted demo. dev/run.sh <label> [max_steps]; variants and tuning via SPRINKLER_* env vars.
set -u
cd "$(dirname "$0")/.."
source ~/IsaacLab-Arena/yam.env
[[ -e /lib/aarch64-linux-gnu/libgomp.so.1 ]] && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 PYTORCH_JIT=0  # aarch64 hosts only
LOCK=${SIM_LOCK:-/tmp/sprinkler-sim.lock}
OUT=~/yam-eval/sprinkler-$1
mkdir -p "$OUT"
flock "$LOCK" inspect-robots run --embodiment yam_sprinkler -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=${CAM_W:-320} -E cam_height=${CAM_H:-176} --policy yam_scripted -P hold_s=${HOLD:-3} --no-prompt \
  --instruction "Set the sprinkler so it waters only the bed between the two orange flags, then take your hands away." \
  --scorer task_score --max-steps "${2:-2400}" --seed 0 --store-frames --log-dir "$OUT" > "$OUT/stdout.log" 2>&1
echo "exit $?" >> "$OUT/stdout.log"
grep -E "task_score|Traceback|Error" "$OUT/stdout.log" | tail -5
