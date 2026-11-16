#!/bin/bash
# Dev only: one simulator trial. dev/run.sh <label> [max_steps]; SOLDER_DEMO, SOLDER_DEBUG and the *_CAL variables pass through.
source ~/IsaacLab-Arena/yam.env
[ -e /lib/aarch64-linux-gnu/libgomp.so.1 ] && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 PYTORCH_JIT=0
cd "$(dirname "$0")/.."
mkdir -p ~/yam-eval
LOCK=${SIM_LOCK:-/tmp/sim.lock}
flock "$LOCK" inspect-robots run --embodiment yam_solder -E task=task.py -E object_jitter=0 -E control_hz=20 \
  -E cam_width=320 -E cam_height=176 --policy yam_scripted --no-prompt \
  --instruction "Solder the wire into the lug with the iron and the solder, let the joint set, and put the iron and the solder back." \
  --scorer task_score --max-steps "${2:-2000}" --seed 0 --store-frames --log-dir ~/yam-eval/lug-solder-"$1" > ~/yam-eval/lug-solder-"$1".out 2>&1
