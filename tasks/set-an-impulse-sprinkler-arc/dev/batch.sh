#!/bin/bash
# Dev only: the demo twice, the shortcut variants and the stall test, one after another (run inside the task folder).
# HOLD keeps the scripted policy from stopping the trial before the run limit, so the water runs as long as it would for a
# model policy; the demo itself ends earlier, on success.
cd "$(dirname "$0")/.."
run() { label=$1; shift; env HOLD=70 "$@" bash dev/run.sh "$label" 2400 > /dev/null; echo "== $label $*: $(grep -h task_score ~/yam-eval/sprinkler-$label/stdout.log | tail -1)"; grep "SPRINKLER t" ~/yam-eval/sprinkler-$label/stdout.log | tail -1 | sed 's/L \[.*R \[[^]]*\]//' | cut -c1-400; }
run demo1 SPRINKLER_DEBUG=1
run demo2 SPRINKLER_DEBUG=1
run v-nothing SPRINKLER_DEBUG=1 SPRINKLER_DEMO=nothing
run v-lower_only SPRINKLER_DEBUG=1 SPRINKLER_DEMO=lower_only
run v-short SPRINKLER_DEBUG=1 SPRINKLER_DEMO=short
run v-head_only SPRINKLER_DEBUG=1 SPRINKLER_DEMO=head_only
run v-no_head SPRINKLER_DEBUG=1 SPRINKLER_DEMO=no_head
[[ "${1:-}" == "--cal" ]] && run cal SPRINKLER_DEBUG=1 SPRINKLER_DEMO=nothing SPRINKLER_CAL=1
exit 0
