#!/bin/bash
# Dev only: the demo twice, then each shortcut variant, one after the other. dev/all.sh [prefix]
cd "$(dirname "$0")"
export SOLDER_DEBUG=1
P=${1:-}
for v in demo1 demo2 hover tip none yank ironfirst short untidy; do
  case $v in demo*) export SOLDER_DEMO= ;; *) export SOLDER_DEMO=$v ;; esac
  ./run.sh "$P$v"
  echo "$v: $(grep -aoE 'task_score: [0-9.]+' ~/yam-eval/lug-solder-$P$v.out | tail -1)"
done
