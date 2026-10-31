#!/usr/bin/env bash
# RQ1 Tier 2: run the remaining compute stages back to back once S priority has finished (order approved by the user):
# S priority scoring -> S sensitivity -> M (GNN/PG, then gradients + SubgraphX, then scoring) -> S remainder (+ scoring).
# Reports are written separately after each stage is checked. Stops at the first stage with failed jobs.
set -u
cd "$(dirname "$0")/.."
D=runs/rq1_tier2
while [ ! -f $D/S/FINISHED_prio ]; do sleep 120; done
run() {  # ITEM PHASE
  bash scripts/tier2_queue.sh "$1" "$2"
  if [ -f $D/$1/STOPPED_$2 ]; then echo "stage $1 $2 stopped: $(cat $D/$1/STOPPED_$2)" > $D/CHAIN_STOPPED; exit 1; fi
  if [ -f $D/$1/FAILED_$2 ]; then echo "stage $1 $2 had failed jobs" > $D/CHAIN_STOPPED; exit 1; fi
}
run S prio_score
run S sens
while [ ! -f $D/M/FINISHED_train ]; do sleep 120; done
[ -f $D/M/FAILED_train ] && { echo "M train had failed jobs" > $D/CHAIN_STOPPED; exit 1; }
run M gp
run M explain
run M score
run S rest
run S rest_score
echo "chain finished $(date '+%F %T')" > $D/CHAIN_DONE
