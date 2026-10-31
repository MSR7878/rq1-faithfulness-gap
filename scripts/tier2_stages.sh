#!/usr/bin/env bash
# RQ1 Tier 2: run the given stages back to back (same stop rules as tier2_chain.sh), e.g.
#   bash scripts/tier2_stages.sh S:rest S:rest_score
# Stops at the first stage whose job list failed (STOPPED_) or that had failed jobs (FAILED_), writing CHAIN_STOPPED;
# writes CHAIN_DONE when every stage finished cleanly.
set -u
cd "$(dirname "$0")/.."
D=runs/rq1_tier2
rm -f $D/CHAIN_DONE
for st in "$@"; do
  item=${st%%:*}; phase=${st#*:}
  bash scripts/tier2_queue.sh "$item" "$phase"
  if [ -f $D/$item/STOPPED_$phase ]; then echo "stage $item $phase stopped: $(cat $D/$item/STOPPED_$phase)" > $D/CHAIN_STOPPED; exit 1; fi
  if [ -f $D/$item/FAILED_$phase ]; then echo "stage $item $phase had failed jobs" > $D/CHAIN_STOPPED; exit 1; fi
done
echo "stages $* finished $(date '+%F %T')" > $D/CHAIN_DONE
