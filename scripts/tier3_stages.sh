#!/usr/bin/env bash
# RQ1 Tier 3: run the given stages back to back; stop at the first failed job list or failed job.
#   bash scripts/tier3_stages.sh E:explain E:score
set -u
cd "$(dirname "$0")/.."
D=runs/rq1_tier3
rm -f $D/STAGES_DONE
for st in "$@"; do
  item=${st%%:*}; phase=${st#*:}
  bash scripts/tier3_queue.sh "$item" "$phase"
  if [ -f $D/$item/STOPPED_$phase ]; then echo "stage $item $phase stopped: $(cat $D/$item/STOPPED_$phase)" > $D/STAGES_STOPPED; exit 1; fi
  if [ -f $D/$item/FAILED_$phase ]; then echo "stage $item $phase had failed jobs" > $D/STAGES_STOPPED; exit 1; fi
done
echo "stages $* finished $(date '+%F %T')" > $D/STAGES_DONE
