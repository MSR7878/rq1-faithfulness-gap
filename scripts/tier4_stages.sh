#!/usr/bin/env bash
# RQ1 Tier 4: run the given stages back to back; stop at the first failed job list or failed job.
#   bash scripts/tier4_stages.sh L1:gea L1:fid M1:train:6 ...      (optional third field = workers for that stage)
# Each finished item writes runs/rq1_tier4/<item>/ITEM_DONE (the last stage of an item is marked with "!").
set -u
cd "$(dirname "$0")/.."
D=runs/rq1_tier4
rm -f $D/STAGES_DONE
for st in "$@"; do
  last=0; [[ $st == *'!' ]] && { last=1; st=${st%!}; }
  item=${st%%:*}; rest=${st#*:}; phase=${rest%%:*}; w=""; [[ $rest == *:* ]] && w=${rest#*:}
  bash scripts/tier4_queue.sh "$item" "$phase" $w
  if [ -f $D/$item/STOPPED_$phase ]; then echo "stage $item $phase stopped: $(cat $D/$item/STOPPED_$phase)" > $D/STAGES_STOPPED; exit 1; fi
  if [ -f $D/$item/FAILED_$phase ]; then echo "stage $item $phase had failed jobs" > $D/STAGES_STOPPED; exit 1; fi
  [ $last = 1 ] && echo "$item finished $(date '+%F %T')" > $D/$item/ITEM_DONE
done
echo "stages $* finished $(date '+%F %T')" > $D/STAGES_DONE
