#!/usr/bin/env bash
# RQ1 Tier 3: run one job list with W workers (GPU from the environment), one thread each. Same layout as tier2_queue.sh.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}; export PY
ITEM=$1; PHASE=$2; W=${WORKERS:-12}
D=runs/rq1_tier3/$ITEM
mkdir -p $D/logs
rm -f $D/STOPPED_$PHASE
$PY -m src.analysis.tier3_jobs $ITEM $PHASE > $D/jobs_$PHASE.txt || { echo "job list failed" > $D/STOPPED_$PHASE; exit 1; }
rm -f $D/FAILED_$PHASE $D/FINISHED_$PHASE
echo "start $(date '+%F %T'), $(wc -l < $D/jobs_$PHASE.txt) jobs, $W workers" > $D/STARTED_$PHASE
nl -ba -w1 -s' ' $D/jobs_$PHASE.txt | xargs -P $W -L 1 bash -c 'n=$0; cmd="$*"; log='"$D"'/logs/'"$PHASE"'_$n.log; eval "$cmd" > $log 2>&1 || echo "$n $cmd" >> '"$D"'/FAILED_'"$PHASE"
echo "finished $(date '+%F %T'); failed jobs: $( [ -f $D/FAILED_$PHASE ] && wc -l < $D/FAILED_$PHASE || echo 0)" > $D/FINISHED_$PHASE
