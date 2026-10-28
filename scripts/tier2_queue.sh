#!/usr/bin/env bash
# RQ1 Tier 2: run one item's job list (one shell command per line) with W workers on GPU 1, one thread each.
#   bash scripts/tier2_queue.sh ITEM [PHASE]     (job list from: python -m src.analysis.tier2_jobs ITEM [PHASE])
# Re-runnable: every job skips work whose output exists. Writes runs/rq1_tier2/<ITEM>/FINISHED_<PHASE> at the end;
# a failed job is listed in runs/rq1_tier2/<ITEM>/FAILED_<PHASE>.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export PY CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ITEM=$1; PHASE=${2:-all}; W=${WORKERS:-12}
D=runs/rq1_tier2/$ITEM
mkdir -p $D/logs
$PY -m src.analysis.tier2_jobs $ITEM $PHASE > $D/jobs_$PHASE.txt || { echo "job list failed" > $D/STOPPED_$PHASE; exit 1; }
rm -f $D/FAILED_$PHASE $D/FINISHED_$PHASE
echo "start $(date '+%F %T'), $(wc -l < $D/jobs_$PHASE.txt) jobs, $W workers" > $D/STARTED_$PHASE
nl -ba -w1 -s' ' $D/jobs_$PHASE.txt | xargs -P $W -L 1 bash -c 'n=$0; shift; cmd="$*"; log='"$D"'/logs/'"$PHASE"'_$n.log; eval "$cmd" > $log 2>&1 || echo "$n $cmd" >> '"$D"'/FAILED_'"$PHASE"
echo "finished $(date '+%F %T'); failed jobs: $( [ -f $D/FAILED_$PHASE ] && wc -l < $D/FAILED_$PHASE || echo 0)" > $D/FINISHED_$PHASE
