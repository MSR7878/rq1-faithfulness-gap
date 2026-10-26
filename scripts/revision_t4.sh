#!/usr/bin/env bash
# RQ1 revision T4: PGExplainer re-training, 35 Tox21 units x 5 settings, W workers on GPU 1. Re-runnable (finished jobs skipped).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_revision/t4
W=${WORKERS:-6}
mkdir -p $R/logs
$PY -m src.analysis.revision_t4 list > $R/jobs.txt
[ -s $R/jobs.txt ] || { echo "empty job list" > $R/STOPPED; exit 1; }
rm -f $R/logs/chunk_*
split -n r/$W -d $R/jobs.txt $R/logs/chunk_
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.revision_t4 run --jobs $(cat $f) > $f.log 2>&1 & done
wait
echo "t4 finished $(date '+%F %T'); jobs done: $(ls $R/jobs/*.json | wc -l)/$(wc -l < $R/jobs.txt)" > $R/FINISHED
