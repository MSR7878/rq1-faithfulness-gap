#!/usr/bin/env bash
# RQ1 revision, T4 addendum: PGExplainer re-runs to recover pre-sigmoid edge logits (W workers, GPU 1, one thread each).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=runs/rq1_revision/logits
W=${WORKERS:-4}
mkdir -p $R/logs
$PY -m src.analysis.revision_logits list > $R/units.txt
[ -s $R/units.txt ] || { echo "empty unit list" > $R/STOPPED; exit 1; }
rm -f $R/logs/chunk_*
split -n r/$W -d $R/units.txt $R/logs/chunk_
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.revision_logits run --units $(cat $f) > $f.log 2>&1 & done
wait
echo "logits finished $(date '+%F %T'): $(ls $R/*.json | wc -l)/$(wc -l < $R/units.txt)" > $R/FINISHED
