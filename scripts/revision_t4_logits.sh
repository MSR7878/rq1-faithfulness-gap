#!/usr/bin/env bash
# RQ1 revision, T4 completion: settings b, c, d1, d2 on the 35 Tox21 units with the explainer MLP saved (W workers, GPU 1, one thread each).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=runs/rq1_revision/t4_logits
W=${WORKERS:-12}
mkdir -p $R/logs
$PY -m src.analysis.revision_t4_logits list > $R/jobs.txt
[ -s $R/jobs.txt ] || { echo "empty job list" > $R/STOPPED; exit 1; }
rm -f $R/logs/chunk_*
split -n r/$W -d $R/jobs.txt $R/logs/chunk_
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.revision_t4_logits run --jobs $(cat $f) > $f.log 2>&1 & done
wait
echo "t4_logits finished $(date '+%F %T'); jobs done: $(ls $R/*.json | wc -l)/$(wc -l < $R/jobs.txt)" > $R/FINISHED
