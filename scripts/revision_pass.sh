#!/usr/bin/env bash
# RQ1 revision: re-scoring pass (Part G on ground-truth units, Part E on all units), W workers on GPU 1.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker: avoids CPU oversubscription
R=runs/rq1_revision/pass
W=${WORKERS:-4}
mkdir -p $R/logs
$PY -m src.analysis.revision_pass list > $R/units_all.txt
awk '$2 ~ /G/ {print $1}' $R/units_all.txt > $R/units_G.txt
awk '{print $1}' $R/units_all.txt > $R/units_E.txt
for P in G E; do
  [ -s $R/units_$P.txt ] || { echo "empty unit list $P" > $R/STOPPED; exit 1; }
  rm -f $R/logs/chunk_${P}_*
  split -n r/$W -d $R/units_$P.txt $R/logs/chunk_${P}_
  for f in $R/logs/chunk_${P}_??; do $PY -u -m src.analysis.revision_pass run --part $P --units $(cat $f) > $f.log 2>&1 & done
  wait
  echo "part $P done $(date '+%F %T'): $(ls $R/$P/*.npz | wc -l)/$(wc -l < $R/units_$P.txt); tracebacks: $(grep -l Traceback $R/logs/chunk_${P}_*.log 2>/dev/null | wc -l)" >> $R/run.log
done
echo "pass finished $(date '+%F %T')" > $R/FINISHED
