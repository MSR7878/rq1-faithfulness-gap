#!/usr/bin/env bash
# RQ1 Part A (GT reliance) + Part B (logit / random-corrected fidelity) on existing models and caches.
# GPU 1 only. Writes only under runs/rq1_reliance_fidelity/. Ends with the report and a DONE file.
#   screen -dmS rf bash scripts/reliance_fidelity.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_reliance_fidelity
W=${WORKERS:-4}
mkdir -p $R/logs
stamp() { date '+%F %T'; }
echo "[$(stamp)] start (workers=$W)" >> $R/run.log
for PART in A B; do
  $PY -m src.analysis.reliance_fidelity list | awk -v p=$PART '$2 ~ p {print $1}' > $R/units_$PART.txt
  split -n r/$W -d $R/units_$PART.txt $R/logs/chunk_${PART}_
  for f in $R/logs/chunk_${PART}_*; do
    $PY -u -m src.analysis.reliance_fidelity run --part $PART --units $(cat $f) > $f.log 2>&1 &
  done
  wait
  echo "[$(stamp)] part $PART done: $(ls $R/part$PART/*.npz 2>/dev/null | wc -l)/$(wc -l < $R/units_$PART.txt) units; errors: $(grep -l Traceback $R/logs/chunk_${PART}_*.log 2>/dev/null | wc -l)" >> $R/run.log
done
$PY -u -m src.analysis.reliance_fidelity_ties > $R/logs/ties.log 2>&1
echo "[$(stamp)] ties done: $(tail -1 $R/logs/ties.log)" >> $R/run.log
$PY -u -m src.analysis.reliance_fidelity_report > $R/report.txt 2> $R/logs/report.err
echo "[$(stamp)] report done (stderr lines: $(wc -l < $R/logs/report.err))" >> $R/run.log
echo "finished $(stamp); report: $R/report.txt" > $R/DONE
