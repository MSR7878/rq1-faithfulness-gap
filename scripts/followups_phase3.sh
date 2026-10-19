#!/usr/bin/env bash
# RQ1 follow-ups, Phase 3: PGExplainer collapse diagnosis (4 workers, GPU 1), then the report.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_followups/phase3
mkdir -p $R/logs
$PY -m src.analysis.followups_phase3 list > $R/units.txt
split -n r/4 -d $R/units.txt $R/logs/chunk_
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.followups_phase3 run --units $(cat $f) > $f.log 2>&1 & done
wait
$PY -u -m src.analysis.followups_phase3 report > $R/report.txt 2> $R/logs/report.err
echo "phase 3 finished $(date '+%F %T')" > $R/FINISHED
