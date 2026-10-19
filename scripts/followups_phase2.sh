#!/usr/bin/env bash
# RQ1 follow-ups, Phase 2: deterministic tie rule over all 107 Part B units (4 workers, GPU 1), then the report.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_followups/phase2
mkdir -p $R/logs
$PY -m src.analysis.reliance_fidelity list | awk '$2 ~ /B/ {print $1}' > $R/units.txt
split -n r/4 -d $R/units.txt $R/logs/chunk_
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.followups_phase2 run --units $(cat $f) > $f.log 2>&1 & done
wait
$PY -u -m src.analysis.followups_phase2 report > $R/report.txt 2> $R/logs/report.err
echo "phase 2 finished $(date '+%F %T')" > $R/FINISHED
