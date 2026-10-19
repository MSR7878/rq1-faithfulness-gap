#!/usr/bin/env bash
# RQ1 follow-ups, Phase 2b: random-tie (rule b) re-scoring of stored top-k SubgraphX/PGExplainer results. GPU 1.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_followups/phase2b
mkdir -p $R/logs
$PY -m src.analysis.followups_phase2b list > $R/units.txt
split -n r/4 -d $R/units.txt $R/logs/chunk_
$PY -u -m src.analysis.followups_phase2b gea > $R/logs/gea.log 2>&1 &
for f in $R/logs/chunk_??; do $PY -u -m src.analysis.followups_phase2b fid --units $(cat $f) > $f.log 2>&1 & done
wait
if grep -q "Traceback\|STOP" $R/logs/*.log; then echo "errors; see logs" > $R/STOPPED; exit 1; fi
$PY -u -m src.analysis.followups_phase2b report > $R/report.txt 2> $R/logs/report.err
echo "phase 2b finished $(date '+%F %T')" > $R/FINISHED
