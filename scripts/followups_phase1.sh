#!/usr/bin/env bash
# RQ1 follow-ups, Phase 1: MUTAG-GT seeds 5-9 with EXACTLY the priority-sweep (P2) protocol of seeds 0-4:
# train --seed S (depth 3, 120 epochs, hidden 300, batch 50), explain COUPLED (--seed S), all three explainers,
# --masking all --explain-pool all. Writes only runs/rq1_followups/phase1/. GPU 1 only, 5 units concurrently.
#   screen -dmS fu1 bash scripts/followups_phase1.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_followups/phase1
mkdir -p $R/logs
stamp() { date '+%F %T'; }
unit() {
  local S=$1 t0=$(date +%s)
  {
    echo "==== $(stamp)  mutag_graphxai seed=$S (coupled: explainer seed = $S) ===="
    [[ -f $R/ckpt_mutag_graphxai_s$S.pt ]] || $PY -u -m src.train.train --dataset mutag_graphxai --save $R/ckpt_mutag_graphxai_s$S.pt \
        --seed $S --device cuda --depth 3 --epochs 120 --hidden 300
    $PY -u -m src.explain.run_phase3 --dataset mutag_graphxai --ckpt $R/ckpt_mutag_graphxai_s$S.pt --seed $S \
        --explainers gnnexplainer,pgexplainer,subgraphx --masking all --explain-pool all \
        --cache $R/cache_mutag_graphxai_s$S.pt --out $R/out_mutag_graphxai_s$S.json
    echo "==== $(stamp) seed=$S DONE wall_clock_seconds=$(( $(date +%s) - t0 )) ===="
  } > $R/logs/mutag_graphxai_s$S.log 2>&1
  echo "[$(stamp)] seed $S finished" >> $R/run.log
}
echo "[$(stamp)] start seeds 5-9" >> $R/run.log
for S in 5 6 7 8 9; do unit $S & done
wait
echo "[$(stamp)] explain done" >> $R/run.log
