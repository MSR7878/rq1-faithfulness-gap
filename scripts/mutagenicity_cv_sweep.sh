#!/usr/bin/env bash
# Full-set GraphXAI Mutagenicity, explained out-of-sample by 5-fold CV.
# Fold f: model trained on the other four folds (model seed f), every graph in
# fold f explained with explainer seed 1000 + f. All five folds concurrently on
# GPU 1. Writes ONLY under runs/mutagenicity_cv/.
#
#   screen -dmS muta_cv bash scripts/mutagenicity_cv_sweep.sh
#
# Three stages per fold against one explanation cache (GNN+PG, then SubgraphX,
# then re-score). Re-runnable: a fold whose out_s<f>.json exists is skipped,
# and a fold that crashed during SubgraphX keeps its cached GNN/PG explanations.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/mutagenicity_cv
mkdir -p $R/logs
FOLDS=${FOLDS:-"0 1 2 3 4"}
stamp() { date '+%F %T'; }

$PY -m src.analysis.mutagenicity_cv_splits > $R/splits.log 2>&1

unit() {
  local S=$1 ES=$((1000 + $1))
  local CK=$R/ckpt_s$S.pt CACHE=$R/cache_s$S.pt OUT=$R/out_s$S.json LOG=$R/logs/f$S.log
  local EX="--dataset mutagenicity_graphxai --ckpt $CK --seed $ES --max-nodes 80 --masking all --cache $CACHE"
  {
    local t0=$(date +%s)
    echo "==== $(stamp) fold=$S model-seed=$S explainer-seed=$ES START ===="
    if [[ -f $OUT ]]; then echo "  $OUT exists -- skip"; return 0; fi
    if [[ ! -f $CK ]]; then
      $PY -u -m src.train.train --dataset mutagenicity_graphxai --save $CK --seed $S \
          --depth 3 --epochs 120 --hidden 300 --split-json $R/split_f$S.json --device cuda
    fi
    echo "{\"fold\": $S, \"model_seed\": $S, \"explainer_seed\": $ES, \"protocol\": \"decoupled\"}" > $R/seeds_s$S.json
    echo "==== $(stamp) stage 1: gnnexplainer + pgexplainer ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer --out $R/out_gp_s$S.json
    echo "==== $(stamp) stage 2: subgraphx ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers subgraphx --out $R/out_sx_s$S.json
    echo "==== $(stamp) stage 3: re-score all three from cache ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer,subgraphx --out $OUT
    echo "==== $(stamp) fold=$S DONE wall_clock_seconds=$(( $(date +%s) - t0 )) ===="
  } > $LOG 2>&1
}

echo "[$(stamp)] cv sweep start: folds=$FOLDS, GPU $CUDA_VISIBLE_DEVICES" >> $R/sweep.log
for S in $FOLDS; do unit $S & done
wait
echo "[$(stamp)] cv sweep done" >> $R/sweep.log
