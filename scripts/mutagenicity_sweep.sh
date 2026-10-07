#!/usr/bin/env bash
# STEP 3: GraphXAI Mutagenicity, 5 seeds, DECOUPLED protocol from the start
# (model seed S, explainer seed ES = 1000 + S). One job per seed, all five
# concurrent on GPU 1 (GPU 0 left free). Everything under runs/mutagenicity/.
#
#   screen -dmS muta_sweep bash scripts/mutagenicity_sweep.sh
#
# Each seed runs three stages against ONE explanation cache, so the cheap
# explainers are cached (and the whole scoring path exercised) ~20 min in,
# long before SubgraphX finishes:
#   1. GNNExplainer + PGExplainer (PG trained on the full train split)
#   2. SubgraphX (rollout 20, sample_num 30 -- locked)
#   3. re-score all three from the cache -> out_s<S>.json
# Re-runnable: a seed whose out_s<S>.json exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/mutagenicity
mkdir -p $R/logs
SEEDS=${SEEDS:-"0 1 2 3 4"}
stamp() { date '+%F %T'; }

unit() {
  local S=$1 ES=$((1000 + $1))
  local CK=$R/ckpt_s$S.pt CACHE=$R/cache_s$S.pt OUT=$R/out_s$S.json LOG=$R/logs/s$S.log
  local EX="--dataset mutagenicity_graphxai --ckpt $CK --seed $ES --max-nodes 80 --masking all --cache $CACHE"
  {
    echo "==== $(stamp) model-seed=$S explainer-seed=$ES START ===="
    if [[ -f $OUT ]]; then echo "  $OUT exists -- skip"; return 0; fi
    if [[ ! -f $CK ]]; then
      $PY -u -m src.train.train --dataset mutagenicity_graphxai --save $CK --seed $S \
          --depth 3 --epochs 120 --hidden 300 --val-frac 0.10 --test-frac 0.20 --device cuda
    fi
    echo "{\"model_seed\": $S, \"explainer_seed\": $ES, \"protocol\": \"decoupled\"}" > $R/seeds_s$S.json
    echo "==== $(stamp) stage 1: gnnexplainer + pgexplainer ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer --out $R/out_gp_s$S.json
    echo "==== $(stamp) stage 2: subgraphx ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers subgraphx --out $R/out_sx_s$S.json
    echo "==== $(stamp) stage 3: re-score all three from cache ===="
    $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer,subgraphx --out $OUT
    echo "==== $(stamp) model-seed=$S explainer-seed=$ES DONE ===="
  } > $LOG 2>&1
}

echo "[$(stamp)] sweep start: seeds=$SEEDS, GPU $CUDA_VISIBLE_DEVICES" >> $R/sweep.log
for S in $SEEDS; do unit $S & done
wait
echo "[$(stamp)] sweep done" >> $R/sweep.log
