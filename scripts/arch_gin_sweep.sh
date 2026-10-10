#!/usr/bin/env bash
# Architecture check: plain PyG GIN (3 x GINConv, hidden 212, sum pooling) on
# MUTAG-GT and GraphXAI Mutagenicity, 5 seeds each, on the D-MPNN runs' own
# splits (--split-json) and the decoupled protocol (model seed S, explainer
# seed 1000 + S). GPU 1 only. Everything under runs/arch_gin/; nothing else in
# runs/ is written.
#
#   bash scripts/arch_gin_sweep.sh train     # smoke test, 10 models, MUTAG-GT 10-fold CV
#   bash scripts/arch_gin_sweep.sh timing    # SubgraphX on 20 molecules per dataset (seed-0 model)
#   screen -dmS arch_gin bash scripts/arch_gin_sweep.sh explain   # 10 units, CONCURRENCY at a time
#
# explain: three stages per unit against ONE explanation cache (GNN+PG, then
# SubgraphX, then re-score all three). Re-runnable: a unit whose out_s<S>.json
# exists is skipped, and a finished stage is read back from the cache.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/arch_gin
CONCURRENCY=${CONCURRENCY:-5}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-"mutag_graphxai mutagenicity_graphxai"}
TRAIN="--arch gin --depth 3 --epochs 120 --hidden 212"       # batch 50, lr 1e-3: train.py defaults, as the D-MPNN
stamp() { date '+%F %T'; }
eargs() { [[ $1 == mutag_graphxai ]] && echo "--masking all --explain-pool all" || echo "--masking all --max-nodes 80"; }

case "${1:-}" in
train)
  mkdir -p $R/logs
  $PY -u -m src.train.test_gin_explainers > $R/logs/smoke_test.log 2>&1 || { echo "SMOKE TEST FAILED"; tail -5 $R/logs/smoke_test.log; exit 1; }
  tail -1 $R/logs/smoke_test.log
  $PY -u -m src.analysis.arch_gin_splits > $R/logs/splits.log 2>&1 || { echo "split export failed"; exit 1; }
  for D in $DATASETS; do
    for S in $SEEDS; do
      [[ -f $R/$D/ckpt_s$S.pt ]] && continue
      $PY -u -m src.train.train --dataset $D --save $R/$D/ckpt_s$S.pt --seed $S $TRAIN \
          --split-json $R/$D/split_s$S.json --device cuda > $R/logs/train_${D}_s$S.log 2>&1 &
    done
    wait
  done
  # accuracy bar for MUTAG-GT: 10-fold CV, same protocol as the D-MPNN's Phase-2 table (seed 0)
  $PY -u -m src.train.train --dataset mutag_graphxai --folds 10 --seed 0 $TRAIN --device cuda \
      > $R/logs/cv10_mutag_graphxai.log 2>&1
  echo "[$(stamp)] train done"
  ;;
timing)
  for D in $DATASETS; do
    T=$R/$D/timing; mkdir -p $T
    t0=$(date +%s)
    $PY -u -m src.explain.run_phase3 --dataset $D --ckpt $R/$D/ckpt_s0.pt --seed 1000 $(eargs $D) --limit 20 \
        --explainers subgraphx --cache $T/cache_sx20.pt --out $T/out_sx20.json > $T/sx20.log 2>&1
    echo "[$(stamp)] $D: SubgraphX 20 molecules in $(( $(date +%s) - t0 ))s (incl. load + scoring)" | tee -a $R/logs/timing.log
  done
  ;;
explain)
  unit() {
    local D=$1 S=$2 ES=$((1000 + $2)) t0=$(date +%s)
    local CK=$R/$D/ckpt_s$S.pt OUT=$R/$D/out_s$S.json
    local EX="--dataset $D --ckpt $CK --seed $ES $(eargs $D) --cache $R/$D/cache_s$S.pt"
    {
      echo "==== $(stamp) $D model-seed=$S explainer-seed=$ES START ===="
      if [[ -f $OUT ]]; then echo "  $OUT exists -- skip"; return 0; fi
      echo "{\"arch\": \"gin\", \"model_seed\": $S, \"explainer_seed\": $ES, \"protocol\": \"decoupled\"}" > $R/$D/seeds_s$S.json
      echo "==== $(stamp) stage 1: gnnexplainer + pgexplainer ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer --out $R/$D/out_gp_s$S.json
      echo "==== $(stamp) stage 2: subgraphx ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers subgraphx --out $R/$D/out_sx_s$S.json
      echo "==== $(stamp) stage 3: re-score all three from cache ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer,subgraphx --out $OUT
      echo "==== $(stamp) $D model-seed=$S explainer-seed=$ES DONE wall_clock_seconds=$(( $(date +%s) - t0 )) ===="
    } >> $R/logs/${D}_s$S.log 2>&1
  }
  mkdir -p $R/logs
  echo "[$(stamp)] explain start: datasets=$DATASETS seeds=$SEEDS concurrency=$CONCURRENCY GPU $CUDA_VISIBLE_DEVICES" >> $R/sweep.log
  n=0
  for D in $DATASETS; do
    for S in $SEEDS; do
      unit $D $S &
      n=$((n + 1))
      if [ "$n" -ge "$CONCURRENCY" ]; then wait -n; n=$((n - 1)); fi
    done
  done
  wait
  echo "[$(stamp)] explain done" >> $R/sweep.log
  ;;
*) echo "usage: $0 train | timing | explain"; exit 1 ;;
esac
