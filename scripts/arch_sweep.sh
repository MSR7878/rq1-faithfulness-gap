#!/usr/bin/env bash
# Architecture checks on GraphXAI Mutagenicity: plain PyG GINE (3 x GINEConv,
# edge_dim 3, hidden 214) and GCN (3 x GCNConv, hidden 300), 5 seeds each, on
# the D-MPNN runs' own splits (--split-json), decoupled protocol (model seed S,
# explainer seed 1000 + S). GPU 1 only. Everything under runs/arch_<arch>/;
# nothing else in runs/ is written and nothing is reused from runs/arch_gin/.
#
#   bash scripts/arch_sweep.sh train     # smoke test, split export, 5 models per arch
#   bash scripts/arch_sweep.sh timing    # SubgraphX on 20 molecules per arch (seed-0 model), one at a time
#   screen -dmS arch bash scripts/arch_sweep.sh explain   # ARCHS in order, CONCURRENCY units at a time
#
# explain: three stages per unit against ONE explanation cache (GNN+PG, then
# SubgraphX, then re-score all three). Re-runnable: a unit whose out_s<S>.json
# exists is skipped, and a finished stage is read back from the cache.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
D=mutagenicity_graphxai
ARCHS=${ARCHS:-"gine gcn"}
CONCURRENCY=${CONCURRENCY:-5}
SEEDS=${SEEDS:-"0 1 2 3 4"}
EX_ARGS="--masking all --max-nodes 80"
stamp() { date '+%F %T'; }
hidden() { [[ $1 == gine ]] && echo 214 || echo 300; }

case "${1:-}" in
train)
  for A in $ARCHS; do mkdir -p runs/arch_$A/logs; done
  FIRST=runs/arch_${ARCHS%% *}/logs
  $PY -u -m src.train.test_arch_explainers $ARCHS > $FIRST/smoke_test.log 2>&1 || { echo "SMOKE TEST FAILED"; tail -5 $FIRST/smoke_test.log; exit 1; }
  tail -1 $FIRST/smoke_test.log
  for A in $ARCHS; do
    R=runs/arch_$A
    $PY -u -m src.analysis.arch_gin_splits $R $D > $R/logs/splits.log 2>&1 || { echo "split export failed ($A)"; exit 1; }
    for S in $SEEDS; do
      [[ -f $R/$D/ckpt_s$S.pt ]] && continue
      $PY -u -m src.train.train --dataset $D --save $R/$D/ckpt_s$S.pt --seed $S --arch $A --depth 3 --epochs 120 \
          --hidden $(hidden $A) --split-json $R/$D/split_s$S.json --device cuda > $R/logs/train_s$S.log 2>&1 &
    done
    wait
  done
  echo "[$(stamp)] train done"
  ;;
timing)
  for A in $ARCHS; do
    R=runs/arch_$A; T=$R/$D/timing; mkdir -p $T
    t0=$(date +%s)
    $PY -u -m src.explain.run_phase3 --dataset $D --ckpt $R/$D/ckpt_s0.pt --seed 1000 $EX_ARGS --limit 20 \
        --explainers subgraphx --cache $T/cache_sx20.pt --out $T/out_sx20.json > $T/sx20.log 2>&1
    echo "[$(stamp)] $A: SubgraphX 20 molecules in $(( $(date +%s) - t0 ))s (incl. load + scoring)" | tee -a $R/sweep.log
  done
  echo "[$(stamp)] timing done"
  ;;
explain)
  unit() {
    local A=$1 S=$2 ES=$((1000 + $2)) t0=$(date +%s)
    local R=runs/arch_$A
    local CK=$R/$D/ckpt_s$S.pt OUT=$R/$D/out_s$S.json
    local EX="--dataset $D --ckpt $CK --seed $ES $EX_ARGS --cache $R/$D/cache_s$S.pt"
    {
      echo "==== $(stamp) $A model-seed=$S explainer-seed=$ES START ===="
      if [[ -f $OUT ]]; then echo "  $OUT exists -- skip"; return 0; fi
      echo "{\"arch\": \"$A\", \"model_seed\": $S, \"explainer_seed\": $ES, \"protocol\": \"decoupled\"}" > $R/$D/seeds_s$S.json
      echo "==== $(stamp) stage 1: gnnexplainer + pgexplainer ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer --out $R/$D/out_gp_s$S.json
      echo "==== $(stamp) stage 2: subgraphx ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers subgraphx --out $R/$D/out_sx_s$S.json
      echo "==== $(stamp) stage 3: re-score all three from cache ===="
      $PY -u -m src.explain.run_phase3 $EX --explainers gnnexplainer,pgexplainer,subgraphx --out $OUT
      echo "==== $(stamp) $A model-seed=$S explainer-seed=$ES DONE wall_clock_seconds=$(( $(date +%s) - t0 )) ===="
    } >> $R/logs/unit_s$S.log 2>&1
    echo "[$(stamp)] $A seed $S finished" >> $R/sweep.log
  }
  for A in $ARCHS; do echo "[$(stamp)] explain start: archs=$ARCHS seeds=$SEEDS concurrency=$CONCURRENCY GPU $CUDA_VISIBLE_DEVICES" >> runs/arch_$A/sweep.log; done
  n=0
  for A in $ARCHS; do
    for S in $SEEDS; do
      unit $A $S &
      n=$((n + 1))
      if [ "$n" -ge "$CONCURRENCY" ]; then wait -n; n=$((n - 1)); fi
    done
  done
  wait
  for A in $ARCHS; do echo "[$(stamp)] explain done (all archs: $ARCHS)" >> runs/arch_$A/sweep.log; done
  ;;
*) echo "usage: $0 train | timing | explain"; exit 1 ;;
esac
