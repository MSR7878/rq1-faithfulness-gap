#!/usr/bin/env bash
# STEP 2 cost estimate for GraphXAI Mutagenicity (GPU 1 only). Three phases,
# strictly sequential so they don't contaminate each other's timing:
#   A. one SubgraphX run, 20 test molecules (rollout 20, sample_num 30)
#   B. four concurrent SubgraphX runs, 20 test molecules each (different
#      subsamples) -> 4-way throughput
#   C. GNNExplainer + PGExplainer on 20 test molecules (PG trained on the
#      full train split) -> the non-SubgraphX cost
# Everything is written under runs/mutagenicity/timing/.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
T=runs/mutagenicity/timing
CK=$T/ckpt_s0.pt
COMMON="--dataset mutagenicity_graphxai --ckpt $CK --limit 20 --max-nodes 80 --masking all"
stamp() { date '+%F %T'; }

echo "[$(stamp)] A: single SubgraphX run" > $T/timing.log
t0=$(date +%s)
$PY -u -m src.explain.run_phase3 $COMMON --explainers subgraphx --seed 1000 \
    --cache $T/cache_A.pt --out $T/out_A.json > $T/A.log 2>&1
echo "[$(stamp)] A done: $(( $(date +%s) - t0 ))s" >> $T/timing.log

echo "[$(stamp)] B: 4 concurrent SubgraphX runs" >> $T/timing.log
t0=$(date +%s)
for s in 1001 1002 1003 1004; do
  $PY -u -m src.explain.run_phase3 $COMMON --explainers subgraphx --seed $s \
      --cache $T/cache_B$s.pt --out $T/out_B$s.json > $T/B$s.log 2>&1 &
done
wait
echo "[$(stamp)] B done: $(( $(date +%s) - t0 ))s" >> $T/timing.log

echo "[$(stamp)] C: GNNExplainer + PGExplainer" >> $T/timing.log
t0=$(date +%s)
$PY -u -m src.explain.run_phase3 $COMMON --explainers gnnexplainer,pgexplainer --seed 1000 \
    --cache $T/cache_C.pt --out $T/out_C.json > $T/C.log 2>&1
echo "[$(stamp)] C done: $(( $(date +%s) - t0 ))s" >> $T/timing.log
echo "[$(stamp)] ALL DONE" >> $T/timing.log
