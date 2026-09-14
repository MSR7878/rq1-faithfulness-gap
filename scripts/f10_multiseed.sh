#!/usr/bin/env bash
# F10 multi-seed test: does PGExplainer's collapse on the 4 "collapsed" Tox21
# endpoints reproduce across model seeds, and do the 2 controls stay stable?
#
#   # one GPU, all 6 endpoints x 5 seeds:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/f10_multiseed.sh
#   # split across 2 GPUs:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/f10_multiseed.sh NR-AR NR-Aromatase NR-ER-LBD
#   CUDA_VISIBLE_DEVICES=1 bash scripts/f10_multiseed.sh SR-ARE NR-ER NR-PPAR-gamma
#
# Per (endpoint, seed): train a fresh 70/15/15 D-MPNN (seed drives BOTH split and
# init -- 5 independent replicates), then run PGExplainer only, under R3, on the
# same n=30 stratified protocol as the scale-out. Model + init are the only thing
# that varies. Outputs runs/f10ms/pg_<EP>_s<SEED>.json (explainer_health has the
# collapse signal). Aggregate with:  python -m src.analysis.f10_multiseed
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
SEEDS=${SEEDS:-0 1 2 3 4}
EPS=${*:-NR-AR NR-Aromatase NR-ER-LBD SR-ARE NR-ER NR-PPAR-gamma}
OUT=runs/f10ms
mkdir -p "$OUT" "$OUT/logs"

# Progress readout for this (endpoint, seed) sweep. bash has no tqdm; this
# mimics its two behaviours that matter for a `| tee log` inside screen:
# overwrite in place on a real TTY, one plain line per unit when logged/piped
# (so redirected output doesn't fill up with '\r'-driven partial lines).
N_EP=$(echo $EPS | wc -w); N_S=$(echo $SEEDS | wc -w)
TOTAL=$((N_EP * N_S)); IDX=0; T0=$(date +%s)
progress() {
  local now elapsed eta
  now=$(date +%s); elapsed=$((now - T0))
  eta=$([[ $IDX -gt 0 ]] && echo $((elapsed * (TOTAL - IDX) / IDX)) || echo "?")
  if [ -t 1 ]; then
    printf "\r[%d/%d] %-28s elapsed=%ds eta=%ss   " "$IDX" "$TOTAL" "$1" "$elapsed" "$eta"
  else
    printf "[%d/%d] %-28s elapsed=%ds eta=%ss\n" "$IDX" "$TOTAL" "$1" "$elapsed" "$eta"
  fi
}

for EP in $EPS; do
  D="tox21_$EP"
  for S in $SEEDS; do
    CK="$OUT/ckpt_${EP}_s${S}.pt"
    JS="$OUT/pg_${EP}_s${S}.json"
    progress "$EP seed=$S (starting)"
    echo "######## $(date '+%F %T')  $EP  seed=$S  ########"
    if [[ -f "$JS" ]]; then echo "  $JS exists -- skip"; IDX=$((IDX + 1)); progress "$EP seed=$S (skipped)"; continue; fi
    if [[ ! -f "$CK" ]]; then
      $PY -u -m src.train.train --dataset "$D" --save "$CK" \
          --depth 3 --seed "$S" --device cuda \
          --epochs 80 --batch-size 128 --hidden 300 \
          2>&1 | tee "$OUT/logs/train_${EP}_s${S}.log"
    else
      echo "  $CK exists -- skip training"
    fi
    $PY -u -m src.explain.run_phase3 --dataset "$D" --ckpt "$CK" \
        --explainers pgexplainer --masking R3 --seed "$S" \
        --limit 30 --pg-train-limit 200 --stratify-frac 0.5 \
        --cache "$OUT/cache_${EP}_s${S}.pt" --out "$JS" \
        2>&1 | tee "$OUT/logs/pg_${EP}_s${S}.log"
    IDX=$((IDX + 1)); progress "$EP seed=$S (done)"; echo
  done
done
[ -t 1 ] && echo
echo "F10 MULTISEED DONE: [$EPS] x [$SEEDS]  $(date '+%F %T')"
