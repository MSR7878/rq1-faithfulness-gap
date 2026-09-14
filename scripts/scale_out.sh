#!/usr/bin/env bash
# RQ1 scale-out sweep driver.
#   CUDA_VISIBLE_DEVICES=0 bash scripts/scale_out.sh tox21   # all 12 Tox21 endpoints
#   CUDA_VISIBLE_DEVICES=1 bash scripts/scale_out.sh bxaic   # 3 extra B-XAIC tasks
# ONLY="<space-separated names>" restricts the list -- run 2 workers per GPU:
#   ONLY="NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD" CUDA_VISIBLE_DEVICES=0 bash scripts/scale_out.sh tox21
#   ONLY="NR-PPAR-gamma SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53"  CUDA_VISIBLE_DEVICES=0 bash scripts/scale_out.sh tox21
#
# Per dataset: train a single-split D-MPNN checkpoint (skipped if runs/ckpt_<d>.pt
# exists), then run_phase3 --masking all (R3 mean+mode, R2 zero, R1 hard) on n=30.
# Re-runnable: finished datasets are skipped at the training step; delete a
# runs/ckpt_<d>.pt to force a redo.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GROUP=${1:?usage: scale_out.sh <tox21|bxaic>}
mkdir -p runs/scaleout_logs

# Progress readout across the endpoint/task loop below. bash has no tqdm;
# this reproduces its two behaviours that matter for `| tee log` inside
# screen: overwrite in place on a real TTY, one plain line per unit when
# logged/piped (so a redirected log doesn't fill up with '\r' partial lines).
IDX=0; T0=$(date +%s)
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

run_one () {
  local D=$1; shift
  progress "$D (starting)"
  echo "######## $(date '+%F %T')  $D  ########"
  if [[ ! -f runs/ckpt_$D.pt ]]; then
    $PY -u -m src.train.train --dataset "$D" --save "runs/ckpt_$D.pt" \
        --depth 3 --seed 0 --device cuda "$@" 2>&1 | tee "runs/scaleout_logs/train_$D.log"
  else
    echo "  runs/ckpt_$D.pt exists -- skip training"
  fi
  $PY -u -m src.explain.run_phase3 --dataset "$D" --masking all --seed 0 "${SWEEP_ARGS[@]}" \
      2>&1 | tee "runs/scaleout_logs/sweep_$D.log"
  IDX=$((IDX + 1)); progress "$D (done)"; echo
}

case "$GROUP" in
  tox21)
    SWEEP_ARGS=(--limit 30 --pg-train-limit 200 --stratify-frac 0.5)  # all 12 endpoints are <=16% positive
    EPS=${ONLY:-NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma \
                SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53}
    TOTAL=$(echo $EPS | wc -w)
    for EP in $EPS; do
      run_one "tox21_$EP" --epochs 80 --batch-size 128 --hidden 300
    done
    ;;
  bxaic)
    TASKS=${ONLY:-PAINS X P}   # structurally distinct from indole: alert-set / halogen / phosphorus
    TOTAL=$(echo $TASKS | wc -w)
    for T in $TASKS; do
      SWEEP_ARGS=(--limit 30 --pg-train-limit 200 --max-nodes 80)
      [[ "$T" == "P" ]] && SWEEP_ARGS+=(--stratify-frac 0.5)   # P is ~13% positive
      run_one "bxaic_$T" --epochs 25 --batch-size 512 --hidden 256
    done
    ;;
  *) echo "unknown group '$GROUP' (want tox21 | bxaic)"; exit 1 ;;
esac
[ -t 1 ] && echo
echo "ALL DONE: $GROUP  $(date '+%F %T')"
