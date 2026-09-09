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

run_one () {
  local D=$1; shift
  echo "######## $(date '+%F %T')  $D  ########"
  if [[ ! -f runs/ckpt_$D.pt ]]; then
    $PY -u -m src.train.train --dataset "$D" --save "runs/ckpt_$D.pt" \
        --depth 3 --seed 0 --device cuda "$@" 2>&1 | tee "runs/scaleout_logs/train_$D.log"
  else
    echo "  runs/ckpt_$D.pt exists -- skip training"
  fi
  $PY -u -m src.explain.run_phase3 --dataset "$D" --masking all --seed 0 "${SWEEP_ARGS[@]}" \
      2>&1 | tee "runs/scaleout_logs/sweep_$D.log"
  echo
}

case "$GROUP" in
  tox21)
    SWEEP_ARGS=(--limit 30 --pg-train-limit 200 --stratify-frac 0.5)  # all 12 endpoints are <=16% positive
    for EP in ${ONLY:-NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma \
                      SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53}; do
      run_one "tox21_$EP" --epochs 80 --batch-size 128 --hidden 300
    done
    ;;
  bxaic)
    for T in ${ONLY:-PAINS X P}; do   # structurally distinct from indole: alert-set / halogen / phosphorus
      SWEEP_ARGS=(--limit 30 --pg-train-limit 200 --max-nodes 80)
      [[ "$T" == "P" ]] && SWEEP_ARGS+=(--stratify-frac 0.5)   # P is ~13% positive
      run_one "bxaic_$T" --epochs 25 --batch-size 512 --hidden 256
    done
    ;;
  *) echo "unknown group '$GROUP' (want tox21 | bxaic)"; exit 1 ;;
esac
echo "ALL DONE: $GROUP  $(date '+%F %T')"
