#!/usr/bin/env bash
# One (dataset, seed) work unit: train a fresh ckpt if missing, then run
# run_phase3's explain+score. Called by scripts/priority_sweep.sh's queue,
# never directly. Env vars (all required): D CK S TRAIN_ARGS EXPL_ARGS
# CACHEF OUTF LOGF
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
mkdir -p "$(dirname "$LOGF")"
{
  echo "==== $(date '+%F %T')  $D seed=$S  ===="
  if [[ -f "$OUTF" ]]; then
    echo "  $OUTF exists -- skip (unit already done)"
  else
    if [[ ! -f "$CK" ]]; then
      # shellcheck disable=SC2086
      "$PY" -u -m src.train.train --dataset "$D" --save "$CK" --seed "$S" --device cuda $TRAIN_ARGS
    else
      echo "  $CK exists -- skip training"
    fi
    # shellcheck disable=SC2086
    "$PY" -u -m src.explain.run_phase3 --dataset "$D" --ckpt "$CK" --seed "$S" \
        --cache "$CACHEF" --out "$OUTF" $EXPL_ARGS
  fi
  echo "==== $(date '+%F %T')  $D seed=$S  DONE ===="
} > "$LOGF" 2>&1
