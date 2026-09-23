#!/usr/bin/env bash
# One (dataset, seed) work unit: train a fresh ckpt if missing, then run
# run_phase3's explain+score. Called by scripts/priority_sweep.sh's queue,
# never directly. Env vars (all required): D CK S TRAIN_ARGS EXPL_ARGS
# CACHEF OUTF LOGF
# ES (optional): the explainer/PGExplainer-MLP-init seed, DECOUPLED from S
# (the model-training seed). Defaults to S for back-compat if unset -- but
# every caller should set ES explicitly (priority_sweep.sh does, ES=S+1000)
# so the base model and PGExplainer's own init/training stochasticity never
# share a seed value. See F10-MECHANISM in configs/rq1_metric_spec.md: model
# seed x MLP-init seed interact, and coupling them into one value was a
# confound in the existing (already-completed) 5-seed priority sweep.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
ES=${ES:-$S}
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
mkdir -p "$(dirname "$LOGF")"
{
  echo "==== $(date '+%F %T')  $D model-seed=$S explainer-seed=$ES  ===="
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
    "$PY" -u -m src.explain.run_phase3 --dataset "$D" --ckpt "$CK" --seed "$ES" \
        --cache "$CACHEF" --out "$OUTF" $EXPL_ARGS
  fi
  echo "==== $(date '+%F %T')  $D model-seed=$S explainer-seed=$ES  DONE ===="
} > "$LOGF" 2>&1
