#!/usr/bin/env bash
# Re-run a specific list of already-checkpointed dataset keys through
# run_phase3 --masking all (recovery after a crash / a metric fix).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/rerun.sh tox21_NR-ER-LBD tox21_SR-ARE
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
mkdir -p runs/scaleout_logs
for D in "$@"; do
  echo "######## $(date '+%F %T')  rerun $D  ########"
  args=(--limit 30 --pg-train-limit 200)
  case "$D" in
    tox21_*)  args+=(--stratify-frac 0.5) ;;
    bxaic_P)  args+=(--max-nodes 80 --stratify-frac 0.5) ;;
    bxaic_*)  args+=(--max-nodes 80) ;;
  esac
  [[ -f runs/ckpt_$D.pt ]] || { echo "  MISSING runs/ckpt_$D.pt -- skip"; continue; }
  "$PY" -u -m src.explain.run_phase3 --dataset "$D" --masking all --seed 0 "${args[@]}" \
     2>&1 | tee "runs/scaleout_logs/rerun_$D.log"
  echo
done
echo "RERUN DONE: $*  $(date '+%F %T')"
