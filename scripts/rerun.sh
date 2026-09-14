#!/usr/bin/env bash
# Re-run a specific list of already-checkpointed dataset keys through
# run_phase3 --masking all (recovery after a crash / a metric fix).
#   CUDA_VISIBLE_DEVICES=0 bash scripts/rerun.sh tox21_NR-ER-LBD tox21_SR-ARE
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
mkdir -p runs/scaleout_logs
TOTAL=$#; IDX=0; T0=$(date +%s)
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
for D in "$@"; do
  progress "$D (starting)"
  echo "######## $(date '+%F %T')  rerun $D  ########"
  args=(--limit 30 --pg-train-limit 200)
  case "$D" in
    tox21_*)  args+=(--stratify-frac 0.5) ;;
    bxaic_P)  args+=(--max-nodes 80 --stratify-frac 0.5) ;;
    bxaic_*)  args+=(--max-nodes 80) ;;
  esac
  [[ -f runs/ckpt_$D.pt ]] || { echo "  MISSING runs/ckpt_$D.pt -- skip"; IDX=$((IDX + 1)); continue; }
  "$PY" -u -m src.explain.run_phase3 --dataset "$D" --masking all --seed 0 "${args[@]}" \
     2>&1 | tee "runs/scaleout_logs/rerun_$D.log"
  IDX=$((IDX + 1)); progress "$D (done)"; echo
done
[ -t 1 ] && echo
echo "RERUN DONE: $*  $(date '+%F %T')"
