#!/usr/bin/env bash
# One-off: re-run mutag_graphxai's P2 explain step under the DECOUPLED
# protocol -- reuses the EXISTING trained checkpoints (model-seed S is
# unchanged; training is not the confound under test) but assigns an
# explainer-init seed ES = S + 1000, independent of S, instead of the
# original sweep's shared value. Writes to runs/decoupled/, leaving the
# coupled-sweep outputs in runs/prio/ untouched for comparison.
#
#   CONCURRENCY=4 bash scripts/decoupled_mutag_check.sh
set -u
cd "$(dirname "$0")/.."
CONCURRENCY=${CONCURRENCY:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
OUT=runs/decoupled
mkdir -p "$OUT/logs"
TARGS="--depth 3 --epochs 120 --hidden 300"
EARGS="--explainers gnnexplainer,pgexplainer,subgraphx --masking all --explain-pool all"

note() { echo "[$(date '+%F %T')] $*"; }

JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT
for s in 0 1 2 3 4; do
  es=$((s + 1000))
  echo "D=mutag_graphxai CK=runs/prio/ckpt_mutag_graphxai_s${s}.pt S=$s ES=$es TRAIN_ARGS='$TARGS' EXPL_ARGS='$EARGS' CACHEF=$OUT/cache_mutag_graphxai_s${s}.pt OUTF=$OUT/out_mutag_graphxai_s${s}.json LOGF=$OUT/logs/mutag_graphxai_s${s}.log bash scripts/_run_unit.sh" >> "$JOBS"
done

note "queue: 5 decoupled mutag_graphxai units, concurrency=$CONCURRENCY, GPU $CUDA_VISIBLE_DEVICES"
pids=()
t0=$(date +%s)
while IFS= read -r CMD; do
  bash -c "$CMD" &
  pids+=("$!")
  if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
    wait -n
    newp=(); for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && newp+=("$p"); done
    pids=("${newp[@]}")
  fi
done < "$JOBS"
wait
note "decoupled mutag_graphxai check: 5/5 units done, elapsed $(( $(date +%s) - t0 ))s"
