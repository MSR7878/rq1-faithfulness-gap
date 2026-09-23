#!/usr/bin/env bash
# F10 mechanism test: decouple the base-model seed from the PGExplainer
# explain-time seed (which drives torch.manual_seed for the edge-mask MLP's
# init + training stochasticity, and the molecule/pg-train subsample RNG).
#
#   Experiment A (fix model, vary explain-seed): ckpt FIXED at a clean
#     (non-collapsing) train-seed for that endpoint; --seed swept 0-4.
#     If collapse still tracks a specific --seed value regardless of which
#     model is loaded -> the mechanism is in PGExplainer's own init/training
#     path, not the base model.
#   Experiment B (fix explain-seed=4, vary model): --seed FIXED at 4 (the
#     worst offender, 9/12 endpoints); ckpt swept across that endpoint's
#     5 differently-trained models. If collapse still happens regardless of
#     which model is loaded -> confirms seed 4 itself (not a specific model)
#     drives it.
#
# Two endpoints for a little replication: NR-AhR (collapses at train-seeds
# 1,2,4 in the original sweep) and SR-MMP (collapses at 0,1,4).
#
#   CONCURRENCY=4 bash scripts/f10_mechanism.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
CONCURRENCY=${CONCURRENCY:-4}
OUT=runs/f10mech
mkdir -p "$OUT/logs"
JOBS=$(mktemp)

EARGS="--explainers pgexplainer --masking R3 --limit 30 --pg-train-limit 200 --stratify-frac 0.5 --max-nodes 60"

add() {  # ep, ckpt_seed (model), explain_seed
  local ep=$1 mseed=$2 eseed=$3
  local ck="runs/prio/ckpt_tox21_${ep}_s${mseed}.pt"
  local tag="${ep}_model${mseed}_expl${eseed}"
  local out="$OUT/out_${tag}.json" cache="$OUT/cache_${tag}.pt" log="$OUT/logs/${tag}.log"
  echo "$PY -u -m src.explain.run_phase3 --dataset tox21_$ep --ckpt $ck --seed $eseed $EARGS --cache $cache --out $out > $log 2>&1" >> "$JOBS"
}

# Experiment A: NR-AhR model fixed at its clean seed 0, explain-seed swept 0-4
for ES in 0 1 2 3 4; do add NR-AhR 0 "$ES"; done
# Experiment A: SR-MMP model fixed at its clean seed 2, explain-seed swept 0-4
for ES in 0 1 2 3 4; do add SR-MMP 2 "$ES"; done
# Experiment B: NR-AhR explain-seed fixed at 4 (worst offender), model swept 0-4
for MS in 0 1 2 3; do add NR-AhR "$MS" 4; done   # model4/expl4 == the original run, skip (reuse it)
# Experiment B: SR-MMP explain-seed fixed at 4, model swept 0-4
for MS in 0 1 2 3; do add SR-MMP "$MS" 4; done

run_queue() {
  local total done pids=() newp
  total=$(wc -l < "$JOBS"); done=0
  echo "queue: $total jobs, concurrency=$CONCURRENCY"
  while IFS= read -r CMD; do
    bash -c "$CMD" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
      wait -n
      newp=(); for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && newp+=("$p"); done
      done=$((done + (${#pids[@]} - ${#newp[@]})))
      pids=("${newp[@]}")
      echo "[$done/$total] done"
    fi
  done < "$JOBS"
  wait
  echo "queue done: $total/$total"
}
run_queue
rm -f "$JOBS"
echo "F10 MECHANISM TEST DONE"
