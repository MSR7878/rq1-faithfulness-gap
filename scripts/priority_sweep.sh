#!/usr/bin/env bash
# Priority-ordered scale-up sweep (revised scope), C-way concurrent workers
# on ONE GPU (default GPU 1, leaves GPU 0 free for other users).
#
#   CONCURRENCY=4 bash scripts/priority_sweep.sh p2   # confirm cost first (~3h)
#   CONCURRENCY=4 bash scripts/priority_sweep.sh p1   # headline finding (~57h)
#   CONCURRENCY=4 bash scripts/priority_sweep.sh p3   # cheap wrap-up (~3h)
#   CONCURRENCY=4 bash scripts/priority_sweep.sh all  # p2 then p1 then p3
#
# P1 -- B-XAIC, 4 tasks x 5 seeds, N=400 per (task,seed) stratified to 90%
#       positive (360 pos + 40 neg target) so GEA gets a few hundred
#       positives per seed instead of 13. max-nodes=80 (existing).
# P2 -- MUTAG-GraphXAI, 5 seeds, --explain-pool all: the FULL 188 graphs, not
#       just the 29-molecule test split (all 188 have NO2/NH2 GT; no large-
#       molecule tail, max N=28 -- confirmed cheap before committing to P1).
# P3 -- Tox21, 12 endpoints x 5 seeds, same n=30 stratified protocol as the
#       original scale-out (no GT, F12 already shows R3 separates nothing on
#       12/12 -- this is a seed-robustness check, not a new sample size), now
#       with --max-nodes 60 (new; see the exclusion-% note this run records).
#
# Re-runnable: a unit is skipped if its runs/prio/out_*.json already exists.
set -u
cd "$(dirname "$0")/.."
CONCURRENCY=${CONCURRENCY:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
WHAT=${1:-all}
SEEDS="0 1 2 3 4"
OUT=runs/prio
mkdir -p "$OUT/logs"
JOBS=$(mktemp)
trap 'rm -f "$JOBS"' EXIT

note() { echo "[$(date '+%F %T')] $*"; }

add_job() {  # D  train_args  expl_args
  local d=$1 targs=$2 eargs=$3
  for s in $SEEDS; do
    local ck="$OUT/ckpt_${d}_s${s}.pt" out="$OUT/out_${d}_s${s}.json"
    local cache="$OUT/cache_${d}_s${s}.pt" log="$OUT/logs/${d}_s${s}.log"
    # ES = explainer/PGExplainer-init seed, decoupled from S (model seed) --
    # a large fixed offset keeps it deterministic/reproducible while never
    # colliding with the model-seed range. See F10-MECHANISM.
    local es=$((s + 1000))
    echo "D=$d CK=$ck S=$s ES=$es TRAIN_ARGS='$targs' EXPL_ARGS='$eargs' CACHEF=$cache OUTF=$out LOGF=$log bash scripts/_run_unit.sh" >> "$JOBS"
  done
}

build_p1() {
  local targs="--depth 3 --epochs 25 --batch-size 512 --hidden 256"
  local eargs="--explainers gnnexplainer,pgexplainer,subgraphx --masking all --limit 400 --stratify-frac 0.9 --max-nodes 80 --pg-train-limit 200"
  for t in indole PAINS X P; do add_job "bxaic_$t" "$targs" "$eargs"; done
}
build_p2() {
  local targs="--depth 3 --epochs 120 --hidden 300"
  local eargs="--explainers gnnexplainer,pgexplainer,subgraphx --masking all --explain-pool all"
  add_job "mutag_graphxai" "$targs" "$eargs"
}
build_p3() {
  local targs="--depth 3 --epochs 80 --batch-size 128 --hidden 300"
  local eargs="--explainers gnnexplainer,pgexplainer,subgraphx --masking all --limit 30 --pg-train-limit 200 --stratify-frac 0.5 --max-nodes 60"
  for ep in NR-AR NR-AR-LBD NR-AhR NR-Aromatase NR-ER NR-ER-LBD NR-PPAR-gamma \
            SR-ARE SR-ATAD5 SR-HSE SR-MMP SR-p53; do
    add_job "tox21_$ep" "$targs" "$eargs"
  done
}

# Progress readout: overwrite in place on a real TTY, one plain line per
# completed job when output is redirected/teed inside screen (same TTY-vs-log
# split as tqdm's mininterval, just implemented in bash -- see run_phase3.py).
run_queue() {
  local total done pids=() newp
  total=$(wc -l < "$JOBS"); done=0
  local t0; t0=$(date +%s)
  note "queue: $total jobs, concurrency=$CONCURRENCY"
  if [[ "${DRY:-0}" == "1" ]]; then cat "$JOBS"; return 0; fi
  while IFS= read -r CMD; do
    bash -c "$CMD" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge "$CONCURRENCY" ]; then
      wait -n
      newp=(); for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && newp+=("$p"); done
      done=$((done + (${#pids[@]} - ${#newp[@]})))
      pids=("${newp[@]}")
      local now elapsed eta
      now=$(date +%s); elapsed=$((now - t0))
      eta=$([[ $done -gt 0 ]] && echo $((elapsed * (total - done) / done)) || echo "?")
      if [ -t 1 ]; then
        printf "\r[%d/%d] elapsed=%ds eta=%ss   " "$done" "$total" "$elapsed" "$eta"
      else
        printf "[%d/%d] elapsed=%ds eta=%ss\n" "$done" "$total" "$elapsed" "$eta"
      fi
    fi
  done < "$JOBS"
  wait
  [ -t 1 ] && echo
  note "queue done: $total/$total"
}

run_tier() {
  local tier=$1 builder=$2
  : > "$JOBS"
  $builder
  note "===== $tier: $(wc -l < "$JOBS") units, concurrency=$CONCURRENCY, GPU $CUDA_VISIBLE_DEVICES ====="
  run_queue
}

case "$WHAT" in
  p1) run_tier P1 build_p1 ;;
  p2) run_tier P2 build_p2 ;;
  p3) run_tier P3 build_p3 ;;
  all) run_tier P2 build_p2; run_tier P1 build_p1; run_tier P3 build_p3 ;;
  *) echo "unknown '$WHAT' (want p1 | p2 | p3 | all)"; exit 1 ;;
esac
note "PRIORITY SWEEP DONE: $WHAT"
