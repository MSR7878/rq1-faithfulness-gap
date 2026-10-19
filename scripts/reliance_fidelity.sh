#!/usr/bin/env bash
# RQ1 Part A (GT reliance) + Part B (logit / random-corrected fidelity) on existing models and caches.
# GPU 1 only. Writes only under runs/rq1_reliance_fidelity/. Ends with the report and a DONE file.
# Order (addendum of 2026-10-19): Part A (kept if already computed) -> validation pass over ALL molecules of
# ALL Part B units -> stop unless every mismatch is a verified boundary tie -> Part B -> tie check -> report.
#   screen -dmS rf bash scripts/reliance_fidelity.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.conda/envs/rq1/bin/python}
export CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R=runs/rq1_reliance_fidelity
W=${WORKERS:-4}
mkdir -p $R/logs
stamp() { date '+%F %T'; }
fanout() {   # fanout <tag> <cmd...> : split the unit list $R/units_<tag>.txt over W workers
  local tag=$1; shift
  rm -f $R/logs/chunk_${tag}_*
  split -n r/$W -d $R/units_$tag.txt $R/logs/chunk_${tag}_
  for f in $R/logs/chunk_${tag}_??; do "$@" --units $(cat $f) > $f.log 2>&1 & done
  wait
}
echo "[$(stamp)] start (workers=$W)" >> $R/run.log

# Part A (no stored fidelity involved; existing unit files are kept)
$PY -m src.analysis.reliance_fidelity list | awk '$2 ~ /A/ {print $1}' > $R/units_A.txt
fanout A $PY -u -m src.analysis.reliance_fidelity run --part A
echo "[$(stamp)] part A: $(ls $R/partA/*.npz | wc -l)/$(wc -l < $R/units_A.txt) units" >> $R/run.log

# validation pass over all Part B units
$PY -m src.analysis.reliance_fidelity list | awk '$2 ~ /B/ {print $1}' > $R/units_B.txt
cp $R/units_B.txt $R/units_V.txt          # the validation pass covers exactly the Part B units
fanout V $PY -u -m src.analysis.reliance_fidelity validate
$PY - <<'EOF' >> $R/run.log
import csv, glob, json, sys
R = "runs/rq1_reliance_fidelity"
units = open(f"{R}/units_B.txt").read().split()
V = {u: json.load(open(f"{R}/validation/{u}.json")) for u in units if glob.glob(f"{R}/validation/{u}.json")}
rows = [r for v in V.values() for r in v["mismatches"]]
with open(f"{R}/validation/mismatch_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["unit", "explainer", "position", "masking", "metric", "stored", "recomputed", "abs_dev",
                                      "tie_at_boundary", "tied_value", "n_tied_inside", "n_tied_outside"])
    w.writeheader(); w.writerows(rows)
missing = [u for u in units if u not in V]
stopped = [u for u, v in V.items() if v["status"] != "ok"]
untied = [r for r in rows if not r["tie_at_boundary"]]
ok = not missing and not stopped and not untied
print(f"validation: {len(V)}/{len(units)} units, {len(rows)} mismatches, untied {len(untied)}, stopped {stopped}, missing {missing} -> {'PASS' if ok else 'FAIL'}")
if not ok:
    open(f"{R}/STOPPED", "w").write("validation failed; see run.log and validation/\n"); sys.exit(1)
EOF
if [[ -f $R/STOPPED ]]; then echo "[$(stamp)] STOPPED after validation" >> $R/run.log; exit 1; fi

# Part B from scratch
fanout B $PY -u -m src.analysis.reliance_fidelity run --part B
echo "[$(stamp)] part B: $(ls $R/partB/*.npz | wc -l)/$(wc -l < $R/units_B.txt) units; worker errors: $(grep -l Traceback $R/logs/chunk_B_*.log 2>/dev/null | wc -l)" >> $R/run.log
if grep -q Traceback $R/logs/chunk_B_*.log 2>/dev/null; then echo "stopped in Part B" > $R/STOPPED; exit 1; fi

$PY -u -m src.analysis.reliance_fidelity_ties > $R/logs/ties.log 2>&1
echo "[$(stamp)] tie check: $(grep -c molecules $R/logs/ties.log) cells" >> $R/run.log
$PY -u -m src.analysis.reliance_fidelity_report > $R/report.txt 2> $R/logs/report.err
echo "[$(stamp)] report done (stderr lines: $(wc -l < $R/logs/report.err))" >> $R/run.log
echo "finished $(stamp); report: $R/report.txt" > $R/DONE
