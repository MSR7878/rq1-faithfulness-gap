"""
Consolidated PGExplainer collapse table across every completed unit -- the
per-unit collapse status (explainer_health.pgexplainer) has existed all
along, scattered across 90 separate out_*.json files (85 coupled-protocol
units in runs/prio/ + 5 decoupled-protocol units in runs/decoupled/), never
assembled into one table. This just parses what's already on disk -- no new
runs, no model/GPU touch.

One row per unit: dataset, task_or_endpoint, seed, protocol, n_empty,
n_whole, n_degenerate, n, collapsed (n_degenerate/n >= 0.5, the threshold
used everywhere else in this project -- F10/F16/F17).

    python -m src.analysis.build_collapse_table
"""
from __future__ import annotations
import csv
import glob
import json
import os
import re

ROWS_OUT = "runs/prio/collapse_table.csv"

# (dataset, task_or_endpoint, seed) parser per filename pattern
PATTERNS = [
    (re.compile(r"^out_mutag_graphxai_s(\d+)\.json$"), lambda m: ("mutag_graphxai", "", int(m.group(1)))),
    (re.compile(r"^out_bxaic_(.+)_s(\d+)\.json$"), lambda m: ("bxaic", m.group(1), int(m.group(2)))),
    (re.compile(r"^out_tox21_(.+)_s(\d+)\.json$"), lambda m: ("tox21", m.group(1), int(m.group(2)))),
]


def parse_name(basename: str):
    for pat, fn in PATTERNS:
        m = pat.match(basename)
        if m:
            return fn(m)
    return None


def rows_from_dir(directory: str, protocol: str):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "out_*.json"))):
        basename = os.path.basename(path)
        parsed = parse_name(basename)
        if parsed is None:
            print(f"  SKIP (unrecognized filename): {path}")
            continue
        dataset, task_or_endpoint, seed = parsed
        j = json.load(open(path))
        h = j.get("explainer_health", {}).get("pgexplainer", {})
        n = h.get("n", 0)
        n_empty = h.get("n_empty", 0)
        n_whole = h.get("n_whole", 0)
        n_degenerate = h.get("n_degenerate", 0)
        collapsed = (n_degenerate / n >= 0.5) if n else False
        rows.append({
            "dataset": dataset, "task_or_endpoint": task_or_endpoint, "seed": seed,
            "protocol": protocol, "n_empty": n_empty, "n_whole": n_whole,
            "n_degenerate": n_degenerate, "n": n, "collapsed": collapsed,
        })
    return rows


def main() -> int:
    rows = rows_from_dir("runs/prio", "coupled") + rows_from_dir("runs/decoupled", "decoupled")
    rows.sort(key=lambda r: (r["dataset"], r["task_or_endpoint"], r["protocol"], r["seed"]))

    os.makedirs(os.path.dirname(ROWS_OUT), exist_ok=True)
    with open(ROWS_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "task_or_endpoint", "seed", "protocol",
                                          "n_empty", "n_whole", "n_degenerate", "n", "collapsed"])
        w.writeheader()
        w.writerows(rows)

    n_total = len(rows)
    n_collapsed = sum(1 for r in rows if r["collapsed"])
    print(f"wrote {n_total} rows -> {ROWS_OUT}")
    print(f"total collapsed: {n_collapsed}/{n_total}")
    for ds in ("mutag_graphxai", "bxaic", "tox21"):
        sub = [r for r in rows if r["dataset"] == ds]
        sub_c = [r for r in sub if r["collapsed"]]
        labels = [f"{r['task_or_endpoint'] or r['dataset']}/s{r['seed']}/{r['protocol']}" for r in sub_c]
        suffix = f"  ({', '.join(labels)})" if labels else ""
        print(f"  {ds:<16} {len(sub_c)}/{len(sub)} collapsed{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
