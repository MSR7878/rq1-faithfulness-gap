"""
How common is B-XAIC/X/seed4's failure mode -- a structurally VALID
(non-degenerate) explainer mask that nonetheless scores significantly BELOW
its matched random baseline? Distinct from collapse (explainer_health flags
a literal constant/degenerate mask); this is a real explanation that is
simply worse than chance, undetectable by the degenerate-mask health check.

Scans every (unit, explainer) pair among mutag_graphxai (5 units) + all 4
B-XAIC tasks (20 units) = 25 coupled units x 3 explainers = 75 tests, using
the per-molecule arrays saved by mutag_random_baseline.py/
bxaic_random_baseline.py (this session's gap-1/2 fix) joined against each
unit's own out_*.json per_molecule GEA, via each unit's cache test_idx for
correct molecule alignment (B-XAIC's test_idx is a non-contiguous subset,
mutag_graphxai's happens to be range(188)).

Tox21 (60 units) is OUT OF SCOPE here: its per-molecule random-baseline
Fid+ values were never persisted (data gap 4, explicitly deferred -- unlike
GEA's random baselines, Tox21's needs a GPU forward pass through the
trained model, not a pure structural recompute) so a per-UNIT test isn't
possible without that recompute. Reported separately as a known gap, not
silently ignored.

    python -m src.analysis.below_random_scan
"""
from __future__ import annotations
import csv
import json
import numpy as np
import torch
from scipy.stats import wilcoxon

EXPL_RAND = [("gnnexplainer", "random_node", "GNN", "RandN"),
             ("subgraphx", "random_node", "SX", "RandN"),
             ("pgexplainer", "random_edge", "PG", "RandE")]
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"

MUTAG = {"cache": lambda s: f"{OUT}/cache_mutag_graphxai_s{s}.pt",
         "out": lambda s: f"{OUT}/out_mutag_graphxai_s{s}.json",
         "npz": f"{OUT}/baseline_arrays_mutag_graphxai.npz"}
BXAIC = {t: {"cache": lambda s, t=t: f"{OUT}/cache_bxaic_{t}_s{s}.pt",
             "out": lambda s, t=t: f"{OUT}/out_bxaic_{t}_s{s}.json",
             "npz": f"{OUT}/baseline_arrays_bxaic_{t}.npz"}
         for t in ("indole", "PAINS", "X", "P")}


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def load_collapse_flags():
    """{(dataset, task_or_endpoint, seed): collapsed_bool} for coupled units, PG only."""
    flags = {}
    with open(f"{OUT}/collapse_table.csv") as f:
        for row in csv.DictReader(f):
            if row["protocol"] != "coupled":
                continue
            flags[(row["dataset"], row["task_or_endpoint"], int(row["seed"]))] = row["collapsed"] == "True"
    return flags


def scan_unit(dataset_key: str, task: str, s: int, paths: dict, collapse_flags: dict):
    blob = torch.load(paths["cache"](s), weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    pos_by_mi = {mi: k for k, mi in enumerate(test_idx)}

    j = json.load(open(paths["out"](s)))
    pm = j["per_molecule"]["R3"]["mean"]

    npz = np.load(paths["npz"])
    mol_idx = npz[f"mol_idx_s{s}"]

    results = []
    for expl, rand_kind, esh, rsh in EXPL_RAND:
        real_rows = pm[expl]
        real_vals = np.array([real_rows[pos_by_mi[mi]]["gea"] for mi in mol_idx])
        rand_vals = npz[f"{rand_kind}_s{s}"]
        d = real_vals - rand_vals
        p = 1.0 if np.allclose(d, 0) else wilcoxon(real_vals, rand_vals, alternative="two-sided", zero_method="wilcox").pvalue
        below = bool(p < 0.05 and real_vals.mean() < rand_vals.mean())
        collapsed = collapse_flags.get((dataset_key, task, s), False) if expl == "pgexplainer" else False
        results.append({
            "dataset": dataset_key, "task": task, "seed": s, "explainer": esh,
            "random": rsh, "real_mean": real_vals.mean(), "rand_mean": rand_vals.mean(),
            "p": p, "star": star(p), "below_random_sig": below, "collapsed": collapsed,
        })
    return results


def main():
    collapse_flags = load_collapse_flags()
    all_rows = []
    for s in SEEDS:
        all_rows += scan_unit("mutag_graphxai", "", s, MUTAG, collapse_flags)
    for t, paths in BXAIC.items():
        for s in SEEDS:
            all_rows += scan_unit("bxaic", t, s, paths, collapse_flags)

    print("=" * 110)
    print(" Non-collapsed units where an explainer scores SIGNIFICANTLY BELOW its matched random baseline")
    print(" (25 coupled units x 3 explainers = 75 tests; mutag_graphxai + all 4 B-XAIC tasks; Tox21 out of scope, see docstring)")
    print("=" * 110)
    flagged = [r for r in all_rows if r["below_random_sig"] and not r["collapsed"]]
    for r in all_rows:
        tag = f"{r['dataset']}{'/' + r['task'] if r['task'] else ''}/s{r['seed']}"
        mark = "  <-- BELOW RANDOM, SIG, NOT COLLAPSED" if (r["below_random_sig"] and not r["collapsed"]) else \
               "  (collapsed, excluded from tally)" if r["collapsed"] else ""
        print(f"  {tag:<22} {r['explainer']:<4} vs {r['random']:<6} "
              f"real={r['real_mean']:.3f} rand={r['rand_mean']:.3f} p={r['p']:.2e} {r['star']:<3}{mark}")

    print(f"\n  TOTAL flagged (non-collapsed, significant, real < random): {len(flagged)}/75")
    by_expl = {}
    for r in flagged:
        by_expl.setdefault(r["explainer"], []).append(r)
    for e, rows in by_expl.items():
        print(f"    {e}: {len(rows)}  " + ", ".join(f"{r['dataset']}{'/' + r['task'] if r['task'] else ''}/s{r['seed']}" for r in rows))
    by_ds = {}
    for r in flagged:
        key = f"{r['dataset']}{'/' + r['task'] if r['task'] else ''}"
        by_ds.setdefault(key, []).append(r)
    print("\n  by dataset/task:")
    for k, rows in by_ds.items():
        print(f"    {k}: {len(rows)}  (" + ", ".join(f"{r['explainer']}/s{r['seed']}" for r in rows) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
