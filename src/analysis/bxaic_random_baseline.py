"""
Random baseline for B-XAIC GEA (all 4 tasks), matched top-k=0.25 budget,
same construction as mutag_random_baseline.py: random-node (uniform k-of-N
nodes) and random-edge (uniform per-edge score -> node_importance_from_edges,
PGExplainer's own fold -> top-k on the result). Processes WHATEVER P1
priority-sweep (task, seed) units have completed so far (globs
runs/prio/cache_bxaic_<task>_s*.pt) -- does not wait for the full sweep.

The question this settles: does SubgraphX beat random on B-XAIC? F3/F9's
"inversion" reading (SX wins because its objective matches the model's
actual decision rule) requires it to. If SX does not clear random here
either, the more likely explanation is that B-XAIC's substructure GT is
trivially recoverable (GraphXAI's own flagged pitfall) rather than SX
uniquely capturing model-rule alignment, and F3/F9 need rewriting.

    python -m src.analysis.bxaic_random_baseline
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..explain.common import node_importance_from_edges

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
ALL_E = EXPL + ["random_node", "random_edge"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX",
      "random_node": "RandN", "random_edge": "RandE"}
TASKS = ["indole", "PAINS", "X", "P"]
OUT = "runs/prio"


def jaccard(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt, pred = gt.bool(), pred.bool()
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0


def topk_mask(imp: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    n = imp.numel()
    k = max(1, int(frac * n))
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(imp, k).indices] = True
    return m


def star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def available_units():
    """{task: [seed, ...]} for whatever P1 units have a completed cache."""
    out = {t: [] for t in TASKS}
    for p in sorted(glob.glob(os.path.join(OUT, "cache_bxaic_*_s*.pt"))):
        m = re.search(r"cache_bxaic_(.+)_s(\d+)\.pt$", os.path.basename(p))
        if m and m.group(1) in out:
            out[m.group(1)].append(int(m.group(2)))
    return out


def per_unit(task: str, seed: int, data_list):
    blob = torch.load(os.path.join(OUT, f"cache_bxaic_{task}_s{seed}.pt"), weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(200000 + seed)

    arr = {e: [] for e in ALL_E}
    for k, mi in enumerate(test_idx):
        d = data_list[mi]
        gt = d.node_gt_mask.bool()
        if not bool(gt.any()):
            continue
        n = d.num_nodes
        for e in EXPL:
            imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
            arr[e].append(jaccard(gt, imp > imp.mean()))
        # random-node
        kk = max(1, int(0.25 * n))
        node_pick = rng.choice(n, size=kk, replace=False)
        pred_node = torch.zeros(n, dtype=torch.bool); pred_node[node_pick] = True
        arr["random_node"].append(jaccard(gt, pred_node))
        # random-edge
        edge_scores = torch.rand(d.edge_index.shape[1])
        folded = node_importance_from_edges(edge_scores, d.edge_index, n)
        arr["random_edge"].append(jaccard(gt, topk_mask(folded, 0.25)))
    return {e: np.array(v, float) for e, v in arr.items()}


def main() -> int:
    from ..data import LOADERS
    units = available_units()
    print("units available (task: seeds):", {t: sorted(s) for t, s in units.items()})

    by_task_seed = {}
    for t in TASKS:
        if not units[t]:
            continue
        data_list, meta = LOADERS[f"bxaic_{t}"]()
        for s in sorted(units[t]):
            by_task_seed[(t, s)] = per_unit(t, s, data_list)

    print("\n" + "=" * 100)
    print(" B-XAIC GEA -- GNN/PG/SX vs random-node/random-edge, matched top-k=0.25, per task/seed")
    print("=" * 100)
    for t in TASKS:
        rows = {s: by_task_seed[(t, s)] for s in sorted(units[t])}
        if not rows:
            print(f"\n[{t}]  -- no completed units yet")
            continue
        n_gea = len(next(iter(rows.values()))["gnnexplainer"])
        print(f"\n[{t}]  n_gea={n_gea}  seeds done: {sorted(rows)}")
        print(f"  {'seed':>4}" + "".join(f"{SH[e]:>10}" for e in ALL_E))
        for s in sorted(rows):
            r = rows[s]
            print(f"  {s:>4}" + "".join(f"{r[e].mean():>10.3f}" for e in ALL_E))
        if len(rows) > 1:
            means = {e: np.array([rows[s][e].mean() for s in sorted(rows)]) for e in ALL_E}
            print(f"  {'mean':>4}" + "".join(f"{means[e].mean():>10.3f}" for e in ALL_E))

    print("\n" + "-" * 100)
    print(" direction counts vs random (per-seed Wilcoxon within each (task,seed); does the real")
    print(" explainer beat its matched random baseline?)  -- SX vs RandN is the F3/F9 mirror question")
    print("-" * 100)
    for t in TASKS:
        rows = {s: by_task_seed[(t, s)] for s in sorted(units[t])}
        if not rows:
            continue
        print(f"\n[{t}]")
        for real, rand in (("gnnexplainer", "random_node"), ("subgraphx", "random_node"),
                           ("pgexplainer", "random_edge")):
            wins = 0
            for s in sorted(rows):
                r = rows[s]
                p = _wpair(r[real], r[rand])
                hi = SH[real] if r[real].mean() > r[rand].mean() else SH[rand]
                wins += int(r[real].mean() > r[rand].mean())
                print(f"    {SH[real]} vs {SH[rand]}  seed {s}: {SH[real]}={r[real].mean():.3f} "
                      f"{SH[rand]}={r[rand].mean():.3f}  p={p:.2e} {star(p):>3}  ({hi} higher)")
            print(f"    -> {SH[real]} beats {SH[rand]} in {wins}/{len(rows)} completed seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
