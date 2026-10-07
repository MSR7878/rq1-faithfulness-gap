"""
Oracle ceiling for top-k GEA: the best Jaccard ANY explainer could score
under the fixed top-k budget k = max(1, int(0.25 * N)), per molecule. No
model involved -- it depends only on N and the ground truth.

A top-k mask has exactly k atoms. Against a ground-truth mask of g atoms the
best choice contains the GT (if k >= g) or is a subset of it (if k < g):

    ceiling = min(g, k) / max(g, k)

For a set-valued ground truth (GraphXAI Mutagenicity, max-over-set GEA) the
ceiling is the max of that expression over every union of motif instances.
Mean-threshold GEA has no size budget, so its ceiling is 1.0 by construction.

Datasets: GraphXAI Mutagenicity (all GT-present graphs <= 80 nodes), MUTAG-GT
(all 188 graphs, the explained pool), B-XAIC indole (the GT-present explained
test molecules of each priority-sweep seed). Random floors for the last two
are read from the saved per-molecule arrays (runs/prio/baseline_arrays_*.npz).

    python -m src.analysis.oracle_ceiling
"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from ..data import LOADERS

OUT = "runs/mutagenicity/oracle_ceiling.json"
SEEDS = [0, 1, 2, 3, 4]


def budget(n: int) -> int:
    return max(1, int(0.25 * n))


def ceil_single(g: int, k: int) -> float:
    return min(g, k) / max(g, k)


def ceil_set(motifs, n: int) -> float:
    k = budget(n)
    node_sets = [set(m[1]) for m in motifs]
    best = 0.0
    for r in range(1, len(node_sets) + 1):
        for combo in itertools.combinations(node_sets, r):
            best = max(best, ceil_single(len(set().union(*combo)), k))
            if best == 1.0:
                return 1.0
    return best


def describe(name, ceil, g, k, extra=""):
    ceil, g, k = np.array(ceil), np.array(g), np.array(k)
    print(f"{name:<34} n={len(ceil):>4}  ceiling mean {ceil.mean():.3f}  median {np.median(ceil):.3f}"
          f"  ==1.0 in {100 * (ceil == 1).mean():.0f}%   GT {g.mean():.2f} atoms  budget {k.mean():.2f}"
          f"   k>GT {100 * (k > g).mean():.0f}%  k<GT {100 * (k < g).mean():.0f}% {extra}")
    return {"n": int(len(ceil)), "ceiling_mean": float(ceil.mean()), "ceiling_median": float(np.median(ceil)),
            "frac_ceiling_1": float((ceil == 1).mean()), "gt_size_mean": float(g.mean()),
            "budget_mean": float(k.mean()), "frac_budget_gt_gt": float((k > g).mean()),
            "frac_budget_lt_gt": float((k < g).mean())}


def main():
    rep = {}

    # ---- GraphXAI Mutagenicity ----
    dl, meta = LOADERS["mutagenicity_graphxai"]()
    motifs = meta.extras["motifs"]
    pos = [i for i, d in enumerate(dl) if bool(d.node_gt_mask.any()) and d.num_nodes <= 80]
    n = [dl[i].num_nodes for i in pos]
    g = [int(dl[i].node_gt_mask.sum()) for i in pos]
    k = [budget(x) for x in n]
    merged = [ceil_single(gi, ki) for gi, ki in zip(g, k)]
    mset = [ceil_set(motifs[i], dl[i].num_nodes) for i in pos]
    print("=" * 150)
    rep["mutagenicity_merged"] = describe("Mutagenicity, merged mask", merged, g, k)
    rep["mutagenicity_max_over_set"] = describe("Mutagenicity, max-over-set", mset, g, k)
    halide_only = [all(t == "halide" for t, _ in motifs[i]) for i in pos]
    for lab, sel in (("halide-only", halide_only), ("NH2/NO2/azo/nitroso", [not h for h in halide_only])):
        idx = [j for j, s in enumerate(sel) if s]
        rep[f"mutagenicity_max_over_set[{lab}]"] = describe(
            f"  {lab}, max-over-set", [mset[j] for j in idx], [g[j] for j in idx], [k[j] for j in idx])
        rep[f"mutagenicity_merged[{lab}]"] = describe(
            f"  {lab}, merged", [merged[j] for j in idx], [g[j] for j in idx], [k[j] for j in idx])

    # ---- MUTAG-GT (mutag_graphxai), all 188 graphs ----
    dl, _ = LOADERS["mutag_graphxai"]()
    pos = [i for i, d in enumerate(dl) if bool(d.node_gt_mask.any())]
    n = [dl[i].num_nodes for i in pos]; g = [int(dl[i].node_gt_mask.sum()) for i in pos]; k = [budget(x) for x in n]
    floor = ""
    p = "runs/prio/baseline_arrays_mutag_graphxai.npz"
    if os.path.exists(p):
        z = np.load(p)
        rn = np.mean([z[f"random_node_s{s}"].mean() for s in SEEDS]); re = np.mean([z[f"random_edge_s{s}"].mean() for s in SEEDS])
        floor = f"  | floor RandNode {rn:.3f} RandEdge {re:.3f}"
        rep["mutag_graphxai_floor"] = {"random_node": float(rn), "random_edge": float(re)}
    rep["mutag_graphxai"] = describe("MUTAG-GT (188 graphs)", [ceil_single(a, b) for a, b in zip(g, k)], g, k, floor)

    # ---- B-XAIC indole, per-seed explained GT-present molecules ----
    dl, _ = LOADERS["bxaic_indole"]()
    p = "runs/prio/baseline_arrays_bxaic_indole.npz"
    if os.path.exists(p):
        z = np.load(p); allc, allg, allk, per_seed = [], [], [], []
        for s in SEEDS:
            idx = z[f"mol_idx_s{s}"].tolist()
            n = [dl[i].num_nodes for i in idx]; g = [int(dl[i].node_gt_mask.sum()) for i in idx]; k = [budget(x) for x in n]
            c = [ceil_single(a, b) for a, b in zip(g, k)]
            per_seed.append(float(np.mean(c))); allc += c; allg += g; allk += k
        rn = np.mean([z[f"random_node_s{s}"].mean() for s in SEEDS]); re = np.mean([z[f"random_edge_s{s}"].mean() for s in SEEDS])
        rep["bxaic_indole"] = describe("B-XAIC indole (5 seeds pooled)", allc, allg, allk,
                                       f"  | floor RandNode {rn:.3f} RandEdge {re:.3f}")
        rep["bxaic_indole"]["per_seed_ceiling"] = per_seed
        rep["bxaic_indole_floor"] = {"random_node": float(rn), "random_edge": float(re)}
        print(f"  indole per-seed ceiling means: {[round(x, 3) for x in per_seed]}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
