"""
Random baseline for mutag_graphxai GEA -- the anchor the paper has been
missing. GraphXAI itself flags trivially-recoverable ground truth as a known
pitfall (a small, structurally distinctive motif can be easy to hit by
chance), so SX averaging 0.065 and PG averaging 0.156 mean nothing without a
"what does chance score here" number.

Two random constructions, both at the SAME top-k=0.25-of-nodes budget every
other number in this project uses (masking.py's num_keep = max(1,
int(0.25*N)), the exact E_i Fidelity/GEF selection rule):

  random-node : uniformly pick max(1, int(0.25*N)) nodes, no model involved.
                The natural baseline for GNNExplainer/SubgraphX (node-level
                methods).
  random-edge : uniform random score per directed edge -> folded to node
                importance via the SAME node_importance_from_edges used for
                PGExplainer's real edge masks -> top-k=0.25 on the folded
                node importance. The natural baseline for PGExplainer
                (edge-level method), built through its exact pipeline.

5 independent replicates (one per seed 0-4, same seeds as the priority
sweep, n=188 molecules/seed -- all 188 have GT) so this plugs directly into
seed_correct_significance.py's per-seed / seed-level tests: GNN and SX vs
random-node, PG vs random-edge, plus the full cross-battery for completeness.

    python -m src.analysis.mutag_random_baseline
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..explain.common import node_importance_from_edges
from .seed_correct_significance import build_arrays_mutag_graphxai, holm, jaccard, star

REAL = ["gnnexplainer", "pgexplainer", "subgraphx"]
ALL_E = REAL + ["random_node", "random_edge"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX",
      "random_node": "RandN", "random_edge": "RandE"}
SEEDS = [0, 1, 2, 3, 4]


def topk_mask(imp: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    n = imp.numel()
    k = max(1, int(frac * n))  # matches masking.py's num_keep exactly
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(imp, k).indices] = True
    return m


def random_baselines(seed: int, data_list):
    rng = np.random.default_rng(seed)
    torch.manual_seed(100000 + seed)  # separate stream from the real explainers' torch seeding
    rn, re = [], []
    for d in data_list:
        gt = d.node_gt_mask.bool()
        if not bool(gt.any()):
            continue
        n = d.num_nodes
        k = max(1, int(0.25 * n))
        # random-node: uniform k-of-n node subset
        node_pick = rng.choice(n, size=k, replace=False)
        pred_node = torch.zeros(n, dtype=torch.bool)
        pred_node[node_pick] = True
        rn.append(jaccard(gt, pred_node))
        # random-edge: uniform random edge scores -> folded node importance -> same top-k
        edge_scores = torch.rand(d.edge_index.shape[1])
        folded = node_importance_from_edges(edge_scores, d.edge_index, n)
        pred_edge = topk_mask(folded, 0.25)
        re.append(jaccard(gt, pred_edge))
    return np.array(rn, float), np.array(re, float)


def _wpair(a, b):
    d = a - b
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def main() -> int:
    from ..data import LOADERS
    data_list, meta = LOADERS["mutag_graphxai"]()

    by_seed_real = build_arrays_mutag_graphxai()
    by_seed = {s: dict(by_seed_real[s]) for s in SEEDS}
    for s in SEEDS:
        rn, re = random_baselines(s, data_list)
        by_seed[s]["random_node"] = rn
        by_seed[s]["random_edge"] = re

    print("=" * 100)
    print(" mutag_graphxai GEA -- GNN/PG/SX vs random-node / random-edge baselines, matched top-k=0.25 budget")
    print("=" * 100)
    print(f"\n{'seed':>4}" + "".join(f"{SH[e]:>10}" for e in ALL_E))
    for s in SEEDS:
        print(f"{s:>4}" + "".join(f"{by_seed[s][e].mean():>10.3f}" for e in ALL_E))
    means = {e: np.array([by_seed[s][e].mean() for s in SEEDS]) for e in ALL_E}
    print(f"{'mean':>4}" + "".join(f"{means[e].mean():>10.3f}" for e in ALL_E))
    print(f"{'std':>4}" + "".join(f"{means[e].std():>10.3f}" for e in ALL_E))

    print("\n" + "-" * 100)
    print(" headline matched comparisons (per-seed Wilcoxon, n=188 each; seed-level sign count, n=5)")
    print("-" * 100)
    headline = [("gnnexplainer", "random_node"), ("subgraphx", "random_node"),
                ("pgexplainer", "random_edge"), ("pgexplainer", "random_node")]
    for a, b in headline:
        per_seed_p = [_wpair(by_seed[s][a], by_seed[s][b]) for s in SEEDS]
        per_seed_hi = [SH[a] if np.median(by_seed[s][a] - by_seed[s][b]) > 0
                       else SH[b] if np.median(by_seed[s][a] - by_seed[s][b]) < 0 else "=" for s in SEEDS]
        n_a_wins = sum(1 for s in SEEDS if means[a][SEEDS.index(s)] > means[b][SEEDS.index(s)])
        print(f"\n  {SH[a]} vs {SH[b]}:  means {means[a].mean():.3f} vs {means[b].mean():.3f}"
              f"  ({SH[a]} higher in {n_a_wins}/5 seeds by mean)")
        for i, s in enumerate(SEEDS):
            print(f"    seed {s}: {SH[a]}={by_seed[s][a].mean():.3f}  {SH[b]}={by_seed[s][b].mean():.3f}"
                  f"  p={per_seed_p[i]:.2e} {star(per_seed_p[i]):>3}  ({per_seed_hi[i]} higher)")
        seed_p = _wpair(means[a], means[b])
        print(f"    seed-level (n=5): {SH[a]} higher in {n_a_wins}/5, p={seed_p:.4f} {star(seed_p)}"
              f"  (min attainable p at n=5 is 0.0625)")

    print("\n" + "-" * 100)
    print(" full cross-battery, seed-level means (n=5 per explainer), Holm over all 10 pairs")
    print("-" * 100)
    from itertools import combinations
    pairs10 = list(combinations(ALL_E, 2))
    raw = {(a, b): _wpair(means[a], means[b]) for a, b in pairs10}
    adj = holm(raw)
    for a, b in pairs10:
        d = means[a] - means[b]
        hi = SH[a] if d.mean() > 0 else SH[b]
        n_hi = int((d > 0).sum()) if d.mean() > 0 else int((d < 0).sum())
        print(f"  {SH[a]:<6}-{SH[b]:<6}  {hi:<6} higher {n_hi}/5  p_Holm={adj[(a,b)]:.4f}  {star(adj[(a,b)])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
