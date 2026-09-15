"""
mutag_graphxai GEA, pooled across the 5 priority-sweep seeds (--explain-pool
all, all 188 graphs/seed = 940 molecule-seed observations): paired Wilcoxon +
Holm, same method as F3's single-seed test (src/analysis/phase3_significance
.py block A), to see whether the ranking / significance changes now that PG's
mean has moved (0.300 single-seed -> 0.156 pooled) and n is 5x larger.

Also reports PGExplainer's GEA with vs without seed 3 (PGExplainer collapsed
-- all-0.000 mask, F10's seed-noise failure mode) and the per-seed spread, so
a reader can see the bimodal distribution the pooled mean averages over.

    python -m src.analysis.mutag_full_pooled_significance
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]
OUT = "runs/prio"
SEEDS = [0, 1, 2, 3, 4]


def jaccard(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt, pred = gt.bool(), pred.bool()
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0


def star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def main() -> int:
    from ..data import LOADERS
    data_list, meta = LOADERS["mutag_graphxai"]()

    # per-explainer arrays over the 940 (molecule, seed) pairs, aligned across explainers
    pooled = {e: [] for e in EXPL}
    per_seed_pg = {}
    for s in SEEDS:
        blob = torch.load(os.path.join(OUT, f"cache_mutag_graphxai_s{s}.pt"), weights_only=False)
        test_idx = list(blob["key"]["test_idx"])
        res = blob["results"]
        seed_pg = []
        for k, mi in enumerate(test_idx):
            gt = data_list[mi].node_gt_mask.bool()
            if not bool(gt.any()):
                continue
            row = {}
            for e in EXPL:
                imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
                pred = imp > imp.mean()
                row[e] = jaccard(gt, pred)
            for e in EXPL:
                pooled[e].append(row[e])
            seed_pg.append(row["pgexplainer"])
        per_seed_pg[s] = np.array(seed_pg, float)

    arr = {e: np.array(pooled[e], float) for e in EXPL}
    n = len(arr["gnnexplainer"])
    print(f"pooled n = {n} (molecule x seed pairs, 188 x 5 seeds)")
    print(f"{'expl':<5}{'mean':>8}{'std':>8}{'median':>8}")
    for e in EXPL:
        print(f"{SH[e]:<5}{arr[e].mean():>8.3f}{arr[e].std():>8.3f}{np.median(arr[e]):>8.3f}")

    print("\n" + "=" * 78)
    print(" 5-seed pooled pairwise significance (Wilcoxon signed-rank, paired by")
    print(" molecule-seed observation; Holm-Bonferroni over the 3 pairs) -- same")
    print(" method as F3's single-seed test")
    print("=" * 78)
    raw = {}
    for a, b in PAIRS:
        d = arr[a] - arr[b]
        raw[(a, b)] = 1.0 if np.allclose(d, 0) else wilcoxon(arr[a], arr[b],
                                                             alternative="two-sided", zero_method="wilcox").pvalue
    adj = holm(raw)
    for a, b in PAIRS:
        med = np.median(arr[a] - arr[b])
        hi = SH[a] if med > 0 else SH[b] if med < 0 else "tie"
        print(f"  {SH[a]}-{SH[b]:<3}  medianD={med:+.3f}  p_raw={raw[(a,b)]:.2e}  "
              f"p_Holm={adj[(a,b)]:.2e}  {star(adj[(a,b)]):>3}  ({hi} higher)")
    order = " > ".join(SH[e] for e in sorted(EXPL, key=lambda e: -arr[e].mean()))
    print(f"\n  order (by mean): {order}")

    print("\n" + "=" * 78)
    print(" PGExplainer: with vs without the collapsed seed 3 (all-0.000, F10 pattern)")
    print("=" * 78)
    print(f"  {'seed':>4}{'PG GEA mean':>14}{'n':>6}{'collapsed?':>12}")
    for s in SEEDS:
        v = per_seed_pg[s]
        collapsed = np.allclose(v, 0.0)
        print(f"  {s:>4}{v.mean():>14.3f}{len(v):>6}{'YES' if collapsed else '':>12}")
    with_s3 = np.concatenate([per_seed_pg[s] for s in SEEDS])
    without_s3 = np.concatenate([per_seed_pg[s] for s in SEEDS if s != 3])
    print(f"\n  WITH seed 3:    mean={with_s3.mean():.3f}  std={with_s3.std():.3f}  n={len(with_s3)}")
    print(f"  WITHOUT seed 3: mean={without_s3.mean():.3f}  std={without_s3.std():.3f}  n={len(without_s3)}")
    non_collapsed_seed_means = [per_seed_pg[s].mean() for s in SEEDS if s != 3]
    print(f"  non-collapsed seed means: {[round(m, 3) for m in non_collapsed_seed_means]}"
          f"  (range {min(non_collapsed_seed_means):.3f}-{max(non_collapsed_seed_means):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
