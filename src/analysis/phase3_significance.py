"""
Phase 3 -- paired significance across the combined R3+R2 sweep.

Reads runs/phase3_<dataset>.json (schema: summary[masking][fill][explainer],
per_molecule[masking][fill][explainer]) for all five variants and reports:

  A. GEA ranking (GT datasets: mutag_graphxai, bxaic) -- per-explainer mean,
     std, median, 95% bootstrap CI; Wilcoxon signed-rank pairwise
     (GNN-PG, GNN-SX, PG-SX), Holm-Bonferroni within the block.
  B. R2 inflation -- per (dataset, explainer): Wilcoxon paired R3-mean vs
     R2-zero on |Fid-| and GEF (is the zero-fill perturbation artifact real?).
  C. Explainer separability under R2 -- per dataset: Wilcoxon pairwise on
     Fid+ / Fid- / GEF (does R2 distinguish explainers where R3 could not?).

    python -m src.analysis.phase3_significance
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SHORT = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]
DATASETS = ["mutag_graphxai", "mutag", "bbbp", "tox21_srp53", "bxaic"]
GT_DATASETS = ["mutag_graphxai", "bxaic"]


def load(ds):
    with open(f"runs/phase3_{ds}.json") as f:
        return json.load(f)


def col(dat, masking, fill, explainer, metric):
    return np.array([r[metric] for r in dat["per_molecule"][masking][fill][explainer] if metric in r],
                    dtype=float)


def common_col(dat, masking, fill, metric):
    """per-explainer arrays over molecule positions where `metric` is present for ALL 3."""
    lists = {e: [(i, r[metric]) for i, r in enumerate(dat["per_molecule"][masking][fill][e]) if metric in r]
             for e in EXPL}
    common = sorted(set.intersection(*[{i for i, _ in lists[e]} for e in EXPL]))
    return common, {e: np.array([dict(lists[e])[i] for i in common]) for e in EXPL}


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def _w(a, b):
    if np.allclose(a - b, 0):
        return float("nan"), 1.0
    return wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def block_A():
    print("\n" + "#" * 78 + "\n# A. GEA ranking (per-molecule Jaccard; masking-independent)\n" + "#" * 78)
    for ds in GT_DATASETS:
        dat = load(ds)
        common, cols = common_col(dat, "R3", "mean", "gea")
        print(f"\n[{ds}]  n={len(common)} GT-present molecules")
        for e in EXPL:
            x = cols[e]
            lo, hi = boot_ci(x)
            print(f"  {SHORT[e]:4s} mean={x.mean():.3f} std={x.std(ddof=1):.3f} median={np.median(x):.3f} "
                  f"95%CI=[{lo:.3f}, {hi:.3f}]")
        raw = {f"{SHORT[a]}-{SHORT[b]}": _w(cols[a], cols[b])[1] for a, b in PAIRS}
        adj = holm(raw)
        for a, b in PAIRS:
            k = f"{SHORT[a]}-{SHORT[b]}"
            hi_side = SHORT[a] if np.median(cols[a] - cols[b]) > 0 else SHORT[b]
            print(f"    {k:8s} medianΔ={np.median(cols[a]-cols[b]):+.3f}  p_raw={raw[k]:.4f}  "
                  f"p_Holm={adj[k]:.4f}  {star(adj[k]):>3}  ({hi_side} higher)")


MASK_FILL = {"R3": ("R3", "mean"), "R2": ("R2", "zero"), "R1": ("R1", "hard")}


def block_B():
    print("\n" + "#" * 78 + "\n# B. Masking inflation vs R3 mean-fill (paired per molecule, |Fid-| and GEF)\n" + "#" * 78)
    print(f"  {'dataset':<16}{'expl':<5}   {'|Fid-|  R3 -> R2 (p) -> R1 (p)':<40}   {'GEF  R3 -> R2 (p) -> R1 (p)'}")
    for ds in DATASETS:
        dat = load(ds)
        for e in EXPL:
            def g(mk, fl, metric):
                return col(dat, mk, fl, e, metric)
            fm = {m: np.abs(g(*MASK_FILL[m], "fid_minus")) for m in ("R3", "R2", "R1")}
            gf = {m: g(*MASK_FILL[m], "gef") for m in ("R3", "R2", "R1")}
            p_f2, p_f1 = _w(fm["R3"], fm["R2"])[1], _w(fm["R3"], fm["R1"])[1]
            p_g2, p_g1 = _w(gf["R3"], gf["R2"])[1], _w(gf["R3"], gf["R1"])[1]
            print(f"  {ds:<16}{SHORT[e]:<5}   "
                  f"{fm['R3'].mean():.2f} ->{fm['R2'].mean():.2f}({p_f2:.3f}) ->{fm['R1'].mean():.2f}({p_f1:.3f})".ljust(40)
                  + f"   {gf['R3'].mean():.2f} ->{gf['R2'].mean():.2f}({p_g2:.3f}) ->{gf['R1'].mean():.2f}({p_g1:.3f})")


def block_C():
    print("\n" + "#" * 78 + "\n# C. Explainer separability UNDER each masking, Wilcoxon pairwise + Holm\n" + "#" * 78)
    for ds in DATASETS:
        dat = load(ds)
        print(f"\n[{ds}]")
        for mk in ("R2", "R1"):
            _, fl = MASK_FILL[mk]
            for metric in ("fid_plus", "fid_minus", "gef"):
                common, cols = common_col(dat, mk, fl, metric)
                raw = {f"{SHORT[a]}-{SHORT[b]}": _w(cols[a], cols[b])[1] for a, b in PAIRS}
                adj = holm(raw)
                means = "  ".join(f"{SHORT[e]}={cols[e].mean():+.3f}" for e in EXPL)
                sig = "  ".join(f"{k}:{star(adj[k])}" for k in raw)
                print(f"  {mk} {metric:<10} {means:<44}  {sig}")


def block_D():
    print("\n" + "#" * 78 + "\n# D. Cross-metric ranking agreement (GT datasets): GEA vs Fid+ under each\n"
          "#    masking. Do the metrics rank the explainers the same way?\n" + "#" * 78)
    for ds in GT_DATASETS:
        dat = load(ds)
        rank_gea = {e: col(dat, "R3", "mean", e, "gea").mean() for e in EXPL}
        order_gea = sorted(EXPL, key=lambda e: -rank_gea[e])
        print(f"\n[{ds}]  GEA (best->worst): {' > '.join(SHORT[e] for e in order_gea)}"
              f"   ({', '.join(f'{SHORT[e]}={rank_gea[e]:.2f}' for e in order_gea)})")
        for mk, fl in (("R3", "mean"), ("R2", "zero"), ("R1", "hard")):
            rank_f = {e: col(dat, mk, fl, e, "fid_plus").mean() for e in EXPL}
            order_f = sorted(EXPL, key=lambda e: -rank_f[e])
            same = "agrees" if order_f == order_gea else "DIFFERS"
            print(f"        {mk}-Fid+ : {' > '.join(SHORT[e] for e in order_f)}"
                  f"   ({', '.join(f'{SHORT[e]}={rank_f[e]:.2f}' for e in order_f)})   [{same} with GEA]")


def main() -> int:
    block_A()
    block_B()
    block_C()
    block_D()
    print("\nWilcoxon two-sided, paired per molecule; Holm-Bonferroni over the 3 pairwise "
          "tests within a block. GEA n=13 (bxaic) -> min attainable two-sided p ~= 2.4e-4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
