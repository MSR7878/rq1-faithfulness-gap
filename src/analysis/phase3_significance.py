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


def block_B():
    print("\n" + "#" * 78 + "\n# B. R2 zero-fill inflation vs R3 mean-fill (paired per molecule)\n" + "#" * 78)
    print(f"  {'dataset':<16}{'expl':<5}{'|Fid-| R3->R2':>22}{'p':>9}   {'GEF R3->R2':>20}{'p':>9}")
    for ds in DATASETS:
        dat = load(ds)
        for e in EXPL:
            fm_r3 = np.abs(col(dat, "R3", "mean", e, "fid_minus"))
            fm_r2 = np.abs(col(dat, "R2", "zero", e, "fid_minus"))
            gf_r3 = col(dat, "R3", "mean", e, "gef")
            gf_r2 = col(dat, "R2", "zero", e, "gef")
            _, p_f = _w(fm_r3, fm_r2)
            _, p_g = _w(gf_r3, gf_r2)
            print(f"  {ds:<16}{SHORT[e]:<5}{fm_r3.mean():>9.3f} -> {fm_r2.mean():<8.3f}{p_f:>9.4f}   "
                  f"{gf_r3.mean():>8.3f} -> {gf_r2.mean():<8.3f}{p_g:>9.4f}")


def block_C():
    print("\n" + "#" * 78 + "\n# C. Explainer separability UNDER R2 (zero-fill), Wilcoxon pairwise + Holm\n" + "#" * 78)
    for ds in DATASETS:
        dat = load(ds)
        print(f"\n[{ds}]")
        for metric in ("fid_plus", "fid_minus", "gef"):
            common, cols = common_col(dat, "R2", "zero", metric)
            raw = {f"{SHORT[a]}-{SHORT[b]}": _w(cols[a], cols[b])[1] for a, b in PAIRS}
            adj = holm(raw)
            means = "  ".join(f"{SHORT[e]}={cols[e].mean():+.3f}" for e in EXPL)
            sig = "  ".join(f"{k}:{star(adj[k])}" for k in raw)
            print(f"  {metric:<10} {means:<44}  {sig}")


def main() -> int:
    block_A()
    block_B()
    block_C()
    print("\nWilcoxon two-sided, paired per molecule; Holm-Bonferroni over the 3 pairwise "
          "tests within a block. GEA n=13 (bxaic) -> min attainable two-sided p ~= 2.4e-4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
