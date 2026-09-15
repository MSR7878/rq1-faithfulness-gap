"""
Seed-aware significance for multi-seed GEA/Fid sweeps -- CORRECTS the naive
"pool every (unit, seed) observation and run one Wilcoxon" approach (used in
mutag_full_pooled_significance.py), which is pseudo-replicated: the same 188
molecules recur across all 5 seeds, so n=940 massively overstates the true
number of independent observations and inflates significance (p~1e-94 is
itself the tell). The actual independent unit is the SEED (a fresh model
fit), not the (molecule, seed) pair.

Two honest alternatives, both reported (no single "right" one -- they trade
off power for validity in opposite directions):

  1. PER-SEED test: paired Wilcoxon within each seed separately (n=188 for
     mutag_graphxai's GEA), Holm over the 3 pairs WITHIN that seed (matching
     F3's original per-block scope). 5 independent tests per pair. If the
     ordering/significance holds in 5/5 seeds, that is a stronger and more
     honest claim than one inflated pooled p -- "reproduces across every
     independent model fit" beats "big n, unclear independence".
  2. SEED-LEVEL test: paired Wilcoxon on the 5 per-seed MEANS (n=5) -- this
     is the test that respects the true unit of independence. Severely
     underpowered by construction: with n=5 and zero_method="wilcox", the
     minimum attainable two-sided p is 0.0625 (all 5 differences same sign),
     so it can NEVER reach the conventional 0.05 threshold even when the
     effect is perfectly consistent. Report it anyway, with that caveat, and
     read the SIGN/consistency (was it 5/5, 4/5, 3/5?) as the real signal,
     not the p-value.

Reusable across mutag_graphxai (this call), and B-XAIC/Tox21 once their
priority-sweep seeds land -- see build_arrays_mutag_graphxai() for the
dataset-specific loader; the two test functions take plain per-seed arrays.

    python -m src.analysis.seed_correct_significance
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import wilcoxon

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]


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


def _wpair(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    if np.allclose(d, 0):
        return 1.0
    return wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue


def per_seed_test(by_seed: dict[int, dict[str, np.ndarray]]) -> None:
    """by_seed[seed][explainer] = 1D array of per-molecule GEA for that seed."""
    seeds = sorted(by_seed)
    print(f"\n--- 1. PER-SEED test (n={len(next(iter(by_seed.values()))[EXPL[0]])} per seed, "
          f"Holm within each seed's 3 pairs) ---")
    print(f"  {'seed':>4}" + "".join(f"{SH[a]}-{SH[b]:<8}" for a, b in PAIRS) + "   means (GNN/PG/SX)")
    agree_order = []
    for s in seeds:
        arr = by_seed[s]
        raw = {(a, b): _wpair(arr[a], arr[b]) for a, b in PAIRS}
        adj = holm(raw)
        cells = []
        for a, b in PAIRS:
            hi = SH[a] if np.median(arr[a] - arr[b]) > 0 else SH[b] if np.median(arr[a] - arr[b]) < 0 else "="
            cells.append(f"{star(adj[(a,b)]):<3}({hi})  ")
        means = {e: arr[e].mean() for e in EXPL}
        order = " > ".join(SH[e] for e in sorted(EXPL, key=lambda e: -means[e]))
        agree_order.append(order)
        print(f"  {s:>4}  " + "".join(cells) + f"  {order}  ({', '.join(f'{SH[e]}={means[e]:.3f}' for e in EXPL)})")
    n_gnn_top = sum(1 for o in agree_order if o.startswith("GNN"))
    print(f"\n  GNN ranks #1 in {n_gnn_top}/{len(seeds)} seeds; exact ordering \"GNN > PG > SX\" in "
          f"{sum(1 for o in agree_order if o == 'GNN > PG > SX')}/{len(seeds)} seeds.")


def seed_level_test(by_seed: dict[int, dict[str, np.ndarray]]) -> None:
    """Paired Wilcoxon on the n=5 per-seed MEANS -- respects the true independence unit."""
    seeds = sorted(by_seed)
    means = {e: np.array([by_seed[s][e].mean() for s in seeds]) for e in EXPL}
    print(f"\n--- 2. SEED-LEVEL test (n={len(seeds)} per-seed means; min attainable two-sided p "
          f"= 0.0625 at this n -- READ THE SIGN/CONSISTENCY, not significance) ---")
    for e in EXPL:
        print(f"  {SH[e]:<4} per-seed means: {[round(v, 3) for v in means[e]]}   "
              f"mean={means[e].mean():.3f}  std={means[e].std():.3f}")
    raw = {(a, b): _wpair(means[a], means[b]) for a, b in PAIRS}
    adj = holm(raw)
    for a, b in PAIRS:
        d = means[a] - means[b]
        n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
        hi = SH[a] if d.mean() > 0 else SH[b]
        print(f"  {SH[a]}-{SH[b]:<3}  {hi} higher in {max(n_pos, n_neg)}/{len(seeds)} seeds  "
              f"p_raw={raw[(a,b)]:.4f}  p_Holm={adj[(a,b)]:.4f}  {star(adj[(a,b)])}")


def build_arrays_mutag_graphxai(prio_dir: str = "runs/prio", seeds=(0, 1, 2, 3, 4)):
    from ..data import LOADERS
    data_list, meta = LOADERS["mutag_graphxai"]()
    by_seed = {}
    for s in seeds:
        blob = torch.load(os.path.join(prio_dir, f"cache_mutag_graphxai_s{s}.pt"), weights_only=False)
        test_idx = list(blob["key"]["test_idx"])
        res = blob["results"]
        arr = {e: [] for e in EXPL}
        for k, mi in enumerate(test_idx):
            gt = data_list[mi].node_gt_mask.bool()
            if not bool(gt.any()):
                continue
            for e in EXPL:
                imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
                arr[e].append(jaccard(gt, imp > imp.mean()))
        by_seed[s] = {e: np.array(v, float) for e, v in arr.items()}
    return by_seed


def main() -> int:
    by_seed = build_arrays_mutag_graphxai()
    print("=" * 92)
    print(" mutag_graphxai GEA, seed-aware significance (corrects the pseudo-replicated n=940 pooled test)")
    print("=" * 92)
    per_seed_test(by_seed)
    seed_level_test(by_seed)

    print("\n--- per-seed GEA, all three explainers (is the huge variance PG-specific or general?) ---")
    seeds = sorted(by_seed)
    print(f"  {'seed':>4}{'GNN':>8}{'PG':>8}{'SX':>8}")
    for s in seeds:
        print(f"  {s:>4}{by_seed[s]['gnnexplainer'].mean():>8.3f}"
              f"{by_seed[s]['pgexplainer'].mean():>8.3f}{by_seed[s]['subgraphx'].mean():>8.3f}")
    for e in EXPL:
        v = np.array([by_seed[s][e].mean() for s in seeds])
        print(f"  {SH[e]:<4} range={v.max()-v.min():.3f}  (min={v.min():.3f} max={v.max():.3f})"
              f"  cv={v.std()/v.mean():.2f}" if v.mean() else f"  {SH[e]:<4} mean=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
