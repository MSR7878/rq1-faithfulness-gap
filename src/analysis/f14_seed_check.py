"""
F14 seed-aware re-test: does GEA binarization sensitivity (mean-threshold vs
top-k=0.25, src/analysis/gea_binarization.py) hold up per seed? Original F14
was single-seed. Reuses the same topk_mask/jaccard construction, applied to
each of the 5 priority-sweep caches for mutag_graphxai + all 4 B-XAIC tasks.

Reports, per (dataset, seed): GEA order under mean-threshold vs under top-k,
and whether they agree (same 3-way ranking).

    python -m src.analysis.f14_seed_check
"""
from __future__ import annotations
import numpy as np
import torch
from ..data import LOADERS
from .gea_binarization import topk_mask, jaccard

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
SEEDS = [0, 1, 2, 3, 4]
OUT = "runs/prio"

DATASETS = {
    "mutag_graphxai": ("mutag_graphxai", lambda s: f"{OUT}/cache_mutag_graphxai_s{s}.pt"),
    "indole": ("bxaic_indole", lambda s: f"{OUT}/cache_bxaic_indole_s{s}.pt"),
    "PAINS": ("bxaic_PAINS", lambda s: f"{OUT}/cache_bxaic_PAINS_s{s}.pt"),
    "X": ("bxaic_X", lambda s: f"{OUT}/cache_bxaic_X_s{s}.pt"),
    "P": ("bxaic_P", lambda s: f"{OUT}/cache_bxaic_P_s{s}.pt"),
}


def gea_orders(data_list, cache_path):
    blob = torch.load(cache_path, weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    mean_vals = {e: [] for e in EXPL}
    topk_vals = {e: [] for e in EXPL}
    for k, mi in enumerate(test_idx):
        gt = data_list[mi].node_gt_mask.bool()
        if not bool(gt.any()):
            continue
        for e in EXPL:
            imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
            mean_vals[e].append(jaccard(gt, imp > imp.mean()))
            topk_vals[e].append(jaccard(gt, topk_mask(imp, 0.25)))
    mean_means = {e: float(np.mean(v)) for e, v in mean_vals.items() if v}
    topk_means = {e: float(np.mean(v)) for e, v in topk_vals.items() if v}
    mo = tuple(sorted(mean_means, key=lambda e: -mean_means[e]))
    to = tuple(sorted(topk_means, key=lambda e: -topk_means[e]))
    return mo, mean_means, to, topk_means


def main():
    for ds, (loader_key, path_fn) in DATASETS.items():
        print(f"\n{'='*90}\n {ds}\n{'='*90}", flush=True)
        data_list, meta = LOADERS[loader_key]()
        agree = 0
        for s in SEEDS:
            mo, mm, to, tm = gea_orders(data_list, path_fn(s))
            same = mo == to
            agree += int(same)
            print(f"  seed {s}  mean-thr: {'>'.join(SH[e] for e in mo)} "
                  f"({', '.join(f'{SH[e]}={mm[e]:.2f}' for e in mo)})  |  "
                  f"top-k: {'>'.join(SH[e] for e in to)} "
                  f"({', '.join(f'{SH[e]}={tm[e]:.2f}' for e in to)})  "
                  f"{'SAME' if same else 'DIFFERENT'}")
        print(f"  -> order agrees between binarizations in {agree}/5 seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
