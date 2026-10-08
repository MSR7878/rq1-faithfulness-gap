"""
EXPLORATORY (not part of any registered prediction). Two checks:

1. What are PGExplainer's below-random masks? For Mutagenicity seeds 0-4 and
   B-XAIC X seeds 0-4: health counts, mean-threshold mask sizes, per-molecule
   spread of node importances, where the GT atoms rank, GEA of the INVERTED
   importance, and the atom-type composition of selected vs GT atoms.
2. Normalised top-k GEA, (GEA - RandNode) / (ceiling - RandNode), per seed,
   for Mutagenicity, MUTAG-GT and B-XAIC indole.

Reads cached explanations only. Writes runs/mutagenicity/exploratory/.

    python -m src.analysis.exploratory_pg_and_normalised
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import torch

from ..data import LOADERS
from ..data.bxaic import SYMBOLS as BX_SYMBOLS
from ..data.graphxai_mutagenicity import ATOM_ORDER as MU_SYMBOLS

OUT = "runs/mutagenicity/exploratory"
SEEDS = [0, 1, 2, 3, 4]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
CEILING = {"mutagenicity": 0.551, "mutag_graphxai": 0.751, "bxaic_indole": 0.724}


def jac(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def topk(imp: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros(len(imp), bool)
    m[torch.topk(torch.as_tensor(imp, dtype=torch.float), k).indices.numpy()] = True
    return m


def pct(c: Counter, symbols, keep=6):
    tot = sum(c.values())
    items = sorted(c.items(), key=lambda kv: -kv[1])[:keep]
    return " ".join(f"{symbols[a]}:{100 * v / tot:.0f}%" for a, v in items) if tot else "-"


def diag_unit(label, dl, cache_path, symbols, expl="pgexplainer"):
    blob = torch.load(cache_path, weights_only=False)
    test_idx = list(blob["key"]["test_idx"]); res = blob["results"][expl]
    n_empty = n_whole = n_degen = 0
    size, frac, std, rng_, uniq, rankp, g_top, g_inv, g_mean, estd = [], [], [], [], [], [], [], [], [], []
    sel_c, gt_c, all_c = Counter(), Counter(), Counter()
    for pos, mi in enumerate(test_idx):
        d = dl[mi]; imp = np.asarray(res[pos]["node_importance"], dtype=np.float64); n = len(imp)
        mm = imp > imp.mean()
        n_empty += int(mm.sum() == 0); n_whole += int(mm.sum() >= n); n_degen += int(imp.std() == 0.0)
        gt = d.node_gt_mask.numpy().astype(bool)
        if not gt.any():
            continue
        atom = d.x.argmax(1).numpy(); k = max(1, int(0.25 * n))
        size.append(int(mm.sum())); frac.append(mm.mean()); std.append(imp.std()); rng_.append(imp.max() - imp.min())
        uniq.append(len(np.unique(np.round(imp, 6))) / n)
        order = imp.argsort().argsort() / max(n - 1, 1)          # 0 = lowest importance, 1 = highest
        rankp.append(order[gt].mean())
        g_top.append(jac(topk(imp, k), gt)); g_inv.append(jac(topk(-imp, k), gt)); g_mean.append(jac(mm, gt))
        ei = res[pos].get("edge_importance")
        if ei is not None:
            estd.append(float(np.std(ei)))
        for a in atom[mm]: sel_c[int(a)] += 1
        for a in atom[gt]: gt_c[int(a)] += 1
        for a in atom: all_c[int(a)] += 1
    size = np.array(size); std = np.array(std)
    r = {"unit": label, "n_explained": len(test_idx), "n_gt_present": int(len(size)),
         "n_empty": n_empty, "n_whole": n_whole, "n_degenerate": n_degen,
         "size_mean": float(size.mean()), "size_median": float(np.median(size)),
         "size_p10": float(np.percentile(size, 10)), "size_p90": float(np.percentile(size, 90)),
         "size_frac_mean": float(np.mean(frac)), "std_median": float(np.median(std)), "std_min": float(std.min()),
         "range_median": float(np.median(rng_)), "uniq_frac_median": float(np.median(uniq)),
         "edge_std_median": float(np.median(estd)) if estd else None,
         "gt_rank_pct_mean": float(np.mean(rankp)), "gea_mean_thr": float(np.mean(g_mean)),
         "gea_topk": float(np.mean(g_top)), "gea_topk_inverted": float(np.mean(g_inv)),
         "selected": pct(sel_c, symbols), "gt": pct(gt_c, symbols), "all_atoms": pct(all_c, symbols)}
    return r


def print_diag(rows):
    print(f"  {'unit':<22}{'n':>5}{'empty':>6}{'whole':>6}{'degen':>6} |{'|M| mean':>9}{'med':>5}{'p10':>5}{'p90':>5}{'|M|/N':>7}"
          f" |{'std med':>9}{'std min':>10}{'range med':>10}{'uniq/N':>8}{'edge std':>9} |{'GT rank':>8}{'GEA m-thr':>10}{'GEA top-k':>10}{'inverted':>9}")
    for r in rows:
        es = f"{r['edge_std_median']:.4f}" if r["edge_std_median"] is not None else "-"
        print(f"  {r['unit']:<22}{r['n_explained']:>5}{r['n_empty']:>6}{r['n_whole']:>6}{r['n_degenerate']:>6} |"
              f"{r['size_mean']:>9.2f}{r['size_median']:>5.0f}{r['size_p10']:>5.0f}{r['size_p90']:>5.0f}{r['size_frac_mean']:>7.2f} |"
              f"{r['std_median']:>9.4f}{r['std_min']:>10.2e}{r['range_median']:>10.4f}{r['uniq_frac_median']:>8.2f}{es:>9} |"
              f"{r['gt_rank_pct_mean']:>8.2f}{r['gea_mean_thr']:>10.3f}{r['gea_topk']:>10.3f}{r['gea_topk_inverted']:>9.3f}")
    print("  atom-type composition (share of atoms), mean-threshold selection vs ground truth vs all atoms in those molecules:")
    for r in rows:
        print(f"    {r['unit']:<22} selected [{r['selected']}]   GT [{r['gt']}]   all [{r['all_atoms']}]")


def check1(rep):
    print("=" * 190); print(" CHECK 1 (exploratory): PGExplainer masks -- near-collapse or valid-but-wrong?"); print("=" * 190)
    print(" GT rank = mean percentile rank of GT atoms' importance within the molecule (1 = highest, 0.5 = chance, 0 = lowest)."
          "\n inverted = top-k GEA of the NEGATED importance (merged-mask GT, GT-present test molecules).")
    dl, _ = LOADERS["mutagenicity_graphxai"]()
    print("\n Mutagenicity, PGExplainer")
    rows = [diag_unit(f"mutagenicity s{s}", dl, f"runs/mutagenicity/cache_s{s}.pt", MU_SYMBOLS) for s in SEEDS]
    print_diag(rows); rep["pg_mutagenicity"] = rows
    for e in ("gnnexplainer", "subgraphx"):
        print(f"\n Mutagenicity, {SH[e]} (reference: what a working explainer's numbers look like)")
        rr = [diag_unit(f"mutagenicity s{s}", dl, f"runs/mutagenicity/cache_s{s}.pt", MU_SYMBOLS, e) for s in SEEDS]
        print_diag(rr); rep[f"{e}_mutagenicity"] = rr
    dlx, _ = LOADERS["bxaic_X"]()
    print("\n B-XAIC X (halogen), PGExplainer")
    rows = [diag_unit(f"bxaic X s{s}", dlx, f"runs/prio/cache_bxaic_X_s{s}.pt", BX_SYMBOLS) for s in SEEDS]
    print_diag(rows); rep["pg_bxaic_X"] = rows
    dlm, _ = LOADERS["mutag_graphxai"]()
    print("\n MUTAG-GT, PGExplainer (seed 3 is the known strict collapse; seeds 1-2 were below random, F18)")
    rows = [diag_unit(f"mutag_graphxai s{s}", dlm, f"runs/prio/cache_mutag_graphxai_s{s}.pt",
                      ["C", "N", "O", "F", "I", "Cl", "Br"]) for s in SEEDS]
    print_diag(rows); rep["pg_mutag_graphxai"] = rows


def check2(rep):
    print("\n" + "=" * 120); print(" CHECK 2 (exploratory): normalised top-k GEA = (GEA - RandNode) / (ceiling - RandNode), per seed"); print("=" * 120)
    per = {}
    # Mutagenicity: persisted arrays (max-over-set, top-k)
    per["mutagenicity"] = {}
    for s in SEEDS:
        z = np.load(f"runs/mutagenicity/arrays_s{s}.npz")
        per["mutagenicity"][s] = {**{e: float(z[f"{e}_topk_set"].mean()) for e in EXPL}, "rand": float(z["rand_node_topk_set"].mean())}
    # MUTAG-GT and indole: top-k GEA recomputed from cached explanations; RandNode from the saved arrays
    for key, cache_fn, npz in (("mutag_graphxai", "runs/prio/cache_mutag_graphxai_s{s}.pt", "runs/prio/baseline_arrays_mutag_graphxai.npz"),
                               ("bxaic_indole", "runs/prio/cache_bxaic_indole_s{s}.pt", "runs/prio/baseline_arrays_bxaic_indole.npz")):
        dl, _ = LOADERS[key]()
        z = np.load(npz); per[key] = {}
        for s in SEEDS:
            blob = torch.load(cache_fn.format(s=s), weights_only=False)
            test_idx = list(blob["key"]["test_idx"]); vals = {e: [] for e in EXPL}
            for pos, mi in enumerate(test_idx):
                gt = dl[mi].node_gt_mask.numpy().astype(bool)
                if not gt.any():
                    continue
                for e in EXPL:
                    imp = np.asarray(blob["results"][e][pos]["node_importance"], dtype=np.float64)
                    vals[e].append(jac(topk(imp, max(1, int(0.25 * len(imp)))), gt))
            per[key][s] = {**{e: float(np.mean(vals[e])) for e in EXPL}, "rand": float(z[f"random_node_s{s}"].mean())}
    rep["normalised"] = {}
    for key, name in (("mutagenicity", "Mutagenicity (max-over-set)"), ("mutag_graphxai", "MUTAG-GT"), ("bxaic_indole", "B-XAIC indole")):
        c = CEILING[key]
        print(f"\n {name}   ceiling {c:.3f}")
        print(f"  {'seed':>4}{'RandNode':>10}" + "".join(f"{'GEA ' + SH[e]:>10}" for e in EXPL) + "   |" + "".join(f"{'norm ' + SH[e]:>10}" for e in EXPL))
        norm = {e: [] for e in EXPL}
        for s in SEEDS:
            p = per[key][s]
            for e in EXPL:
                norm[e].append((p[e] - p["rand"]) / (c - p["rand"]))
            print(f"  {s:>4}{p['rand']:>10.3f}" + "".join(f"{p[e]:>10.3f}" for e in EXPL) + "   |" + "".join(f"{norm[e][-1]:>10.3f}" for e in EXPL))
        print(f"  {'mean':>4}{'':>10}{'':>30}   |" + "".join(f"{np.mean(norm[e]):>10.3f}" for e in EXPL))
        print(f"  {'std':>4}{'':>10}{'':>30}   |" + "".join(f"{np.std(norm[e]):>10.3f}" for e in EXPL))
        rep["normalised"][key] = {"ceiling": c, "per_seed_raw": per[key],
                                  "norm": {SH[e]: {"per_seed": norm[e], "mean": float(np.mean(norm[e])), "std": float(np.std(norm[e]))} for e in EXPL}}


def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"status": "EXPLORATORY -- not part of any registered prediction"}
    check1(rep)
    check2(rep)
    json.dump(rep, open(f"{OUT}/exploratory.json", "w"), indent=1)
    print(f"\nsaved -> {OUT}/exploratory.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
