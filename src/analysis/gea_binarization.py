"""
GEA under two explanation binarizations, on the cached explanations:

  mean-threshold  : pred = node_imp > node_imp.mean()          (current pipeline / GraphXAI)
  top-k=0.25      : pred = top max(1, int(0.25*N)) nodes by importance
                    -- the SAME node set Fidelity/GEF's E_i uses (K_FRAC, masking.py)

Reports, for mutag_graphxai + all 4 B-XAIC tasks (indole, PAINS, X, P):
  1. GEA per explainer under both binarizations, side by side, + paired Wilcoxon/Holm
  2. F3 (mutag_graphxai vs B-XAIC inversion) and F9 (per-task) verdicts under top-k
  3. block D / F8 "1 of 8": GEA-order vs Fid+-order when BOTH use top-k
  4. per-explainer mean predicted-mask size under each binarization (vs N and |GT|)

    python -m src.analysis.gea_binarization
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..data import LOADERS

EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
PAIRS = [("gnnexplainer", "pgexplainer"), ("gnnexplainer", "subgraphx"), ("pgexplainer", "subgraphx")]

# cache file -> (loader key, phase3 json for Fid+ orderings or None)
CASES = [
    ("mutag_graphxai", "runs/expl_cache_mutag_graphxai.pt", "mutag_graphxai", "runs/phase3_mutag_graphxai.json"),
    ("bxaic:indole",   "runs/expl_cache_bxaic.pt",          "bxaic",         "runs/phase3_bxaic.json"),
    ("bxaic:PAINS",    "runs/expl_cache_bxaic_PAINS.pt",     "bxaic_PAINS",   "runs/scaleout/phase3_bxaic_PAINS.json"),
    ("bxaic:X",        "runs/expl_cache_bxaic_X.pt",         "bxaic_X",       "runs/scaleout/phase3_bxaic_X.json"),
    ("bxaic:P",        "runs/expl_cache_bxaic_P.pt",         "bxaic_P",       "runs/scaleout/phase3_bxaic_P.json"),
]


def jaccard(gt: torch.Tensor, pred: torch.Tensor) -> float:
    gt, pred = gt.bool(), pred.bool()
    tp = int((gt & pred).sum()); fp = int((~gt & pred).sum()); fn = int((gt & ~pred).sum())
    d = tp + fp + fn
    return tp / d if d else 0.0


def topk_mask(imp: torch.Tensor, frac: float = 0.25) -> torch.Tensor:
    n = imp.numel()
    k = max(1, int(frac * n))          # == masking.py num_keep for E_i (int(), not round())
    m = torch.zeros(n, dtype=torch.bool)
    m[torch.topk(imp, k).indices] = True
    return m


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def star(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _pairtests(arru: dict) -> dict:
    raw = {}
    for a, b in PAIRS:
        d = arru[a] - arru[b]
        raw[(a, b)] = 1.0 if np.allclose(d, 0) else wilcoxon(arru[a], arru[b],
                                                             alternative="two-sided", zero_method="wilcox").pvalue
    return holm(raw)


def collect(cache_path: str, loader_key: str):
    blob = torch.load(cache_path, weights_only=False)
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    data_list, meta = LOADERS[loader_key]()

    rows = {e: dict(gea_mean=[], gea_topk=[], sz_mean=[], sz_topk=[], n=[], gt=[]) for e in EXPL}
    for k, gi in enumerate(test_idx):
        d = data_list[gi]
        gt = d.node_gt_mask.bool()
        if not bool(gt.any()):
            continue
        for e in EXPL:
            imp = torch.as_tensor(res[e][k]["node_importance"], dtype=torch.float)
            assert imp.numel() == d.num_nodes, (loader_key, gi, imp.numel(), d.num_nodes)
            pm = imp > imp.mean()
            pk = topk_mask(imp, 0.25)
            r = rows[e]
            r["gea_mean"].append(jaccard(gt, pm)); r["gea_topk"].append(jaccard(gt, pk))
            r["sz_mean"].append(int(pm.sum())); r["sz_topk"].append(int(pk.sum()))
            r["n"].append(int(d.num_nodes)); r["gt"].append(int(gt.sum()))
    return {e: {kk: np.array(vv, float) for kk, vv in rows[e].items()} for e in EXPL}


def fidplus_orders(js_path: str) -> dict:
    j = json.load(open(js_path))
    S = j["summary"]
    out = {}
    for mk, fl in (("R3", "mean"), ("R3", "mode"), ("R2", "zero"), ("R1", "hard")):
        if fl not in S.get(mk, {}):
            continue
        means = {e: S[mk][fl][e]["fid_plus"][0] for e in EXPL}
        out[f"{mk}-{fl}"] = tuple(sorted(EXPL, key=lambda e: -means[e]))
    return out


def order(arru: dict, key: str) -> tuple:
    return tuple(sorted(EXPL, key=lambda e: -arru[e][key].mean()))


def main() -> int:
    data = {}
    for label, cache, lk, js in CASES:
        data[label] = (collect(cache, lk), js)

    # ---------- 1 + 4 : GEA side by side + mask sizes ----------
    print("=" * 104)
    print(" 1+4.  GEA per explainer: mean-threshold vs top-k=0.25  |  mean predicted-mask size |M_pr|")
    print("=" * 104)
    for label, (d, _) in data.items():
        n_gea = len(d["gnnexplainer"]["gea_mean"])
        nbar = d["gnnexplainer"]["n"].mean()
        gtbar = d["gnnexplainer"]["gt"].mean()
        print(f"\n[{label}]  n_gea={n_gea}   mean N={nbar:.1f} nodes   mean |GT motif|={gtbar:.1f}")
        print(f"  {'expl':<5}{'GEA mean-thr':>14}{'GEA top-k.25':>14}{'  |':>3}"
              f"{'|M| mean-thr':>14}{'|M| top-k':>12}{'(top-k target':>15}= {int(max(1,0.25*nbar))})")
        for e in EXPL:
            r = d[e]
            print(f"  {SH[e]:<5}{r['gea_mean'].mean():>10.3f}±{r['gea_mean'].std():<4.2f}"
                  f"{r['gea_topk'].mean():>10.3f}±{r['gea_topk'].std():<4.2f}{'  |':>3}"
                  f"{r['sz_mean'].mean():>12.1f}  {r['sz_topk'].mean():>10.1f}")
        # pairwise sig under each
        for tag, key in (("mean-thr", "gea_mean"), ("top-k.25", "gea_topk")):
            arru = {e: d[e][key] for e in EXPL}
            adj = _pairtests(arru)
            om = " > ".join(SH[e] for e in sorted(EXPL, key=lambda e: -arru[e].mean()))
            sig = "  ".join(f"{SH[a]}-{SH[b]}:{star(adj[(a,b)])}" for a, b in PAIRS)
            print(f"    [{tag}] order {om:<18}  {sig}")

    # ---------- 2 : F3 / F9 verdicts ----------
    print("\n" + "=" * 104)
    print(" 2.  F3 (inversion) and F9 (per-task) under top-k=0.25 GEA")
    print("=" * 104)
    for label, (d, _) in data.items():
        arru = {e: d[e]["gea_topk"] for e in EXPL}
        adj = _pairtests(arru)
        om = " > ".join(f"{SH[e]}({arru[e].mean():.2f})" for e in sorted(EXPL, key=lambda e: -arru[e].mean()))
        sig = " , ".join(f"{SH[a]}-{SH[b]} {star(adj[(a,b)])}" for a, b in PAIRS)
        print(f"  [{label:<14}] {om:<34}   {sig}")

    # ---------- 3 : block D / F8 ----------
    print("\n" + "=" * 104)
    print(" 3.  block D -- GEA-order vs Fid+-order, GEA under mean-thr (current F8) vs top-k (matched)")
    print("=" * 104)
    for gkey, gtag in (("gea_mean", "GEA=mean-threshold  [current F8]"), ("gea_topk", "GEA=top-k.25  [matched to Fid+]")):
        agree = total = 0
        print(f"\n  --- {gtag} ---")
        for label in ("mutag_graphxai", "bxaic:indole"):
            d, js = data[label]
            go = order(d, gkey)
            fo = fidplus_orders(js)
            print(f"  [{label}]  GEA order: {' > '.join(SH[e] for e in go)}"
                  f"   ({', '.join(f'{SH[e]}={d[e][gkey].mean():.2f}' for e in go)})")
            for cond, ford in fo.items():
                ok = ford == go
                agree += ok; total += 1
                print(f"      {cond:<9} Fid+ order: {' > '.join(SH[e] for e in ford):<18} "
                      f"[{'agrees' if ok else 'DIFFERS'} with GEA]")
        print(f"  => {agree} of {total} (dataset x Fid+ condition) cells agree with the GEA order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
