"""
EXPLORATORY (registered predictions untouched). Three checks on cached
explanations -- no model, no new explanations:

1. NULL-MATCHING AUDIT. Mean-threshold mask sizes vs the top-k budget, and
   what a random mask of each explainer's OWN mean-threshold size scores
   (RandUnc|e, 20 draws) next to the shared RandNode drawn at the top-k budget.
2. MATCHED-BUDGET DIRECTION COUNTS. Top-k 0.25 GEA for GNN/PG/SX against
   RandNode (and RandEdge) drawn at the SAME budget, 20 draws per molecule,
   per-seed paired Wilcoxon + Holm over the three explainers.
3. EDGE-LEVEL GEA. PGExplainer's and GNNExplainer's NATIVE edge masks at a
   top-k 0.25 edge budget against the edge ground truth and a random-edge
   null at the same budget; plus the degree of PGExplainer's selected atoms
   on Mutagenicity and what other edge->node folding rules would give.

    python -m src.analysis.exploratory_null_audit
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..data import LOADERS
from ..metrics.gea_set import gea_max_over_set

OUT = "runs/mutagenicity/exploratory"
SEEDS = [0, 1, 2, 3, 4]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
N_DRAWS = 20
DATASETS = [
    ("mutag_graphxai", "MUTAG-GT", "runs/prio/cache_mutag_graphxai_s{s}.pt"),
    ("bxaic_indole", "B-XAIC indole", "runs/prio/cache_bxaic_indole_s{s}.pt"),
    ("bxaic_PAINS", "B-XAIC PAINS", "runs/prio/cache_bxaic_PAINS_s{s}.pt"),
    ("bxaic_X", "B-XAIC X", "runs/prio/cache_bxaic_X_s{s}.pt"),
    ("bxaic_P", "B-XAIC P", "runs/prio/cache_bxaic_P_s{s}.pt"),
    ("mutagenicity_graphxai", "Mutagenicity", "runs/mutagenicity/cache_s{s}.pt"),
]
HEADLINE = ["mutag_graphxai", "bxaic_indole", "mutagenicity_graphxai"]


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def holm(pv):
    items = sorted(pv.items(), key=lambda kv: kv[1]); out, run, m = {}, 0.0, len(items)
    for r, (k, p) in enumerate(items):
        run = min(max(run, (m - r) * p), 1.0); out[k] = run
    return out


def wpair(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)


def topk(v, k):
    m = np.zeros(len(v), bool)
    m[torch.topk(torch.as_tensor(np.asarray(v), dtype=torch.float), k).indices.numpy()] = True
    return m


def jac(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 0.0


def fold(edge_score, ei, n, how="mean"):
    out = np.zeros(n); cnt = np.zeros(n)
    if how == "max":
        out[:] = -np.inf
        for end in (0, 1):
            np.maximum.at(out, ei[end], edge_score)
        out[np.isinf(out)] = 0.0
        return out
    for end in (0, 1):
        np.add.at(out, ei[end], edge_score); np.add.at(cnt, ei[end], 1.0)
    return out if how == "sum" else out / np.maximum(cnt, 1.0)


class Scorer:
    """GEA of a node mask: plain Jaccard, or max-over-set when motifs are given."""
    def __init__(self, gt, motifs=None):
        self.gt, self.motifs = gt, motifs

    def __call__(self, mask):
        return gea_max_over_set(mask, self.motifs).max_over_set if self.motifs is not None else jac(mask, self.gt)


def rand_node_mean(scorer, n, size, rng):
    if size <= 0:
        return 0.0
    acc = 0.0
    for _ in range(N_DRAWS):
        m = np.zeros(n, bool); m[rng.choice(n, size=min(size, n), replace=False)] = True
        acc += scorer(m)
    return acc / N_DRAWS


def rand_edge_mean(scorer, n, k, ei, rng):
    acc = 0.0
    for _ in range(N_DRAWS):
        acc += scorer(topk(fold(rng.random(ei.shape[1]), ei, n), k))
    return acc / N_DRAWS


def collect(key, cache_tmpl):
    """Per seed: per-molecule arrays for checks 1-3."""
    dl, meta = LOADERS[key]()
    motifs_all = meta.extras["motifs"] if hasattr(meta, "extras") else None
    has_edge_gt = bool(meta.has_edge_gt)
    out = {}
    for s in SEEDS:
        blob = torch.load(cache_tmpl.format(s=s), weights_only=False)
        test_idx = list(blob["key"]["test_idx"]); res = blob["results"]
        rng = np.random.default_rng(30_000 + s)
        A = {k: [] for k in ["mol_idx", "n", "k", "gt", "rn_k", "re_k"]}
        for e in EXPL:
            for k in ("mean", "topk", "size", "unc"):
                A[f"{e}_{k}"] = []
        E = {k: [] for k in ["pg", "gnn", "rand", "pg_inv", "ke", "E", "gt_e"]}
        D = {k: [] for k in ["deg_top", "deg_bot", "deg_mean_sel", "deg_gt", "deg_all", "t1_top", "t1_bot", "t1_gt", "t1_all",
                             "fold_mean", "fold_sum", "fold_max", "gnn_deg_top", "gnn_t1_top"]}
        for pos, mi in enumerate(test_idx):
            d = dl[mi]; gt = d.node_gt_mask.numpy().astype(bool)
            if not gt.any():
                continue
            n = d.num_nodes; k = max(1, int(0.25 * n)); ei = d.edge_index.numpy()
            sc = Scorer(gt, motifs_all[mi] if motifs_all is not None else None)
            A["mol_idx"].append(mi); A["n"].append(n); A["k"].append(k); A["gt"].append(int(gt.sum()))
            A["rn_k"].append(rand_node_mean(sc, n, k, rng)); A["re_k"].append(rand_edge_mean(sc, n, k, ei, rng))
            imps = {}
            for e in EXPL:
                imp = np.asarray(res[e][pos]["node_importance"], dtype=np.float64); imps[e] = imp
                mm = imp > imp.mean(); size = int(mm.sum())
                A[f"{e}_mean"].append(sc(mm)); A[f"{e}_topk"].append(sc(topk(imp, k))); A[f"{e}_size"].append(size)
                A[f"{e}_unc"].append(rand_node_mean(sc, n, size, rng))
            # ---- check 3: native edge masks ----
            if has_edge_gt and getattr(d, "edge_gt_mask", None) is not None:
                egt = d.edge_gt_mask.numpy().astype(bool)
                if egt.any():
                    ne = ei.shape[1]; ke = max(1, int(0.25 * ne))
                    if motifs_all is not None:   # per-motif edge sets -> max-over-set on edges
                        em = [("m", np.nonzero(np.isin(ei[0], nodes) | np.isin(ei[1], nodes))[0].tolist()) for _, nodes in motifs_all[mi]]
                        esc = Scorer(egt, em)
                    else:
                        esc = Scorer(egt)
                    pe = np.asarray(res["pgexplainer"][pos]["edge_importance"], dtype=np.float64)
                    ge = np.asarray(res["gnnexplainer"][pos]["edge_importance"], dtype=np.float64)
                    E["pg"].append(esc(topk(pe, ke))); E["gnn"].append(esc(topk(ge, ke))); E["pg_inv"].append(esc(topk(-pe, ke)))
                    acc = 0.0
                    for _ in range(N_DRAWS):
                        m = np.zeros(ne, bool); m[rng.choice(ne, size=ke, replace=False)] = True; acc += esc(m)
                    E["rand"].append(acc / N_DRAWS); E["ke"].append(ke); E["E"].append(ne); E["gt_e"].append(int(egt.sum()))
            # ---- degree of PG's selected atoms; alternative folds ----
            deg = np.bincount(ei[0], minlength=n)
            imp = imps["pgexplainer"]; top = topk(imp, k); bot = topk(-imp, k); msel = imp > imp.mean()
            D["deg_top"].append(deg[top].mean()); D["deg_bot"].append(deg[bot].mean())
            D["deg_mean_sel"].append(deg[msel].mean() if msel.any() else np.nan)
            D["deg_gt"].append(deg[gt].mean()); D["deg_all"].append(deg.mean())
            D["t1_top"].append((deg[top] == 1).mean()); D["t1_bot"].append((deg[bot] == 1).mean())
            D["t1_gt"].append((deg[gt] == 1).mean()); D["t1_all"].append((deg == 1).mean())
            gtop = topk(imps["gnnexplainer"], k); D["gnn_deg_top"].append(deg[gtop].mean()); D["gnn_t1_top"].append((deg[gtop] == 1).mean())
            pe = np.asarray(res["pgexplainer"][pos]["edge_importance"], dtype=np.float64)
            for how in ("mean", "sum", "max"):
                D[f"fold_{how}"].append(sc(topk(fold(pe, ei, n, how), k)))
        out[s] = ({k: np.array(v, dtype=float) for k, v in A.items()}, {k: np.array(v, dtype=float) for k, v in E.items()},
                  {k: np.array(v, dtype=float) for k, v in D.items()})
        # persist the per-molecule arrays (node-level GEA + nulls, edge-level, degree/fold diagnostics)
        os.makedirs(f"{OUT}/null_audit_arrays", exist_ok=True)
        np.savez(f"{OUT}/null_audit_arrays/{key}_s{s}.npz", **out[s][0],
                 **{f"edge_{k}": v for k, v in out[s][1].items()}, **{f"diag_{k}": v for k, v in out[s][2].items()})
    return out


def sig_counts(per, real_key, null_key, family):
    """family: list of (real_key, null_key) tested together under Holm, per seed."""
    above = below = higher = 0; cells = []
    for s in SEEDS:
        A = per[s][0]
        raw = {i: wpair(A[r], A[nl]) for i, (r, nl) in enumerate(family)}
        adj = holm(raw); i = family.index((real_key, null_key))
        a, b = A[real_key].mean(), A[null_key].mean()
        higher += a > b; above += (a > b and adj[i] < 0.05); below += (a < b and adj[i] < 0.05)
        cells.append(f"{a:.3f}/{b:.3f}{star(adj[i])}")
    return int(above), int(below), int(higher), cells


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {key: collect(key, tmpl) for key, _, tmpl in DATASETS}
    name = {key: nm for key, nm, _ in DATASETS}
    rep = {"status": "EXPLORATORY", "n_draws": N_DRAWS}
    P = print

    # ------------------------------------------------------------------ check 1
    P("=" * 150); P(" CHECK 1: null-matching audit -- mask sizes and what a size-matched random mask scores (means over 5 seeds)"); P("=" * 150)
    P(f"  {'dataset':<16}{'N':>6}{'GT':>6}{'budget k':>9} |{'|M| GNN':>8}{'|M| PG':>8}{'|M| SX':>8} |{'RandNode@k':>11} |"
      f"{'Unc@|GNN|':>10}{'Unc@|PG|':>10}{'Unc@|SX|':>10} | mean-thr GEA vs OWN-size null: sig above / sig below (of 5 seeds)")
    rep["check1"] = {}
    for key, nm, _ in DATASETS:
        per = data[key]
        g = lambda k: float(np.mean([per[s][0][k].mean() for s in SEEDS]))
        fam = [(f"{e}_mean", f"{e}_unc") for e in EXPL]
        own = {SH[e]: sig_counts(per, f"{e}_mean", f"{e}_unc", fam)[:2] for e in EXPL}
        fam_old = [(f"{e}_mean", "rn_k") for e in EXPL]
        old = {SH[e]: sig_counts(per, f"{e}_mean", "rn_k", fam_old)[:2] for e in EXPL}
        P(f"  {nm:<16}{g('n'):>6.1f}{g('gt'):>6.2f}{g('k'):>9.2f} |" + "".join(f"{g(e + '_size'):>8.2f}" for e in EXPL)
          + f" |{g('rn_k'):>11.3f} |" + "".join(f"{g(e + '_unc'):>10.3f}" for e in EXPL)
          + " | " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in own.items())
          + "   [vs RandNode@k: " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in old.items()) + "]")
        rep["check1"][key] = {"n": g("n"), "gt": g("gt"), "k": g("k"), "size": {SH[e]: g(e + "_size") for e in EXPL},
                              "rand_node_at_k": g("rn_k"), "rand_unc_own_size": {SH[e]: g(e + "_unc") for e in EXPL},
                              "meanthr_vs_own_size": own, "meanthr_vs_randnode_at_k": old}

    # ------------------------------------------------------------------ check 2
    P("\n" + "=" * 150); P(" CHECK 2: matched budget -- top-k 0.25 GEA vs RandNode at the same budget (20 draws), per-seed Wilcoxon + Holm over 3 explainers"); P("=" * 150)
    rep["check2"] = {}
    for key, nm, _ in DATASETS:
        per = data[key]; fam = [(f"{e}_topk", "rn_k") for e in EXPL]
        P(f"\n  {nm}   (per seed: explainer / RandNode, Holm significance)")
        rep["check2"][key] = {}
        for e in EXPL:
            ab, be, hi, cells = sig_counts(per, f"{e}_topk", "rn_k", fam)
            P(f"    {SH[e]:<3} vs RandNode  " + "  ".join(cells) + f"   higher {hi}/5  sig above {ab}/5  sig below {be}/5")
            rep["check2"][key][SH[e]] = {"sig_above": ab, "sig_below": be, "higher": hi, "cells": cells}
        ab, be, hi, cells = sig_counts(per, "pgexplainer_topk", "re_k", [("pgexplainer_topk", "re_k")])
        P(f"    PG  vs RandEdge  " + "  ".join(cells) + f"   higher {hi}/5  sig above {ab}/5  sig below {be}/5   (single test per seed)")
        rep["check2"][key]["PG_vs_RandEdge"] = {"sig_above": ab, "sig_below": be, "higher": hi, "cells": cells}
    P("\n  3x3 summary, seeds (of 5) significantly ABOVE RandNode at the matched top-k budget")
    P(f"    {'':<16}{'GNN':>6}{'PG':>6}{'SX':>6}")
    tot_a = {SH[e]: 0 for e in EXPL}; tot_b = {SH[e]: 0 for e in EXPL}
    for key in HEADLINE:
        P(f"    {name[key]:<16}" + "".join(f"{rep['check2'][key][SH[e]]['sig_above']:>6}" for e in EXPL))
        for e in EXPL:
            tot_a[SH[e]] += rep["check2"][key][SH[e]]["sig_above"]; tot_b[SH[e]] += rep["check2"][key][SH[e]]["sig_below"]
    P(f"    {'total /15':<16}" + "".join(f"{tot_a[SH[e]]:>6}" for e in EXPL))
    P("  3x3 summary, seeds (of 5) significantly BELOW RandNode at the matched top-k budget")
    for key in HEADLINE:
        P(f"    {name[key]:<16}" + "".join(f"{rep['check2'][key][SH[e]]['sig_below']:>6}" for e in EXPL))
    P(f"    {'total /15':<16}" + "".join(f"{tot_b[SH[e]]:>6}" for e in EXPL))
    rep["check2"]["totals_above"] = tot_a; rep["check2"]["totals_below"] = tot_b

    # ------------------------------------------------------------------ check 3
    P("\n" + "=" * 150); P(" CHECK 3: edge-level GEA on NATIVE edge masks, top-k 0.25 of directed edges, vs a random-edge null at the same budget"); P("=" * 150)
    rep["check3"] = {}
    for key, nm, _ in DATASETS:
        per = data[key]
        if len(per[0][1]["pg"]) == 0:
            P(f"\n  {nm}: no edge ground truth -- skipped"); continue
        P(f"\n  {nm}   edges {np.mean([per[s][1]['E'].mean() for s in SEEDS]):.1f} (directed), edge GT {np.mean([per[s][1]['gt_e'].mean() for s in SEEDS]):.1f},"
          f" edge budget {np.mean([per[s][1]['ke'].mean() for s in SEEDS]):.1f}")
        rep["check3"][key] = {}
        for who in ("pg", "gnn"):
            ab = be = 0; cells = []
            for s in SEEDS:
                Ed = per[s][1]
                adj = holm({w: wpair(Ed[w], Ed["rand"]) for w in ("pg", "gnn")})
                a, b = Ed[who].mean(), Ed["rand"].mean()
                ab += (a > b and adj[who] < 0.05); be += (a < b and adj[who] < 0.05)
                cells.append(f"{a:.3f}/{b:.3f}{star(adj[who])}")
            node = "  ".join(f"{per[s][0][('pgexplainer' if who == 'pg' else 'gnnexplainer') + '_topk'].mean():.3f}" for s in SEEDS)
            P(f"    {who.upper():<4} edge GEA / RandEdge  " + "  ".join(cells) + f"   sig above {ab}/5  sig below {be}/5   | node-level top-k GEA: {node}")
            rep["check3"][key][who] = {"sig_above": int(ab), "sig_below": int(be), "cells": cells}
        P("    PG   inverted edge mask:  " + "  ".join(f"{per[s][1]['pg_inv'].mean():.3f}" for s in SEEDS))
        rep["check3"][key]["pg_inverted"] = [float(per[s][1]["pg_inv"].mean()) for s in SEEDS]

    P("\n  PGExplainer on Mutagenicity: degree of the atoms it selects (GT-present test molecules; degree-1 = terminal atom)")
    P(f"    {'seed':>4}{'deg top-k':>10}{'deg bottom-k':>13}{'deg mean-thr':>13}{'deg GT':>8}{'deg all':>8} |{'term% top-k':>12}{'term% bot-k':>12}{'term% GT':>9}{'term% all':>10}"
      f" |{'GNN deg top-k':>14}{'GNN term%':>10}")
    per = data["mutagenicity_graphxai"]; rep["degree"] = {}
    for s in SEEDS:
        D = per[s][2]; m = lambda k: float(np.nanmean(D[k]))
        P(f"    {s:>4}{m('deg_top'):>10.2f}{m('deg_bot'):>13.2f}{m('deg_mean_sel'):>13.2f}{m('deg_gt'):>8.2f}{m('deg_all'):>8.2f} |"
          f"{100 * m('t1_top'):>12.0f}{100 * m('t1_bot'):>12.0f}{100 * m('t1_gt'):>9.0f}{100 * m('t1_all'):>10.0f} |{m('gnn_deg_top'):>14.2f}{100 * m('gnn_t1_top'):>10.0f}")
        rep["degree"][s] = {k: m(k) for k in D}
    P("\n  PGExplainer node-level top-k GEA under three edge->node folding rules (mean = the rule in use)")
    P(f"    {'dataset':<16}" + "".join(f"{'s' + str(s) + ' mean/sum/max':>24}" for s in SEEDS))
    rep["folds"] = {}
    for key, nm, _ in DATASETS:
        per = data[key]
        P(f"    {nm:<16}" + "".join(f"{per[s][2]['fold_mean'].mean():>10.3f}/{per[s][2]['fold_sum'].mean():.3f}/{per[s][2]['fold_max'].mean():.3f}" for s in SEEDS))
        rep["folds"][key] = {s: {h: float(per[s][2][f'fold_{h}'].mean()) for h in ("mean", "sum", "max")} for s in SEEDS}

    json.dump(rep, open(f"{OUT}/null_audit.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {OUT}/null_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
