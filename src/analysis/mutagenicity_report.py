"""
STEP 4 report for the GraphXAI Mutagenicity sweep (runs/mutagenicity/).
Reads each seed's checkpoint, explanation cache and out_s<S>.json; computes
GEA (top-k 0.25 and mean-threshold; max-over-set and merged), the random
nulls, per-seed Wilcoxon + Holm (never pooled across seeds), seed-level
direction counts, the PGExplainer collapse record, mask sizes, the
toxicophore-type stratification and the oracle ceiling.

Persists per-molecule arrays by default:
    runs/mutagenicity/arrays_s<S>.npz   (GEA, nulls, win/loss, sizes, indices)
    runs/mutagenicity/report.json       (every number printed)

Nulls, 20 draws per molecule, averaged:
    rand_node  uniform k-of-N node set at the top-k budget
    rand_edge  uniform per-edge scores -> node_importance_from_edges -> top-k
    rand_conn  random CONNECTED subgraph, size-matched to the mask it is
               compared with (k under top-k; the explainer's own mask size
               under mean-threshold)
    rand_unc   uniform node set, size-matched the same way

    python -m src.analysis.mutagenicity_report
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from ..data import LOADERS
from ..explain.common import node_importance_from_edges
from ..metrics.gea_set import gea_max_over_set
from .connectivity_baseline import build_adj, random_connected_subset
from .oracle_ceiling import budget, ceil_set, ceil_single

R = "runs/mutagenicity"
SEEDS = [0, 1, 2, 3, 4]
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
N_DRAWS = 20
VARIANTS = ["set", "merged"]          # max-over-set, merged-mask


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def holm(pv: dict) -> dict:
    items = sorted(pv.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for rank, (k, p) in enumerate(items):
        running = min(max(running, (m - rank) * p), 1.0)
        out[k] = running
    return out


def wpair(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)


def mask_from_nodes(nodes, n):
    m = np.zeros(n, bool); m[list(nodes)] = True
    return m


def topk_mask(imp: torch.Tensor, k: int) -> np.ndarray:
    m = np.zeros(imp.numel(), bool); m[torch.topk(imp, k).indices.numpy()] = True
    return m


def both(pred, motifs):
    g = gea_max_over_set(pred, motifs)
    return g.max_over_set, g.merged


def null_draws(kind, size, d, adj, motifs, rng):
    """Mean (max-over-set, merged) GEA over N_DRAWS random masks of ``size`` atoms."""
    n = d.num_nodes
    acc = np.zeros(2)
    for _ in range(N_DRAWS):
        if kind == "rand_node" or kind == "rand_unc":
            m = mask_from_nodes(rng.choice(n, size=min(size, n), replace=False), n)
        elif kind == "rand_conn":
            m = mask_from_nodes(random_connected_subset(adj, n, size, rng), n)
        elif kind == "rand_edge":
            es = torch.as_tensor(rng.random(d.edge_index.shape[1]), dtype=torch.float)
            m = topk_mask(node_importance_from_edges(es, d.edge_index, n), size)
        acc += np.array(both(m, motifs)) if size > 0 else 0.0
    return acc / N_DRAWS


def analyse_seed(s, dl, motifs_all):
    ck = torch.load(f"{R}/ckpt_s{s}.pt", map_location="cpu", weights_only=False)
    blob = torch.load(f"{R}/cache_s{s}.pt", weights_only=False)
    out = json.load(open(f"{R}/out_s{s}.json"))
    test_idx = list(blob["key"]["test_idx"])
    res = blob["results"]
    rng = np.random.default_rng(20_000 + s)

    A = {"mol_idx": [], "n_nodes": [], "gt_size": [], "n_motifs": [], "halide_only": [], "budget": [],
         "ceiling_set": [], "ceiling_merged": []}
    for e in EXPL:
        for b in ("topk", "mean"):
            for v in VARIANTS:
                A[f"{e}_{b}_{v}"] = []
        A[f"{e}_mean_size"] = []
        for nk in ("rand_conn", "rand_unc"):
            for v in VARIANTS:
                A[f"{nk}_mean_{e}_{v}"] = []
    for nk in ("rand_node", "rand_edge", "rand_conn"):
        for v in VARIANTS:
            A[f"{nk}_topk_{v}"] = []

    for pos_in_test, mi in enumerate(test_idx):
        d = dl[mi]
        if not bool(d.node_gt_mask.any()):
            continue
        motifs = motifs_all[mi]; n = d.num_nodes; k = budget(n)
        adj = build_adj(d.edge_index, n)
        A["mol_idx"].append(mi); A["n_nodes"].append(n); A["gt_size"].append(int(d.node_gt_mask.sum()))
        A["n_motifs"].append(len(motifs)); A["budget"].append(k)
        A["halide_only"].append(all(t == "halide" for t, _ in motifs))
        A["ceiling_set"].append(ceil_set(motifs, n)); A["ceiling_merged"].append(ceil_single(int(d.node_gt_mask.sum()), k))
        for nk in ("rand_node", "rand_edge", "rand_conn"):
            sv, mv = null_draws(nk, k, d, adj, motifs, rng)
            A[f"{nk}_topk_set"].append(sv); A[f"{nk}_topk_merged"].append(mv)
        for e in EXPL:
            imp = torch.as_tensor(res[e][pos_in_test]["node_importance"], dtype=torch.float)
            sv, mv = both(topk_mask(imp, k), motifs)
            A[f"{e}_topk_set"].append(sv); A[f"{e}_topk_merged"].append(mv)
            mm = (imp > imp.mean()).numpy(); size = int(mm.sum())
            sv, mv = both(mm, motifs)
            A[f"{e}_mean_set"].append(sv); A[f"{e}_mean_merged"].append(mv); A[f"{e}_mean_size"].append(size)
            for nk in ("rand_conn", "rand_unc"):
                sv, mv = null_draws(nk, size, d, adj, motifs, rng) if size > 0 else (0.0, 0.0)
                A[f"{nk}_mean_{e}_set"].append(sv); A[f"{nk}_mean_{e}_merged"].append(mv)

    A = {k: np.array(v) for k, v in A.items()}
    for e in EXPL:   # win/loss vs the size-matched connected null, per molecule
        for v in VARIANTS:
            A[f"{e}_mean_{v}_win_vs_rand_conn"] = A[f"{e}_mean_{v}"] > A[f"rand_conn_mean_{e}_{v}"]
            A[f"{e}_topk_{v}_win_vs_rand_conn"] = A[f"{e}_topk_{v}"] > A[f"rand_conn_topk_{v}"]
    split_of = {i: name for name, idxs in ck["split"].items() for i in idxs}
    np.savez(f"{R}/arrays_s{s}.npz", model_seed=s, explainer_seed=1000 + s,
             explained_idx=np.array(test_idx), explained_split=np.array([split_of[i] for i in test_idx]),
             split_train=np.array(ck["split"]["train"]), split_val=np.array(ck["split"]["val"]),
             split_test=np.array(ck["split"]["test"]), **A)
    return A, ck, out


def comparisons(binar):
    """(explainer, null-array-key-template, label) for one binarization."""
    c = []
    for e in EXPL:
        if binar == "topk":
            c += [(e, "rand_node_topk_{v}", "RandNode"), (e, "rand_edge_topk_{v}", "RandEdge"),
                  (e, "rand_conn_topk_{v}", "RandConn")]
        else:
            c += [(e, "rand_node_topk_{v}", "RandNode"), (e, "rand_edge_topk_{v}", "RandEdge"),
                  (e, f"rand_conn_mean_{e}_{{v}}", "RandConn"), (e, f"rand_unc_mean_{e}_{{v}}", "RandUnc")]
    return c


def tests_for(A, binar, v, sel=None):
    sel = np.ones(len(A["mol_idx"]), bool) if sel is None else sel
    raw, info = {}, {}
    for e, tmpl, lab in comparisons(binar):
        a = A[f"{e}_{binar}_{v}"][sel]; b = A[tmpl.format(v=v)][sel]
        key = f"{SH[e]} vs {lab}"
        raw[key] = wpair(a, b) if len(a) > 5 else 1.0
        info[key] = (float(a.mean()), float(b.mean()), int((a > b).sum()), int((a < b).sum()), int(len(a)))
    adj = holm(raw)
    return {k: {"real": info[k][0], "null": info[k][1], "wins": info[k][2], "losses": info[k][3], "n": info[k][4],
                "p_holm": adj[k], "higher": info[k][0] > info[k][1],
                "clears": bool(info[k][0] > info[k][1] and adj[k] < 0.05),
                "sig_below": bool(info[k][0] < info[k][1] and adj[k] < 0.05)} for k in raw}


def main():
    dl, meta = LOADERS["mutagenicity_graphxai"]()
    motifs_all = meta.extras["motifs"]
    seeds = [s for s in SEEDS if os.path.exists(f"{R}/out_s{s}.json")]
    per = {s: analyse_seed(s, dl, motifs_all) for s in seeds}
    rep = {"seeds": seeds, "n_draws": N_DRAWS}
    P = print

    P("=" * 118); P(" GraphXAI Mutagenicity -- decoupled protocol (model seed s, explainer seed 1000+s), test split only"); P("=" * 118)
    P("\n[1] MODEL (D-MPNN)   seed  test_acc  test_auroc  | explained (<=80 nodes)  GT-present")
    acc, au = [], []
    rep["model"] = {}
    for s in seeds:
        A, ck, out = per[s]
        acc.append(ck["metrics"]["test_acc"]); au.append(ck["metrics"]["test_auroc"])
        P(f"                     {s:>4}  {acc[-1]:.4f}    {au[-1]:.4f}      | {out['config']['n_test']:>5}"
          f"                 {len(A['mol_idx']):>5}")
        rep["model"][s] = {"test_acc": acc[-1], "test_auroc": au[-1], "n_explained": out["config"]["n_test"],
                           "n_gt_present": int(len(A["mol_idx"]))}
    P(f"                     mean  {np.mean(acc):.4f}    {np.mean(au):.4f}   (std {np.std(acc):.4f} / {np.std(au):.4f})")

    rep["gea"] = {}
    for binar, bname in (("mean", "mean-threshold"), ("topk", "top-k 0.25")):
        for v, vname in (("set", "max-over-set"), ("merged", "merged mask")):
            P(f"\n[2] GEA, {bname}, {vname} -- per-seed means over GT-present test molecules")
            cols = [(f"{e}_{binar}_{v}", SH[e]) for e in EXPL]
            if binar == "topk":
                cols += [(f"rand_node_topk_{v}", "RandNode"), (f"rand_edge_topk_{v}", "RandEdge"), (f"rand_conn_topk_{v}", "RandConn")]
            else:
                cols += [(f"rand_node_topk_{v}", "RandNode"), (f"rand_edge_topk_{v}", "RandEdge")]
                cols += [(f"rand_conn_mean_{e}_{v}", f"Conn|{SH[e]}") for e in EXPL]
                cols += [(f"rand_unc_mean_{e}_{v}", f"Unc|{SH[e]}") for e in EXPL]
            P(f"  {'seed':>4}" + "".join(f"{lab:>10}" for _, lab in cols) + (f"{'ceiling':>10}" if binar == "topk" else ""))
            table = {}
            for s in seeds:
                A = per[s][0]
                vals = [float(A[k].mean()) for k, _ in cols]
                table[s] = dict(zip([lab for _, lab in cols], vals))
                extra = f"{A['ceiling_' + v].mean():>10.3f}" if binar == "topk" else ""
                P(f"  {s:>4}" + "".join(f"{x:>10.3f}" for x in vals) + extra)
            allv = np.array([[table[s][lab] for _, lab in cols] for s in seeds])
            P(f"  {'mean':>4}" + "".join(f"{x:>10.3f}" for x in allv.mean(0)))
            P(f"  {'std':>4}" + "".join(f"{x:>10.3f}" for x in allv.std(0)))
            rep["gea"][f"{binar}_{v}"] = {"per_seed": table,
                                         "mean": dict(zip([lab for _, lab in cols], allv.mean(0).tolist())),
                                         "std": dict(zip([lab for _, lab in cols], allv.std(0).tolist()))}

    rep["tests"] = {}
    for binar, bname in (("mean", "mean-threshold"), ("topk", "top-k 0.25")):
        for v, vname in (("set", "max-over-set"), ("merged", "merged mask")):
            P(f"\n[3] explainer vs null, {bname}, {vname}: per-seed paired Wilcoxon, Holm within seed (never pooled)")
            T = {s: tests_for(per[s][0], binar, v) for s in seeds}
            keys = list(T[seeds[0]].keys())
            P(f"  {'comparison':<16}" + "".join(f"{'seed ' + str(s):>22}" for s in seeds) + "   higher  clears  sig.below")
            for k in keys:
                cells = "".join(f"{T[s][k]['real']:>8.3f}/{T[s][k]['null']:<6.3f}{star(T[s][k]['p_holm']):>4}"
                                f"{'+' if T[s][k]['higher'] else '-':>3}" for s in seeds)
                P(f"  {k:<16}{cells}   {sum(T[s][k]['higher'] for s in seeds)}/{len(seeds)}"
                  f"     {sum(T[s][k]['clears'] for s in seeds)}/{len(seeds)}     {sum(T[s][k]['sig_below'] for s in seeds)}/{len(seeds)}")
            rep["tests"][f"{binar}_{v}"] = {str(s): T[s] for s in seeds}

    P("\n[4] PREDICTIONS (scoring rules fixed in the spec before the results were read)")
    Tm = {s: tests_for(per[s][0], "mean", "set") for s in seeds}
    p1 = sum(Tm[s]["GNN vs RandNode"]["clears"] for s in seeds)
    p2 = sum(Tm[s]["SX vs RandConn"]["clears"] for s in seeds)
    P(f"  P1  GNNExplainer clears RandNode in >= 4/5 seeds:  observed {p1}/{len(seeds)}  -> {'HOLDS' if p1 >= 4 else 'FAILS'}")
    P(f"  P2  SubgraphX beats RandConn in >= 4/5 seeds:      observed {p2}/{len(seeds)}  -> {'HOLDS' if p2 >= 4 else 'FAILS'}")
    rep["predictions"] = {"p1_gnn_clears_randnode": p1, "p2_sx_beats_randconn": p2, "n_seeds": len(seeds)}

    P("\n[5] PGExplainer collapse (strict: n_degenerate/n >= 0.5) and explainer health, per seed")
    rep["health"] = {}
    for s in seeds:
        h = per[s][2]["explainer_health"]
        P(f"  seed {s}: " + "   ".join(f"{SH[e]} degenerate {h[e]['n_degenerate']}/{h[e]['n']} empty {h[e]['n_empty']} whole {h[e]['n_whole']}"
                                      f"{'  COLLAPSED' if h[e]['n_degenerate'] / max(h[e]['n'], 1) >= 0.5 else ''}" for e in EXPL))
        rep["health"][s] = {e: {**h[e], "collapsed": h[e]["n_degenerate"] / max(h[e]["n"], 1) >= 0.5} for e in EXPL}
    npg = sum(rep["health"][s]["pgexplainer"]["collapsed"] for s in seeds)
    P(f"  PGExplainer collapsed in {npg}/{len(seeds)} seeds (P3: reported as observed)")
    rep["predictions"]["p3_pg_collapsed_seeds"] = npg

    P("\n[6] MASK SIZES (GT-present test molecules): mean-threshold mask size vs the top-k budget vs GT size")
    P(f"  {'seed':>4}{'N':>8}{'GT atoms':>10}{'motifs':>8}{'budget k':>10}" + "".join(f"{'|M| ' + SH[e]:>10}" for e in EXPL))
    rep["sizes"] = {}
    for s in seeds:
        A = per[s][0]
        row = [A["n_nodes"].mean(), A["gt_size"].mean(), A["n_motifs"].mean(), A["budget"].mean()] + [A[f"{e}_mean_size"].mean() for e in EXPL]
        P(f"  {s:>4}" + f"{row[0]:>8.1f}{row[1]:>10.2f}{row[2]:>8.2f}{row[3]:>10.2f}" + "".join(f"{x:>10.2f}" for x in row[4:]))
        rep["sizes"][s] = dict(zip(["n_nodes", "gt_size", "n_motifs", "budget", "GNN", "PG", "SX"], [float(x) for x in row]))

    P("\n[7] STRATIFIED BY TOXICOPHORE TYPE (max-over-set GEA): halide-only graphs vs graphs with NH2/NO2/azo/nitroso")
    rep["stratified"] = {}
    for binar, bname in (("mean", "mean-threshold"), ("topk", "top-k 0.25")):
        for lab, want in (("halide-only", True), ("NH2/NO2/azo/nitroso", False)):
            P(f"  -- {bname}, {lab}")
            T = {s: tests_for(per[s][0], binar, "set", per[s][0]["halide_only"] == want) for s in seeds}
            ns = [int((per[s][0]["halide_only"] == want).sum()) for s in seeds]
            gt = np.mean([per[s][0]["gt_size"][per[s][0]["halide_only"] == want].mean() for s in seeds])
            ce = np.mean([per[s][0]["ceiling_set"][per[s][0]["halide_only"] == want].mean() for s in seeds])
            P(f"     n per seed {ns}   GT size {gt:.2f} atoms" + (f"   top-k ceiling {ce:.3f}" if binar == "topk" else ""))
            keys = [k for k in T[seeds[0]] if ("RandNode" in k or "RandConn" in k)]
            for k in keys:
                real = np.mean([T[s][k]["real"] for s in seeds]); null = np.mean([T[s][k]["null"] for s in seeds])
                P(f"     {k:<16} mean {real:.3f} vs {null:.3f}   per-seed: "
                  + " ".join(f"{T[s][k]['real']:.3f}/{T[s][k]['null']:.3f}{star(T[s][k]['p_holm'])}" for s in seeds)
                  + f"   clears {sum(T[s][k]['clears'] for s in seeds)}/{len(seeds)}  sig.below {sum(T[s][k]['sig_below'] for s in seeds)}/{len(seeds)}")
            rep["stratified"][f"{binar}|{lab}"] = {"n_per_seed": ns, "gt_size": float(gt), "ceiling": float(ce),
                                                   "tests": {str(s): T[s] for s in seeds}}

    P("\n[8] Fid+ / Fid- / GEF on the explained test molecules (per-seed means; from out_s<S>.json)")
    rep["fidelity"] = {}
    for m, fill in (("R3", "mean"), ("R2", "zero"), ("R1", "hard")):
        P(f"  -- {m} ({fill})")
        for e in EXPL:
            rows = [[per[s][2]["summary"][m][fill][e][k][0] for k in ("fid_plus", "fid_minus", "gef")] for s in seeds]
            a = np.array(rows)
            P(f"     {SH[e]:<4} Fid+ " + " ".join(f"{x:.3f}" for x in a[:, 0]) + f"  (mean {a[:, 0].mean():.3f})"
              + "   Fid- " + " ".join(f"{x:.3f}" for x in a[:, 1]) + f"  (mean {a[:, 1].mean():.3f})"
              + "   GEF " + " ".join(f"{x:.3f}" for x in a[:, 2]) + f"  (mean {a[:, 2].mean():.3f})")
            rep["fidelity"][f"{m}|{e}"] = {"fid_plus": a[:, 0].tolist(), "fid_minus": a[:, 1].tolist(), "gef": a[:, 2].tolist()}

    json.dump(rep, open(f"{R}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {R}/report.json and {R}/arrays_s<S>.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
