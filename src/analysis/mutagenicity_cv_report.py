"""
Report for the full-set Mutagenicity 5-fold CV run (runs/mutagenicity_cv/).
Each graph is explained once, by a model that never saw it.

Per-molecule arrays are computed by the same code as the 5-seed test-split
report (mutagenicity_report.analyse_seed) and saved to
runs/mutagenicity_cv/arrays_s<fold>.npz (mol_idx per fold included).

    python -m src.analysis.mutagenicity_cv_report
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from ..data import LOADERS
from . import mutagenicity_report as mr
from .mutagenicity_report import EXPL, SH, holm, star, wpair

R = "runs/mutagenicity_cv"
R_SEED = "runs/mutagenicity"
FOLDS = [0, 1, 2, 3, 4]
TYPES = ["NH2", "NO2", "halide", "nitroso", "azo"]
# the four registered comparisons (Holm family within a fold): (label, real key, null key)
FAMILY = [("GNN vs RandNode", "gnnexplainer_topk_set", "rand_node_topk_set"),
          ("PG vs RandNode", "pgexplainer_topk_set", "rand_node_topk_set"),
          ("SX vs RandNode", "subgraphx_topk_set", "rand_node_topk_set"),
          ("SX vs RandConn", "subgraphx_mean_set", "rand_conn_mean_subgraphx_set")]
EXTRA = [("SX top-k vs RandConn@k", "subgraphx_topk_set", "rand_conn_topk_set"),
         ("PG vs RandEdge", "pgexplainer_topk_set", "rand_edge_topk_set"),
         ("GNN mean-thr vs RandUnc|GNN", "gnnexplainer_mean_set", "rand_unc_mean_gnnexplainer_set"),
         ("PG mean-thr vs RandUnc|PG", "pgexplainer_mean_set", "rand_unc_mean_pgexplainer_set"),
         ("SX mean-thr vs RandUnc|SX", "subgraphx_mean_set", "rand_unc_mean_subgraphx_set")]


def family_tests(A, family):
    raw = {lab: wpair(A[r], A[nl]) for lab, r, nl in family}
    adj = holm(raw); out = {}
    for lab, r, nl in family:
        a, b = float(A[r].mean()), float(A[nl].mean())
        out[lab] = {"real": a, "null": b, "p_holm": adj[lab], "n": int(len(A[r])),
                    "wins": int((A[r] > A[nl]).sum()), "losses": int((A[r] < A[nl]).sum()),
                    "above": bool(a > b and adj[lab] < 0.05), "below": bool(a < b and adj[lab] < 0.05), "higher": bool(a > b)}
    return out


def table(P, units, T, family, unit_name):
    P(f"  {'comparison':<28}" + "".join(f"{unit_name + ' ' + str(u):>22}" for u in units) + "   higher  sig.above  sig.below")
    for lab, _, _ in family:
        cells = "".join(f"{T[u][lab]['real']:>9.3f}/{T[u][lab]['null']:<6.3f}{star(T[u][lab]['p_holm']):>4}  " for u in units)
        P(f"  {lab:<28}{cells}   {sum(T[u][lab]['higher'] for u in units)}/{len(units)}"
          f"      {sum(T[u][lab]['above'] for u in units)}/{len(units)}        {sum(T[u][lab]['below'] for u in units)}/{len(units)}")


def main():
    dl, meta = LOADERS["mutagenicity_graphxai"]()
    motifs_all = meta.extras["motifs"]
    folds = [f for f in FOLDS if os.path.exists(f"{R}/out_s{f}.json")]
    mr.R = R                                   # reuse the test-split analysis code on the CV outputs
    per = {f: mr.analyse_seed(f, dl, motifs_all) for f in folds}
    rep = {"folds": folds}
    P = print
    P("=" * 124); P(" GraphXAI Mutagenicity, full set, 5-fold CV -- every graph explained once by a model that never saw it"); P("=" * 124)

    # ---- 1. models and counts ----
    P("\n[1] MODEL per fold, and what was explained")
    P(f"  {'fold':>4}{'test_acc':>10}{'test_auroc':>12}{'fold size':>11}{'explained (<=80 atoms)':>24}{'GT-bearing':>12}{'wall-clock':>12}")
    wall = {}
    for f in folds:
        m = re.findall(r"wall_clock_seconds=(\d+)", open(f"{R}/logs/f{f}.log", errors="replace").read())
        wall[f] = int(m[-1]) if m else None
    tot_e = tot_g = 0; rep["model"] = {}
    for f in folds:
        A, ck, out = per[f]
        ne, ng = out["config"]["n_test"], len(A["mol_idx"]); tot_e += ne; tot_g += ng
        w = f"{wall[f] / 3600:.2f} h" if wall[f] else "-"
        P(f"  {f:>4}{ck['metrics']['test_acc']:>10.4f}{ck['metrics']['test_auroc']:>12.4f}{len(ck['split']['test']):>11}{ne:>24}{ng:>12}{w:>12}")
        rep["model"][f] = {"test_acc": ck["metrics"]["test_acc"], "test_auroc": ck["metrics"]["test_auroc"],
                           "fold_size": len(ck["split"]["test"]), "n_explained": ne, "n_gt_bearing": ng, "wall_clock_s": wall[f]}
    acc = [rep["model"][f]["test_acc"] for f in folds]; au = [rep["model"][f]["test_auroc"] for f in folds]
    P(f"  {'mean':>4}{np.mean(acc):>10.4f}{np.mean(au):>12.4f}{'':>11}{'total ' + str(tot_e):>24}{'total ' + str(tot_g):>12}")
    all_idx = np.concatenate([per[f][0]["mol_idx"] for f in folds])
    P(f"  unique GT-bearing molecules across folds: {len(np.unique(all_idx))} (each appears once: {len(np.unique(all_idx)) == len(all_idx)})")
    rep["totals"] = {"explained": tot_e, "gt_bearing": tot_g}

    pooled = {k: np.concatenate([per[f][0][k] for f in folds]) for k in per[folds[0]][0] if per[folds[0]][0][k].ndim == 1}

    # ---- 2. GEA tables ----
    rep["gea"] = {}
    for binar, bname in (("topk", "top-k 0.25 (primary)"), ("mean", "mean-threshold")):
        for v, vname in (("set", "max-over-set"), ("merged", "merged mask")):
            cols = [(f"{e}_{binar}_{v}", SH[e]) for e in EXPL]
            cols += [(f"rand_node_topk_{v}", "RandNode"), (f"rand_edge_topk_{v}", "RandEdge")]
            cols += [(f"rand_conn_topk_{v}", "RandConn@k")] if binar == "topk" else [(f"rand_conn_mean_subgraphx_{v}", "RandConn|SX")]
            P(f"\n[2] GEA, {bname}, {vname}: per-fold means over GT-bearing molecules")
            P(f"  {'fold':>10}" + "".join(f"{lab:>12}" for _, lab in cols) + (f"{'ceiling':>10}" if binar == "topk" else ""))
            tab = {}
            for f in folds:
                A = per[f][0]; vals = [float(A[k].mean()) for k, _ in cols]; tab[f] = dict(zip([l for _, l in cols], vals))
                P(f"  {f:>10}" + "".join(f"{x:>12.3f}" for x in vals) + (f"{A['ceiling_' + v].mean():>10.3f}" if binar == "topk" else ""))
            arr = np.array([[tab[f][l] for _, l in cols] for f in folds])
            P(f"  {'fold mean':>10}" + "".join(f"{x:>12.3f}" for x in arr.mean(0)))
            P(f"  {'fold std':>10}" + "".join(f"{x:>12.3f}" for x in arr.std(0)))
            pm = [float(pooled[k].mean()) for k, _ in cols]
            P(f"  {'pooled':>10}" + "".join(f"{x:>12.3f}" for x in pm) + (f"{pooled['ceiling_' + v].mean():>10.3f}" if binar == "topk" else "")
              + f"   (all {len(all_idx)} unique molecules)")
            rep["gea"][f"{binar}_{v}"] = {"per_fold": tab, "pooled": dict(zip([l for _, l in cols], pm))}

    # ---- 3. tests ----
    P("\n[3] per-fold paired Wilcoxon, Holm within fold over the four registered comparisons (cells: explainer/null, significance)")
    T = {f: family_tests(per[f][0], FAMILY) for f in folds}
    table(P, folds, T, FAMILY, "fold")
    P("\n    additional comparisons (reported, not part of the registered family; Holm within fold over these five)")
    TX = {f: family_tests(per[f][0], EXTRA) for f in folds}
    table(P, folds, TX, EXTRA, "fold")
    P(f"\n    POOLED test, one per comparison, over all {len(all_idx)} unique molecules (Holm over the four)."
      "\n    Each molecule appears once, but the explained model differs by fold, so this is not a single-model test.")
    TP = family_tests(pooled, FAMILY)
    for lab, _, _ in FAMILY:
        t = TP[lab]
        P(f"      {lab:<18} {t['real']:.3f} vs {t['null']:.3f}   p_Holm {t['p_holm']:.2e} {star(t['p_holm'])}   molecule wins {t['wins']} / losses {t['losses']}"
          f" / ties {t['n'] - t['wins'] - t['losses']}   -> {'ABOVE' if t['above'] else 'BELOW' if t['below'] else 'n.s.'}")
    rep["tests"] = {"per_fold": {str(f): T[f] for f in folds}, "per_fold_extra": {str(f): TX[f] for f in folds}, "pooled": TP}

    P("\n[3b] REGISTERED PREDICTIONS (rules fixed before the run)")
    a = sum(T[f]["GNN vs RandNode"]["above"] for f in folds); b = sum(T[f]["SX vs RandConn"]["above"] for f in folds)
    c = sum(T[f]["PG vs RandNode"]["above"] for f in folds)
    P(f"  (a) GNNExplainer beats RandNode in >= 4/5 folds:            observed {a}/{len(folds)} -> {'HELD' if a >= 4 else 'DID NOT HOLD'}")
    P(f"  (b) SubgraphX beats its connectivity-matched null >= 4/5:   observed {b}/{len(folds)} -> {'HELD' if b >= 4 else 'DID NOT HOLD'}")
    P(f"  (c) PGExplainer not reliably above RandNode (< 4/5 folds):  significantly above in {c}/{len(folds)},"
      f" below in {sum(T[f]['PG vs RandNode']['below'] for f in folds)}/{len(folds)} -> {'HELD' if c < 4 else 'DID NOT HOLD'}")
    rep["predictions"] = {"a_gnn_above_randnode": a, "b_sx_above_randconn": b, "c_pg_above_randnode": c,
                          "a_held": a >= 4, "b_held": b >= 4, "c_held": c < 4}

    # ---- 4. collapse ----
    P("\n[4] explainer health per fold (strict collapse rule: n_degenerate/n >= 0.5)")
    rep["health"] = {}
    for f in folds:
        h = per[f][2]["explainer_health"]
        P(f"  fold {f}: " + "   ".join(f"{SH[e]} degenerate {h[e]['n_degenerate']}/{h[e]['n']} empty {h[e]['n_empty']} whole {h[e]['n_whole']}"
                                      f"{'  COLLAPSED' if h[e]['n_degenerate'] / max(h[e]['n'], 1) >= 0.5 else ''}" for e in EXPL))
        rep["health"][f] = {e: {**h[e], "collapsed": h[e]["n_degenerate"] / max(h[e]["n"], 1) >= 0.5} for e in EXPL}
    P(f"  PGExplainer collapsed in {sum(rep['health'][f]['pgexplainer']['collapsed'] for f in folds)}/{len(folds)} folds")

    # ---- 5. per toxicophore type (exploratory) ----
    P("\n[5] EXPLORATORY: GEA by toxicophore type, pooled over all unique GT-bearing molecules (max-over-set)."
      "\n    A graph is counted under every type it contains, so the rows overlap.")
    P(f"  {'type':<9}{'graphs':>7}{'GT atoms':>9} | top-k:{'GNN':>8}{'PG':>8}{'SX':>8}{'RandNode':>10}{'ceiling':>9} | mean-thr:{'GNN':>8}{'PG':>8}{'SX':>8}{'RandConn|SX':>13}")
    types_of = [set(t for t, _ in motifs_all[int(i)]) for i in pooled["mol_idx"]]
    rep["by_type"] = {}
    for t in TYPES:
        sel = np.array([t in s for s in types_of])
        if sel.sum() == 0:
            continue
        g = lambda k: float(pooled[k][sel].mean())
        P(f"  {t:<9}{int(sel.sum()):>7}{g('gt_size'):>9.2f} |       {g('gnnexplainer_topk_set'):>8.3f}{g('pgexplainer_topk_set'):>8.3f}{g('subgraphx_topk_set'):>8.3f}"
          f"{g('rand_node_topk_set'):>10.3f}{g('ceiling_set'):>9.3f} |          {g('gnnexplainer_mean_set'):>8.3f}{g('pgexplainer_mean_set'):>8.3f}"
          f"{g('subgraphx_mean_set'):>8.3f}{g('rand_conn_mean_subgraphx_set'):>13.3f}")
        rep["by_type"][t] = {"n": int(sel.sum()), **{k: g(k) for k in ("gt_size", "gnnexplainer_topk_set", "pgexplainer_topk_set",
                             "subgraphx_topk_set", "rand_node_topk_set", "ceiling_set", "gnnexplainer_mean_set",
                             "pgexplainer_mean_set", "subgraphx_mean_set", "rand_conn_mean_subgraphx_set")}}

    # ---- 6. against the 5-seed test-split run ----
    P("\n[6] against the 5-seed test-split run (runs/mutagenicity/, same four comparisons, same Holm family)")
    seeds = [s for s in range(5) if os.path.exists(f"{R_SEED}/arrays_s{s}.npz")]
    TS = {s: family_tests(dict(np.load(f"{R_SEED}/arrays_s{s}.npz")), FAMILY) for s in seeds}
    P(f"  {'comparison':<18}{'5-seed: above/below':>22}{'mean real/null':>18}   |{'5-fold: above/below':>22}{'mean real/null':>18}   same direction?")
    rep["vs_seed_run"] = {}
    for lab, _, _ in FAMILY:
        sa = sum(TS[s][lab]["above"] for s in seeds); sb = sum(TS[s][lab]["below"] for s in seeds)
        fa = sum(T[f][lab]["above"] for f in folds); fb = sum(T[f][lab]["below"] for f in folds)
        sm = (np.mean([TS[s][lab]["real"] for s in seeds]), np.mean([TS[s][lab]["null"] for s in seeds]))
        fm = (np.mean([T[f][lab]["real"] for f in folds]), np.mean([T[f][lab]["null"] for f in folds]))
        same = (sa, sb) == (fa, fb)
        P(f"  {lab:<18}{str(sa) + '/' + str(len(seeds)) + ' , ' + str(sb) + '/' + str(len(seeds)):>22}{sm[0]:>11.3f}/{sm[1]:.3f}   |"
          f"{str(fa) + '/' + str(len(folds)) + ' , ' + str(fb) + '/' + str(len(folds)):>22}{fm[0]:>11.3f}/{fm[1]:.3f}   {'same counts' if same else 'DIFFERS'}")
        rep["vs_seed_run"][lab] = {"seed_above": sa, "seed_below": sb, "fold_above": fa, "fold_below": fb, "same_counts": same}
    P("  per-unit direction (+ sig above, - sig below, . n.s.):")
    for lab, _, _ in FAMILY:
        sg = lambda t: "+" if t["above"] else "-" if t["below"] else "."
        P(f"    {lab:<18} seeds 0-4: {' '.join(sg(TS[s][lab]) for s in seeds)}    folds 0-4: {' '.join(sg(T[f][lab]) for f in folds)}")

    # ---- 7. fidelity ----
    P("\n[7] Fid+ / Fid- / GEF on all explained molecules, per-fold means (R3 mean fill, R2, R1)")
    rep["fidelity"] = {}
    for m, fill in (("R3", "mean"), ("R2", "zero"), ("R1", "hard")):
        P(f"  -- {m} ({fill})")
        for e in EXPL:
            a_ = np.array([[per[f][2]["summary"][m][fill][e][k][0] for k in ("fid_plus", "fid_minus", "gef")] for f in folds])
            P(f"     {SH[e]:<4} Fid+ " + " ".join(f"{x:.3f}" for x in a_[:, 0]) + f"  (mean {a_[:, 0].mean():.3f})"
              + "   Fid- " + " ".join(f"{x:.3f}" for x in a_[:, 1]) + f"  (mean {a_[:, 1].mean():.3f})"
              + "   GEF " + " ".join(f"{x:.3f}" for x in a_[:, 2]) + f"  (mean {a_[:, 2].mean():.3f})")
            rep["fidelity"][f"{m}|{e}"] = {"fid_plus": a_[:, 0].tolist(), "fid_minus": a_[:, 1].tolist(), "gef": a_[:, 2].tolist()}

    json.dump(rep, open(f"{R}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {R}/report.json and {R}/arrays_s<fold>.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
