"""
RQ1 final export: tables (a)-(f) under the random-tie rule, old | random-tie | change.
Reads stored arrays and the random-tie arrays (Phase 2b + runs/rq1_final_export/gea);
writes CSV + plain text to runs/rq1_final_export/tables/. Plan: runs/rq1_final_export/PLAN.md.

    python -m src.analysis.final_export_tables
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from . import exploratory_null_audit as na
from .mutagenicity_cv_report import FAMILY, family_tests
from .mutagenicity_report import wpair
from .mutagenicity_cv_report import rank_direction

OUT = "runs/rq1_final_export"
T = f"{OUT}/tables"
AUD = "runs/mutagenicity/exploratory/null_audit_arrays"
P2B = "runs/rq1_followups/phase2b/gea"
P1A = "runs/rq1_followups/phase1/null_audit_arrays"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNNExplainer", "pgexplainer": "PGExplainer", "subgraphx": "SubgraphX"}
ARCH = {"D-MPNN": "runs/mutagenicity", "GIN": "runs/arch_gin/mutagenicity_graphxai", "GINE": "runs/arch_gine/mutagenicity_graphxai",
        "GCN": "runs/arch_gcn/mutagenicity_graphxai"}
CEILING = {"MUTAG-GT": 0.751, "B-XAIC indole": 0.724, "Mutagenicity D-MPNN": 0.551}
S04, S59 = [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
LOG = []


def P(s=""):
    print(s); LOG.append(s)


def flag(old, new):
    return "*" if abs(new - old) > 0.01 else ""


# ---------------------------------------------------------------- loaders: per-seed per-molecule arrays, old and random-tie
def audit(key, s):
    """old: stored top-k (torch order); new: random ties. Keys: e -> array, 'rand' -> RandNode, plus mean-threshold arrays."""
    st = dict(np.load(f"{AUD}/{key}_s{s}.npz") if s in S04 else np.load(f"{P1A}/{key}_s{s}.npz"))
    rt = dict(np.load(f"{P2B}/audit_{key}_s{s}.npz") if s in S04 else np.load(f"{OUT}/gea/audit_{key}_s{s}.npz"))
    assert np.array_equal(st["mol_idx"].astype(int), rt["mol_idx"].astype(int)), (key, s)
    assert np.array_equal(st["gnnexplainer_topk"], rt["gnnexplainer_b"]), f"GNNExplainer differs: {key} s{s}"
    old = {e: st[f"{e}_topk"] for e in EXPL}; new = {e: rt[f"{e}_b"] for e in EXPL}
    return old, new, st, rt


def arch(model, s, cv=False):
    z = dict(np.load(f"runs/mutagenicity_cv/arrays_s{s}.npz" if cv else f"{ARCH[model]}/arrays_s{s}.npz"))
    rt = dict(np.load(f"{P2B}/cv_f{s}.npz" if cv else f"{P2B}/arch_{model}_s{s}.npz"))
    assert np.array_equal(z["mol_idx"].astype(int), rt["mol_idx"].astype(int))
    assert np.array_equal(z["gnnexplainer_topk_set"], rt["gnnexplainer_b"]), f"GNNExplainer differs: {model} s{s}"
    zb = dict(z)
    for e in EXPL:
        zb[f"{e}_topk_set"] = rt[f"{e}_b"]
    return z, zb


def per_seed_table(name, seeds, get, csvw, null_key="rand"):
    """rows GNN, PG, SX, random node; cols seeds + mean + std; old | new | change."""
    P(f"\n  -- {name}  (cells: old -> random-tie; * = |change| > 0.01)")
    P(f"  {'':<14}" + "".join(f"{'s' + str(s):>17}" for s in seeds) + f"{'mean':>17}{'std':>17}")
    vals = {}
    for s in seeds:
        o, n = get(s)
        for r in EXPL + ["random node"]:
            vals.setdefault(r, {"old": [], "new": []})
            vals[r]["old"].append(float(np.mean(o[r]))); vals[r]["new"].append(float(np.mean(n[r])))
    for r in EXPL + ["random node"]:
        o, n = np.array(vals[r]["old"]), np.array(vals[r]["new"])
        cells = list(zip(o, n)) + [(o.mean(), n.mean()), (o.std(), n.std())]
        P(f"  {SH.get(r, r):<14}" + "".join(f"{a:>7.3f}->{b:.3f}{flag(a, b):1}" for a, b in cells))
        for lab, (a, b) in zip([f"s{s}" for s in seeds] + ["mean", "std"], cells):
            csvw.writerow([name, SH.get(r, r), lab, f"{a:.6f}", f"{b:.6f}", f"{b - a:+.6f}", flag(a, b)])
    return vals


def signs_main(per_old, per_new, seeds):
    """main-table rule (exploratory_null_audit.sig_counts): Holm over the 3 explainers vs RandNode, direction = higher mean."""
    fam = [(f"{e}_topk", "rn_k") for e in EXPL]; out = {}
    for lab, per in (("old", per_old), ("new", per_new)):
        for s in seeds:
            A = per[s]; raw = {i: wpair(A[r], A[nl]) for i, (r, nl) in enumerate(fam)}; adj = na.holm(raw)
            for i, e in enumerate(EXPL):
                a, b = A[f"{e}_topk"].mean(), A["rn_k"].mean()
                out.setdefault((e, lab), []).append("+" if a > b and adj[i] < .05 else "-" if a < b and adj[i] < .05 else ".")
    return out


def cnt(sg):
    return f"{sg.count('+')}/{sg.count('-')}/{sg.count('.')}"


def main():
    os.makedirs(T, exist_ok=True)
    f = open(f"{T}/all_tables.csv", "w", newline=""); w = csv.writer(f)
    w.writerow(["table", "row", "column", "old", "random_tie", "change", "flag_gt_0.01"])
    ada = json.load(open(f"{OUT}/ada_checks.json"))
    P("RQ1 FINAL EXPORT -- tables under the random-tie rule (20 seeded tie orders per molecule, averaged per molecule)")
    P("GNNExplainer: stored values used; checked identical to the random-tie arrays for every unit (assertion in the loaders).")
    dev = ada["mean_threshold_max_abs_dev"]
    P(f"Mean-threshold GEA recomputed from the caches for {len(dev)} units (float64): exact in 58; GINE s1 and GCN s1 differ on one PGExplainer"
      f" molecule each (max {max(dev.values()):.3f}) because that mask has values exactly at its mean; in float32, the precision of the stored code,"
      f" all units reproduce exactly (checked separately). No ties; stored values reused.")

    # (a) / (b)
    P("\n(a) per-seed top-k GEA (k = 0.25), seeds 0-4")
    for name, key, kind in (("MUTAG-GT", "mutag_graphxai", "audit"), ("B-XAIC indole", "bxaic_indole", "audit"), ("Mutagenicity D-MPNN", None, "arch")):
        if kind == "audit":
            get = lambda s, key=key: (lambda o, n, st, rt: ({**o, "random node": st["rn_k"]}, {**n, "random node": st["rn_k"]}))(*audit(key, s))
        else:
            get = lambda s: (lambda z, zb: ({e: z[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]},
                                            {e: zb[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]}))(*arch("D-MPNN", s))
        per_seed_table(f"(a) {name}", S04, get, w)
    P("\n(b) MUTAG-GT seeds 5-9 (old = Phase 1 torch order)")
    per_seed_table("(b) MUTAG-GT seeds 5-9", S59,
                   lambda s: (lambda o, n, st, rt: ({**o, "random node": st["rn_k"]}, {**n, "random node": st["rn_k"]}))(*audit("mutag_graphxai", s)), w)

    # (c) normalised
    P("\n(c) normalised GEA = (GEA - RandNode) / (ceiling - RandNode), per seed means, mean +/- std over seeds 0-4")
    P("    ceilings as before (oracle top-k ceiling, mean over the pooled molecules): MUTAG-GT 0.751, B-XAIC indole 0.724, Mutagenicity 0.551.")
    P("    RandNode = the 20-draw matched-budget null (the earlier exploratory table used the 1-draw RandNode for MUTAG-GT and indole).")
    for name, key in (("MUTAG-GT", "mutag_graphxai"), ("B-XAIC indole", "bxaic_indole"), ("Mutagenicity D-MPNN", None)):
        c = CEILING[name]; norm = {e: {"old": [], "new": []} for e in EXPL}
        for s in S04:
            if key:
                o, n, st, _ = audit(key, s); rn = st["rn_k"].mean()
            else:
                z, zb = arch("D-MPNN", s); o = {e: z[f"{e}_topk_set"] for e in EXPL}; n = {e: zb[f"{e}_topk_set"] for e in EXPL}
                rn = z["rand_node_topk_set"].mean()
            for e in EXPL:
                norm[e]["old"].append((o[e].mean() - rn) / (c - rn)); norm[e]["new"].append((n[e].mean() - rn) / (c - rn))
        P(f"  -- {name} (ceiling {c:.3f})")
        for e in EXPL:
            a, b = np.array(norm[e]["old"]), np.array(norm[e]["new"])
            P(f"    {SH[e]:<14} old {a.mean():.3f} +/- {a.std():.3f}   random-tie {b.mean():.3f} +/- {b.std():.3f}   change {b.mean() - a.mean():+.3f}{flag(a.mean(), b.mean())}"
              f"   per seed " + " ".join(f"{x:.3f}->{y:.3f}" for x, y in zip(a, b)))
            w.writerow([f"(c) {name}", SH[e], "mean", f"{a.mean():.6f}", f"{b.mean():.6f}", f"{b.mean() - a.mean():+.6f}", flag(a.mean(), b.mean())])
            w.writerow([f"(c) {name}", SH[e], "std", f"{a.std():.6f}", f"{b.std():.6f}", f"{b.std() - a.std():+.6f}", flag(a.std(), b.std())])

    # (d) CV
    P("\n(d) Mutagenicity 5-fold CV")
    get = lambda f_: (lambda z, zb: ({e: z[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]},
                                     {e: zb[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]}))(*arch(None, f_, cv=True))
    per_seed_table("(d) CV per fold (columns are folds)", S04, get, w)
    pooled_o = {r: np.concatenate([get(f_)[0][r] for f_ in S04]) for r in EXPL + ["random node"]}
    pooled_n = {r: np.concatenate([get(f_)[1][r] for f_ in S04]) for r in EXPL + ["random node"]}
    P("  pooled over the 632 unique molecules: " + "   ".join(f"{SH.get(r, r)} {pooled_o[r].mean():.3f}->{pooled_n[r].mean():.3f}{flag(pooled_o[r].mean(), pooled_n[r].mean())}" for r in EXPL + ["random node"]))
    for r in EXPL + ["random node"]:
        w.writerow(["(d) CV pooled", SH.get(r, r), "pooled", f"{pooled_o[r].mean():.6f}", f"{pooled_n[r].mean():.6f}", f"{pooled_n[r].mean() - pooled_o[r].mean():+.6f}", flag(pooled_o[r].mean(), pooled_n[r].mean())])
    To = [family_tests(arch(None, f_, cv=True)[0], FAMILY) for f_ in S04]; Tn = [family_tests(arch(None, f_, cv=True)[1], FAMILY) for f_ in S04]
    P("  folds above / below the null (registered 4-test family, Holm within fold):")
    for lab, _, _ in FAMILY:
        so = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in To]; sn = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in Tn]
        P(f"    {lab:<18} old {cnt(so)} ({''.join(so)})   random-tie {cnt(sn)} ({''.join(sn)}){'   DIRECTION CHANGE' if so != sn else ''}")
        w.writerow(["(d) CV counts", lab, "above/below/ns", cnt(so), cnt(sn), "", "DIRECTION CHANGE" if so != sn else ""])

    # (e) architectures
    P("\n(e) per-seed top-k GEA (max-over-set), Mutagenicity, other architectures")
    for m in ("GIN", "GCN", "GINE"):
        per_seed_table(f"(e) {m}", S04, lambda s, m=m: (lambda z, zb: ({e: z[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]},
                                                                      {e: zb[f"{e}_topk_set"] for e in EXPL} | {"random node": z["rand_node_topk_set"]}))(*arch(m, s)), w)

    # (f) direction counts
    P("\n(f) direction counts (above/below/n.s.), stored vs random-tie; per-seed signs in brackets")
    P("  main table and B-XAIC: main-table rule (per-seed paired Wilcoxon; Holm over the 3 explainers vs RandNode at top-k; direction = higher mean)")
    for name, key, seeds in (("MUTAG-GT seeds 0-4", "mutag_graphxai", S04), ("MUTAG-GT seeds 0-9", "mutag_graphxai", S04 + S59),
                             ("B-XAIC indole", "bxaic_indole", S04), ("Mutagenicity (D-MPNN, max-over-set)", "mutagenicity_graphxai", S04),
                             ("B-XAIC PAINS", "bxaic_PAINS", S04), ("B-XAIC X", "bxaic_X", S04), ("B-XAIC P", "bxaic_P", S04)):
        po, pn = {}, {}
        for s in seeds:
            o, n, st, rt = audit(key, s)
            po[s] = dict(st); pn[s] = dict(st)
            for e in EXPL:
                pn[s][f"{e}_topk"] = n[e]
        sg = signs_main(po, pn, seeds)
        for e in EXPL:
            so, sn = sg[(e, "old")], sg[(e, "new")]
            P(f"    {name:<36}{SH[e]:<14} old {cnt(so)} ({''.join(so)})   random-tie {cnt(sn)} ({''.join(sn)}){'   DIRECTION CHANGE' if so != sn else ''}")
            w.writerow([f"(f) {name}", SH[e], "above/below/ns vs RandNode", cnt(so), cnt(sn), "", "DIRECTION CHANGE" if so != sn else ""])
    P("  SubgraphX vs connectivity-matched null (mean-threshold mask vs RandConn of the same size; no ties, unchanged):")
    for name, src in (("MUTAG-GT seeds 0-9", "runs/rq1_followups/phase1/connectivity_arrays_mutag_graphxai.npz"), ("B-XAIC indole", "runs/prio/connectivity_arrays_indole.npz")):
        z = np.load(src); seeds = S04 + S59 if "phase1" in src else S04; sg = []
        for s in seeds:
            p = wpair(z[f"sx_s{s}"], z[f"rand_conn_s{s}"]); d = rank_direction(z[f"sx_s{s}"], z[f"rand_conn_s{s}"])
            sg.append("+" if d > 0 and p < .05 else "-" if d < 0 and p < .05 else ".")
        P(f"    {name:<36}{cnt(sg)} ({''.join(sg)})"); w.writerow([f"(f) {name}", "SubgraphX vs RandConn (mean-thr)", "above/below/ns", cnt(sg), cnt(sg), "", ""])
    P("  architecture table: registered 4-test family (Holm within seed; direction = rank test)")
    for m in ARCH:
        To = [family_tests(arch(m, s)[0], FAMILY) for s in S04]; Tn = [family_tests(arch(m, s)[1], FAMILY) for s in S04]
        for lab, _, _ in FAMILY:
            so = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in To]; sn = ["+" if t[lab]["above"] else "-" if t[lab]["below"] else "." for t in Tn]
            P(f"    {m:<8}{lab:<18} old {cnt(so)} ({''.join(so)})   random-tie {cnt(sn)} ({''.join(sn)}){'   DIRECTION CHANGE' if so != sn else ''}")
            w.writerow([f"(f) architecture {m}", lab, "above/below/ns", cnt(so), cnt(sn), "", "DIRECTION CHANGE" if so != sn else ""])
    f.close()
    open(f"{T}/all_tables.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
