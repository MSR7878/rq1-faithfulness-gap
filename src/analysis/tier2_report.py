"""
RQ1 Tier 2 reports (runs/rq1_tier2/PLAN.md). One section per item; writes runs/rq1_tier2/<item>.txt / .json.

    python -m src.analysis.tier2_report H|G|R|S|M
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import tier2_common as C

T = C.T2
LOG = []
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX", "hetero": "hetero", "deg_high": "deg-high", "deg_low": "deg-low",
      "sal": "Saliency", "ig": "IG", "ixg": "IxG", "sx_fixed": "SX-fixed"}
RULES = ["R1", "R2", "R3"]
BETTER = {"fidp": +1, "fidm": -1}


def P(s=""):
    print(s); LOG.append(s)


def save(name, rep):
    open(f"{T}/{name}.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{T}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


_MEMO = {}


def load(path):
    if path not in _MEMO:
        _MEMO[path] = dict(np.load(path, allow_pickle=True))
    return _MEMO[path]


def fmt_ci(p, lo, hi):
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"


# ------------------------------------------------------------------ GEA tables
def gea_vs(groups, arrays, methods, refs, family_extra=None, title=""):
    """arrays: unit -> dict of per-molecule arrays (same GT molecules); refs: list of (key, label).
    Holm family per seed = methods x refs (+ family_extra pairs). Returns rep[group][(m, ref)] = {...}."""
    rep = {}
    P(title)
    P(f"  {'group':<22}{'method':<10}" + "".join(f"{'minus ' + lab + ' [95% CI]':>34}{'counts':>9}" for _, lab in refs) + f"{'mean GEA':>10}")
    for g, units in groups.items():
        sig = {}
        for u in units:
            A = arrays[u]; raw = {(m, r): C.wil(A[m], A[r]) for m in methods for r, _ in refs}
            for k, s in C.holm_signs(raw).items():
                sig.setdefault(k, []).append(s)
        rep[g] = {}
        for m in methods:
            cells = []
            for r, lab in refs:
                p, lo, hi = C.boot([arrays[u][m] for u in units], [arrays[u][r] for u in units])
                rep[g][f"{m}|{r}"] = {"diff": p, "lo": lo, "hi": hi, "cat": C.ci_cat(lo, hi), "signs": sig[(m, r)]}
                cells.append(f"{fmt_ci(p, lo, hi):>34}{C.counts(sig[(m, r)]):>9}")
            mg = float(np.mean([np.mean(arrays[u][m]) for u in units])); rep[g][f"{m}|mean"] = mg
            P(f"  {g:<22}{SH.get(m, m):<10}" + "".join(cells) + f"{mg:>10.3f}")
    return rep


def method_vs_method(groups, arrays, methods, others, title):
    """method minus other method; Holm family per seed = methods x others."""
    rep = {}
    P(title)
    P(f"  {'group':<22}{'method':<10}" + "".join(f"{'minus ' + SH[o] + ' [95% CI]':>34}{'counts':>9}" for o in others))
    for g, units in groups.items():
        sig = {}
        for u in units:
            A = arrays[u]; raw = {(m, o): C.wil(A[m], A[o]) for m in methods for o in others}
            for k, s in C.holm_signs(raw).items():
                sig.setdefault(k, []).append(s)
        rep[g] = {}
        for m in methods:
            cells = []
            for o in others:
                p, lo, hi = C.boot([arrays[u][m] for u in units], [arrays[u][o] for u in units])
                rep[g][f"{m}|{o}"] = {"diff": p, "lo": lo, "hi": hi, "cat": C.ci_cat(lo, hi), "signs": sig[(m, o)]}
                cells.append(f"{fmt_ci(p, lo, hi):>34}{C.counts(sig[(m, o)]):>9}")
            P(f"  {g:<22}{SH.get(m, m):<10}" + "".join(cells))
    return rep


# ------------------------------------------------------------------ Fid tables
def randnode_E(unit):
    return load(f"runs/rq1_revision/pass/E/{unit}.npz")


def fid_table(groups, fid_of, null_of, methods, title):
    """'+' = better than random-node (Fid+ higher, Fid- lower). Holm family per (unit, method, space) = 6."""
    rep = {}
    P(title)
    for space in ("prob", "logit"):
        P(f"\n  [{space} space] counts better/worse/n.s. over seeds and bootstrap mean (method - random-node) [95% CI]")
        P(f"  {'group':<22}{'method':<10}{'metric':<6}" + "".join(f"{r:>40}" for r in RULES))
        for g, units in groups.items():
            for m in methods:
                sig = {}
                for u in units:
                    F, N = fid_of(u), null_of(u)
                    assert list(F["mol_idx"]) == list(N["mol_idx"]), u
                    raw = {(mt, r): C.wil(F[f"{m}_{r}_{mt}_{space}"], N[f"randnode_{r}_{mt}_{space}"]) for mt in ("fidp", "fidm") for r in RULES}
                    for k, s in C.holm_signs(raw).items():
                        b = "." if s == "." else ("+" if (s == "+") == (BETTER[k[0]] > 0) else "-")
                        sig.setdefault(k, []).append(b)
                for mt in ("fidp", "fidm"):
                    cells = []
                    for r in RULES:
                        p, lo, hi = C.boot([fid_of(u)[f"{m}_{r}_{mt}_{space}"] for u in units], [null_of(u)[f"randnode_{r}_{mt}_{space}"] for u in units])
                        rep[f"{g}|{m}|{space}|{mt}|{r}"] = {"diff": p, "lo": lo, "hi": hi, "signs": sig[(mt, r)]}
                        cells.append(f"{C.counts(sig[(mt, r)]):>9} {fmt_ci(p, lo, hi):>30}")
                    P(f"  {g:<22}{SH.get(m, m):<10}{'Fid+' if mt == 'fidp' else 'Fid-':<6}" + "".join(cells))
    return rep


# ------------------------------------------------------------------ H
def H():
    rep = {}
    A = {u: load(f"{T}/H/gea/{u}.npz") for g in C.GROUPS.values() for u in g}
    P("H -- HEURISTIC BASELINES (final protocol: top 25%, random ties, shared nulls; see PLAN.md)")
    P("  PGExplainer tie rule (stored mask, exact ties split by reproduced float64 logits): tie-boundary molecules / resolved by logits / random fallback")
    for g, units in C.GROUPS.items():
        P(f"    {g:<22}" + "  ".join(f"s{u.rsplit('_s', 1)[1]} {int(A[u]['pg_tie_boundary'])}/{int(A[u]['pg_resolved_by_logits'])}/{int(A[u]['pg_random_fallback'])}" for u in units))
    refs = [("rn", "random-node"), ("rc", "random-connected")]
    rep["heur_vs_null"] = gea_vs(C.GROUPS, A, ["hetero", "deg_high", "deg_low"], refs,
                                 title="\n  [1] heuristic GEA minus null; Holm family per seed = 3 heuristics x 2 nulls = 6")
    rep["expl_vs_null"] = gea_vs(C.GROUPS, A, C.EXPL, refs,
                                 title="\n  [2] explainers re-scored under the final protocol (SX = stored legacy budget); Holm family per seed = 3 x 2 = 6")
    rep["heur_vs_expl"] = method_vs_method(C.GROUPS, A, ["hetero", "deg_high", "deg_low"], C.EXPL,
                                           "\n  [3] heuristic minus explainer GEA; Holm family per seed = 3 heuristics x 3 explainers = 9")
    rep["fid"] = fid_table(C.GROUPS, lambda u: load(f"{T}/H/fid/{u}.npz"), randnode_E, ["hetero", "deg_high", "deg_low"],
                           "\n  [4] Fid+ / Fid- vs random-node (Part-E arrays, same molecules); Holm family per (unit, heuristic, space) = 6")
    groups_h1 = ["Mutagenicity D-MPNN", "Mutagenicity GIN", "Mutagenicity GINE", "Mutagenicity GCN", "MUTAG-GT (10 seeds)"]
    h1 = {g: rep["heur_vs_null"][g]["hetero|rn"] for g in groups_h1}
    held1 = all(v["lo"] > 0 for v in h1.values())
    P("\n  P-H1: hetero minus random-node 95% CI entirely above 0 in all five groups")
    for g, v in h1.items():
        P(f"    {g:<22}{fmt_ci(v['diff'], v['lo'], v['hi'])}  {'above' if v['lo'] > 0 else 'NOT above'}")
    P(f"  P-H1 {'HELD' if held1 else 'FAILED'}")
    d = rep["heur_vs_null"]["Mutagenicity D-MPNN"]; e = rep["expl_vs_null"]["Mutagenicity D-MPNN"]
    means = {x: e[f"{x}|mean"] for x in C.EXPL}; hm = d["hetero|mean"]; held2 = hm >= min(means.values())
    P(f"\n  P-H2: Mutagenicity D-MPNN mean GEA: hetero {hm:.3f}; GNN {means['gnnexplainer']:.3f}, PG {means['pgexplainer']:.3f}, "
      f"SX (legacy budget) {means['subgraphx']:.3f} -> hetero >= lowest ({min(means.values()):.3f}): {'HELD' if held2 else 'FAILED'}")
    rep["P-H1"] = {"held": held1, "cells": h1}; rep["P-H2"] = {"held": bool(held2), "hetero": hm, "explainers": means}
    save("H", rep)


# ------------------------------------------------------------------ G
def G():
    rep = {"checks": {}}
    P("G -- GRADIENT METHODS (Captum 0.7.0: Saliency, Integrated Gradients 50 steps zero baseline, Input x Gradient; target = predicted-class logit)")
    P("  node attribution = sum over atom features of |attribution|; final GEA protocol; shared nulls")
    P("\n  [0] checks per unit: molecules, all finite, all non-zero, IG completeness |sum attr - (f(x) - f(0))| / |f(x) - f(0)| median / max")
    ok_all = True
    for g, units in C.GROUPS.items():
        cells = []
        for u in units:
            z = load(f"{T}/G/grad/{u}.npz"); rel = np.asarray(z["ig_delta"], float) / np.maximum(np.abs(np.asarray(z["ig_fx_f0"], float)), 1e-12)
            fin, nz = bool(np.all(z["finite"])), bool(np.all(z["nonzero"])); med = float(np.median(rel)); ok = fin and nz and med <= 0.05
            ok_all &= ok
            rep["checks"][u] = {"n": len(z["mol_idx"]), "finite": fin, "nonzero": nz, "ig_rel_median": med, "ig_rel_max": float(rel.max()), "ok": ok}
            cells.append(f"s{u.rsplit('_s', 1)[1]} {len(z['mol_idx'])} {'ok' if fin and nz else 'FAIL'} {med:.3f}/{rel.max():.2f}")
        P(f"    {g:<22}" + "  ".join(cells))
    P(f"  checks: {'all passed' if ok_all else 'FAILED -- stop and report'}")
    rep["checks_ok"] = ok_all
    A = {}
    for g in C.GROUPS.values():
        for u in g:
            a, h = load(f"{T}/G/gea/{u}.npz"), load(f"{T}/H/gea/{u}.npz")
            assert list(a["mol_idx"]) == list(h["mol_idx"]), u
            A[u] = {**h, **{m: a[m] for m in ("sal", "ig", "ixg")}}
    refs = [("rn", "random-node"), ("rc", "random-connected")]
    rep["vs_null"] = gea_vs(C.GROUPS, A, ["sal", "ig", "ixg"], refs, title="\n  [1] gradient GEA minus null; Holm family per seed = 3 methods x 2 nulls = 6")
    rep["vs_expl"] = method_vs_method(C.GROUPS, A, ["sal", "ig", "ixg"], C.EXPL,
                                      "\n  [2] gradient minus explainer GEA (SX = legacy budget); Holm family per seed = 3 x 3 = 9")
    rep["fid"] = fid_table(C.GROUPS, lambda u: load(f"{T}/G/fid/{u}.npz"), randnode_E, ["sal", "ig", "ixg"],
                           "\n  [3] Fid+ / Fid- vs random-node; Holm family per (unit, method, space) = 6")
    v = rep["vs_null"]["Mutagenicity D-MPNN"]["ig|rn"]
    P(f"\n  P-G1: IG minus random-node on Mutagenicity D-MPNN {fmt_ci(v['diff'], v['lo'], v['hi'])} -> {'HELD' if v['lo'] > 0 else 'FAILED'}")
    rep["P-G1"] = {"held": v["lo"] > 0, **v}
    save("G", rep)


# ------------------------------------------------------------------ R
def R():
    from .revision_analysis import hboot
    rep = {"models": {}}
    RG = {"Mutagenicity D-MPNN": [f"mutagenicity_dmpnn_s{s}" for s in range(5)], "B-XAIC indole": [f"bxaic_indole_s{s}" for s in range(5)]}
    M = ["gnnexplainer", "pgexplainer", "subgraphx", "ig", "sal"]; SHR = {**SH, "subgraphx": "SX-fixed"}
    P("R -- RANDOMISED-MODEL SANITY CHECK (full = every module re-initialised; head = output FFN only; seed 5000 + model seed)")
    P("  SubgraphX (scoring budget) on the fixed 100 ground-truth molecules per seed; other methods on all GT molecules.")
    import torch
    P("\n  [0] test accuracy / AUROC / share predicted positive of the randomised models (trained in brackets)")
    for g, units in RG.items():
        for k in ("full", "head"):
            cells = []
            for u in units:
                ck = torch.load(f"{T}/R/ckpt/{u}_{k}.pt", map_location="cpu", weights_only=False); m = ck["metrics"]
                rep["models"][f"{u}|{k}"] = {kk: m[kk] for kk in ("test_acc", "test_auroc", "frac_pred_pos")}
                cells.append(f"s{u[-1]} {m['test_acc']:.2f}/{m['test_auroc']:.2f}/{m['frac_pred_pos']:.2f}")
            P(f"    {g:<22}{k:<6}" + "  ".join(cells))
    Z = {(u, k): load(f"{T}/R/score/{u}_{k}.npz") for us in RG.values() for u in us for k in ("full", "head")}
    P("\n  [1] GEA of the explanations of the randomised model minus null; Holm family per (dataset, randomisation, seed) = 5 methods x 2 nulls = 10")
    P(f"  {'dataset':<22}{'kind':<6}{'method':<10}{'minus random-node [95% CI]':>32}{'90% CI':>20}{'counts':>8}{'minus random-conn. [95% CI]':>32}{'counts':>8}{'GEA':>7}")
    cells_r1 = []
    rep["gea"] = {}
    for g, units in RG.items():
        for k in ("full", "head"):
            sig = {}
            for u in units:
                z = Z[(u, k)]
                raw = {(m, r): C.wil(z[f"{m}_gea"], z[f"{m}_{r}"]) for m in M for r in ("rn", "rc")}
                for kk, sg in C.holm_signs(raw).items():
                    sig.setdefault(kk, []).append(sg)
            for m in M:
                real = [Z[(u, k)][f"{m}_gea"] for u in units]
                p, lo, hi, vals = hboot(real, [Z[(u, k)][f"{m}_rn"] for u in units])
                l90, h90 = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
                pc, loc, hic, _ = hboot(real, [Z[(u, k)][f"{m}_rc"] for u in units])
                within = -0.05 <= p <= 0.05; cells_r1.append((g, k, m, p, within, l90, h90))
                rep["gea"][f"{g}|{k}|{m}"] = {"rn": [p, lo, hi], "rn90": [l90, h90], "rc": [pc, loc, hic], "signs_rn": sig[(m, "rn")],
                                             "signs_rc": sig[(m, "rc")], "within_0.05": within, "gea": float(np.mean([np.mean(x) for x in real]))}
                P(f"  {g:<22}{k:<6}{SHR[m]:<10}{fmt_ci(p, lo, hi):>32}{f'[{l90:+.3f}, {h90:+.3f}]':>20}{C.counts(sig[(m, 'rn')]):>8}"
                  f"{fmt_ci(pc, loc, hic):>32}{C.counts(sig[(m, 'rc')]):>8}{np.mean([np.mean(x) for x in real]):>7.3f}")
    held = all(c[4] for c in cells_r1)
    P(f"\n  P-R1: point estimate of (method - random-node) within [-0.05, +0.05] in every one of the {len(cells_r1)} cells: "
      f"{sum(c[4] for c in cells_r1)}/{len(cells_r1)} -> {'HELD' if held else 'FAILED'}")
    for c in cells_r1:
        if not c[4]:
            P(f"    outside: {c[0]} {c[1]} {SHR[c[2]]} {c[3]:+.3f}")
    eq = sum(1 for c in cells_r1 if c[5] > -0.05 and c[6] < 0.05)
    P(f"  equivalence view (90% CI inside +/-0.05): {eq}/{len(cells_r1)} cells")
    rep["P-R1"] = {"held": held, "within": sum(c[4] for c in cells_r1), "cells": len(cells_r1), "equivalent_90": eq}
    P("\n  [2] trained vs randomised atom scores per GT molecule: Spearman (mean over seeds of per-seed means; bootstrap 95% CI) and top-k Jaccard;")
    P("      n skipped = molecules with a constant score vector on either side; SX-fixed trained = the item-S run (pending until S priority is done)")
    P(f"  {'dataset':<22}{'kind':<6}{'method':<10}{'Spearman [95% CI]':>30}{'skipped':>9}{'top-k Jaccard [95% CI]':>30}")
    rep["corr"] = {}
    for g, units in RG.items():
        for k in ("full", "head"):
            for m in M:
                rr = [Z[(u, k)][f"{m}_spearman"] for u in units]; jj = [Z[(u, k)][f"{m}_jaccard"] for u in units]
                if all(np.all(np.isnan(x)) for x in jj):
                    P(f"  {g:<22}{k:<6}{SHR[m]:<10}{'pending (no trained scores yet)':>30}"); continue
                skip = int(sum(np.isnan(x).sum() for x in rr))
                r_ok = [x[~np.isnan(x)] for x in rr]; j_ok = [x[~np.isnan(x)] for x in jj]
                rs = hboot(r_ok, [np.zeros_like(x) for x in r_ok]) if all(len(x) for x in r_ok) else (np.nan, np.nan, np.nan)
                js = hboot(j_ok, [np.zeros_like(x) for x in j_ok])
                rep["corr"][f"{g}|{k}|{m}"] = {"spearman": list(rs[:3]), "skipped": skip, "jaccard": list(js[:3])}
                P(f"  {g:<22}{k:<6}{SHR[m]:<10}{fmt_ci(*rs[:3]):>30}{skip:>9}{fmt_ci(*js[:3]):>30}")
    save("R", rep)


# ------------------------------------------------------------------ S
def S(stage="prio"):
    import os
    rep = {"stage": stage}
    groups = {g: us for g, us in C.GROUPS.items() if all(os.path.exists(f"{T}/S/gea/{u}.npz") for u in us)}
    P(f"S -- SUBGRAPHX WITH THE SCORING BUDGET (stage: {stage}; groups done: {', '.join(groups)})")
    P("  old = stored runs, budget max(3, round(0.25 N)), MCTS leaf 3; new = budget k = max(1, floor(0.25 N)), leaf min(3, k);")
    P("  both scored under the final protocol on the same shared nulls (GEA on GT molecules; Fid on all explained molecules)")
    Z = {u: (load(f"{T}/H/gea/{u}.npz"), load(f"{T}/S/gea/{u}.npz")) for us in groups.values() for u in us}
    P("\n  [0] coalition size against the scoring k (all explained molecules), budget changed, run time")
    P(f"  {'group':<22}{'old |S|=k':>10}{'<k':>7}{'>k':>7}{'new |S|=k':>11}{'<k':>7}{'>k':>7}{'budget changed':>16}{'new s/mol':>11}")
    rep["sizes"] = {}
    for g, us in groups.items():
        k = np.concatenate([Z[u][1]["k"] for u in us]); o = np.concatenate([Z[u][1]["old_size"] for u in us])
        n = np.concatenate([Z[u][1]["new_size"] for u in us])
        ob = np.concatenate([Z[u][1]["old_max_nodes"] for u in us]); sec = np.concatenate([Z[u][1]["seconds"] for u in us])
        r = {"old_eq": float(np.mean(o == k)), "old_lt": float(np.mean(o < k)), "old_gt": float(np.mean(o > k)), "new_eq": float(np.mean(n == k)),
             "new_lt": float(np.mean(n < k)), "new_gt": float(np.mean(n > k)), "budget_changed": float(np.mean(ob != k)), "sec": float(sec.mean())}
        rep["sizes"][g] = r
        P(f"  {g:<22}{r['old_eq']:>10.2f}{r['old_lt']:>7.2f}{r['old_gt']:>7.2f}{r['new_eq']:>11.2f}{r['new_lt']:>7.2f}{r['new_gt']:>7.2f}"
          f"{r['budget_changed']:>16.2f}{r['sec']:>11.1f}")
    P("\n  [1] SX GEA minus null, old vs new; Holm family per seed = SX vs {random-node, random-connected} = 2 (old and new each)")
    P(f"  {'group':<22}{'null':<6}{'old [95% CI]':>30}{'counts':>8}{'new [95% CI]':>30}{'counts':>8}{'GEA old/new':>14}   flag")
    rep["gea"] = {}
    changes = []
    for g, us in groups.items():
        for r, lab in (("rn", "node"), ("rc", "conn")):
            res = {}
            for tag, key, idx in (("old", "subgraphx", 0), ("new", "sx_fixed", 1)):
                sig = []
                for u in us:
                    raw = {rr: C.wil(Z[u][idx][key], Z[u][0][rr]) for rr in ("rn", "rc")}
                    sig.append(C.holm_signs(raw)[r])
                p_, lo, hi = C.boot([Z[u][idx][key] for u in us], [Z[u][0][r] for u in us])
                res[tag] = {"diff": p_, "lo": lo, "hi": hi, "cat": C.ci_cat(lo, hi), "signs": sig,
                            "gea": float(np.mean([np.mean(Z[u][idx][key]) for u in us]))}
            flag = []
            if res["old"]["cat"] != res["new"]["cat"]:
                flag.append(f"CI {res['old']['cat']} -> {res['new']['cat']}")
            if C.counts(res["old"]["signs"]) != C.counts(res["new"]["signs"]):
                flag.append("counts changed")
            if flag:
                changes.append((g, lab, "; ".join(flag)))
            rep["gea"][f"{g}|{r}"] = {**res, "flag": flag}
            P(f"  {g:<22}{lab:<6}{fmt_ci(res['old']['diff'], res['old']['lo'], res['old']['hi']):>30}{C.counts(res['old']['signs']):>8}"
              f"{fmt_ci(res['new']['diff'], res['new']['lo'], res['new']['hi']):>30}{C.counts(res['new']['signs']):>8}"
              f"{res['old']['gea']:>7.3f}/{res['new']['gea']:.3f}   {'; '.join(flag)}")
    rep["changes"] = changes
    rep["fid_new"] = fid_table(groups, lambda u: load(f"{T}/S/fid/{u}.npz"), randnode_E, ["sx_fixed"],
                               "\n  [2a] Fid+ / Fid- of the NEW SubgraphX vs random-node; Holm family per (unit, space) = 6")
    rep["fid_old"] = fid_table(groups, lambda u: load(f"runs/rq1_revision/pass/E/{u}.npz"), randnode_E, ["subgraphx"],
                               "\n  [2b] the same for the OLD (stored) SubgraphX, Part-E arrays (random ties)")
    valid = [g for g in groups if not g.startswith("MUTAG-GT")]
    n_same = sum(rep["gea"][f"{g}|rn"]["old"]["cat"] == rep["gea"][f"{g}|rn"]["new"]["cat"] for g in valid)
    mg = rep["gea"].get("MUTAG-GT (10 seeds)|rn")
    below = mg is not None and mg["new"]["hi"] < 0
    final = len(valid) == 8
    held = n_same == len(valid) and below
    stage_txt = "final" if final else f"PARTIAL, {len(valid)} of 8 valid groups"
    mg_txt = f"{mg['new']['hi']:+.3f}" if mg is not None else "n/a"
    P(f"\n  P-S1 ({stage_txt}): SX - random-node CI category unchanged in {n_same}/{len(valid)} valid groups; "
      f"MUTAG-GT new upper bound {mg_txt} -> {'below 0' if below else 'NOT below 0'} -> {('HELD' if held else 'FAILED') + ('' if final else ' so far')}")
    rep["P-S1"] = {"final": final, "held": bool(held), "valid_groups": len(valid), "unchanged": n_same}
    if "Mutagenicity D-MPNN" in groups:
        us = groups["Mutagenicity D-MPNN"]
        hm = float(np.mean([np.mean(Z[u][0]["hetero"]) for u in us]))
        ex = {e: float(np.mean([np.mean(Z[u][0][e]) for u in us])) for e in ("gnnexplainer", "pgexplainer")}
        ex["sx_fixed"] = float(np.mean([np.mean(Z[u][1]["sx_fixed"]) for u in us]))
        P(f"  P-H2 secondary line (fixed-budget SX): hetero {hm:.3f}; GNN {ex['gnnexplainer']:.3f}, PG {ex['pgexplainer']:.3f}, "
          f"SX-fixed {ex['sx_fixed']:.3f} -> hetero >= lowest: {hm >= min(ex.values())}")
        rep["P-H2_secondary"] = {"hetero": hm, **ex}
    save(f"S_{stage}", rep)


def S_sens():
    from .tier2_jobs import SENS, SENS_UNITS
    rep = {}
    sub = json.load(open(f"{T}/subsets.json"))
    P("S SENSITIVITY -- SubgraphX search settings on the fixed 100 GT molecules (scoring budget)")
    P("  base = rollout 20 / sample 30 (the item-S run); 'rep' = base settings with another seed tag (Monte-Carlo noise floor)")
    P("  changed = share of molecules whose selected coalition differs from the base one")
    P(f"  {'unit':<24}{'setting':<10}{'rollout':>8}{'sample':>7}{'mean GEA':>10}{'minus base':>11}{'s/mol':>8}{'changed':>9}")
    for u in SENS_UNITS:
        dkey = C.dataset_of(u)
        mols = sub[u]
        base = C.sx_shard_rows(f"{T}/S/sx/{u}")
        rows = {"base": (20, 30, base)}
        for name, (r, s_, _) in SENS.items():
            rows[name] = (r, s_, C.sx_shard_rows(f"{T}/S/sens/{name}/{u}"))
        g0 = None
        rep[u] = {}
        for name, (r, s_, R_) in rows.items():
            assert all(m in R_ for m in mols), (u, name)
            g = float(C.gea_array(u, f"sens-{name}", lambda m, R_=R_: np.asarray(R_[m]["node_importance"], float), mols, dkey).mean())
            if name == "base":
                g0 = g
            ch = float(np.mean([set(R_[m]["coalition"]) != set(base[m]["coalition"]) for m in mols]))
            sec = float(np.mean([R_[m]["seconds"] for m in mols]))
            rep[u][name] = {"rollout": r, "sample": s_, "gea": g, "minus_base": g - g0, "sec": sec, "changed": ch}
            P(f"  {u:<24}{name:<10}{r:>8}{s_:>7}{g:>10.3f}{g - g0:>+11.3f}{sec:>8.1f}{ch:>9.2f}")
    held = all(abs(rep[u]["r100"]["minus_base"]) < 0.03 for u in SENS_UNITS)
    vals = ", ".join(f"{u} {rep[u]['r100']['minus_base']:+.3f}" for u in SENS_UNITS)
    P(f"\n  P-S2: |GEA(rollout 100) - GEA(rollout 20)| < 0.03 on both subsets: {vals} -> {'HELD' if held else 'FAILED'}")
    rep["P-S2"] = held
    save("S_sensitivity", rep)


# ------------------------------------------------------------------ M
def M():
    import torch
    rep = {}
    MG = {"Mutagenicity D-MPNN": "mutagenicity_dmpnn", "B-XAIC indole": "bxaic_indole"}
    meth = ["gnnexplainer", "pgexplainer", "sx_fixed", "ig", "hetero"]
    P("M -- MORE SEEDS (5-9 trained with the original commands; seeds 0-4 from items H, G, S; SX = scoring budget)")
    P("\n  [0] new models: test accuracy / AUROC")
    for g, key in MG.items():
        cells = []
        for s in range(5, 10):
            ck = torch.load(C.m_units()[f"{key}_s{s}"]["ckpt"], map_location="cpu", weights_only=False)
            cells.append(f"s{s} {ck['metrics']['test_acc']:.3f}/{ck['metrics']['test_auroc']:.3f}")
            rep[f"{key}_s{s}"] = {"acc": ck["metrics"]["test_acc"], "auroc": ck["metrics"]["test_auroc"]}
        P(f"    {g:<22}" + "  ".join(cells))
    A = {}
    for key in MG.values():
        for s in range(10):
            u = f"{key}_s{s}"
            if s < 5:
                h, gg, ss = load(f"{T}/H/gea/{u}.npz"), load(f"{T}/G/gea/{u}.npz"), load(f"{T}/S/gea/{u}.npz")
                A[u] = {"rn": h["rn"], "rc": h["rc"], "gnnexplainer": h["gnnexplainer"], "pgexplainer": h["pgexplainer"], "hetero": h["hetero"],
                        "ig": gg["ig"], "sx_fixed": ss["sx_fixed"]}
            else:
                A[u] = load(f"{T}/M/gea/{u}.npz")
    P("\n  [1] method minus random-node GEA: 5 seeds vs 10 seeds; Holm family per seed = 5 methods x 2 nulls = 10;")
    P("      sign test = binomial test on the number of seeds with a positive mean difference")
    P(f"  {'group':<22}{'method':<10}{'5 seeds [95% CI]':>30}{'counts':>8}{'10 seeds [95% CI]':>30}{'counts':>9}{'sign test 10':>16}  match")
    allm = True
    rep["cells"] = {}
    for g, key in MG.items():
        sig = {}
        for s in range(10):
            z = A[f"{key}_s{s}"]
            raw = {(m, r): C.wil(z[m], z[r]) for m in meth for r in ("rn", "rc")}
            for kk, sg in C.holm_signs(raw).items():
                sig.setdefault(kk, []).append(sg)
        for m in meth:
            r5 = C.boot([A[f"{key}_s{s}"][m] for s in range(5)], [A[f"{key}_s{s}"]["rn"] for s in range(5)])
            r10 = C.boot([A[f"{key}_s{s}"][m] for s in range(10)], [A[f"{key}_s{s}"]["rn"] for s in range(10)])
            pos, nz, sp = C.seed_sign_test([A[f"{key}_s{s}"][m] for s in range(10)], [A[f"{key}_s{s}"]["rn"] for s in range(10)])
            c5, c10 = C.ci_cat(r5[1], r5[2]), C.ci_cat(r10[1], r10[2])
            match = c5 == c10
            allm &= match
            rep["cells"][f"{g}|{m}"] = {"5": list(r5), "10": list(r10), "cat5": c5, "cat10": c10, "signs": sig[(m, "rn")], "sign_test": [pos, nz, sp]}
            st = f"{pos}/{nz} p={sp:.3f}"
            P(f"  {g:<22}{SH.get(m, m):<10}{fmt_ci(*r5):>30}{C.counts(sig[(m, 'rn')][:5]):>8}{fmt_ci(*r10):>30}{C.counts(sig[(m, 'rn')]):>9}"
              f"{st:>16}  {'yes' if match else 'NO'}")
    P(f"\n  P-M1: CI category (vs random-node) with 10 seeds equals the 5-seed one in every (dataset, method) cell -> {'HELD' if allm else 'FAILED'}")
    rep["P-M1"] = allm
    save("M", rep)


if __name__ == "__main__":
    a = sys.argv[1]
    if a.startswith("S:"):
        S(a[2:])
    else:
        {"H": H, "G": G, "R": R, "S_sens": S_sens, "M": M}[a]()
