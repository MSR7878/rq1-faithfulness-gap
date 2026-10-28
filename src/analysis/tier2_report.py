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


if __name__ == "__main__":
    {"H": H, "G": G}[sys.argv[1]]()
