"""
RQ1 Tier 5 reports (runs/rq1_tier5/PLAN.md). Every number is method minus null (or minus another method) with the hierarchical
bootstrap 95% CI (seeds, then molecules); counts are per-seed better / worse / n.s. after Holm (families as in the PLAN).
    python -m src.analysis.tier5_report N0|N4|N1|N2|N3      -> runs/rq1_tier5/<ITEM>.txt / .json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import revision_analysis as RA
from . import tier2_common as C
from . import tier5_common as T5
from .tier2_report import fmt_ci
from .tier4_report import flip, holm_counts

LOG = []
SH = {"eig": "EiG", "eig_own": "EiG (own size)", "same": "SAME", "same_own": "SAME (own size)", "gstarx": "GStarX", "gnnshap": "GNNShap",
      "gnnexplainer": "GNN", "pgexplainer": "PG (logit)", "sx_fixed": "SX fixed", "sal": "Saliency", "ig": "IG", "ixg": "IxG",
      "hetero": "hetero", "deg_high": "high deg", "deg_low": "low deg", "rank3": "rank-avg(GNN,SX,IG)", "rank2": "rank-avg(GNN,SX)",
      "ig_smooth": "IG smoothed"}
NEWB = ["eig", "same", "gstarx", "gnnshap"]                       # the fixed-budget versions (predictions)
NEWV = ["eig", "eig_own", "same", "same_own", "gstarx", "gnnshap"]
OLDN = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig", "ixg"]
RULES3 = ["hetero", "deg_high", "deg_low"]
CELL_ORDER = [f"{d}|{a}" for d in ("bxaic_indole", "bxaic_PAINS", "mutagenicity_graphxai") for a in T5.ARCH4] + ["mutag_graphxai|dmpnn"]
CELL_NAME = {"bxaic_indole": "indole", "bxaic_PAINS": "PAINS", "mutagenicity_graphxai": "Mutagenicity", "mutag_graphxai": "MUTAG-GT"}


def P(s=""):
    print(s, flush=True); LOG.append(s)


def save(name, rep):
    open(f"{T5.T5}/{name}.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{T5.T5}/{name}.json", "w"), indent=1, default=float)


def cname(cell):
    d, a = cell.split("|"); return f"{CELL_NAME[d]} {a.upper().replace('DMPNN', 'D-MPNN')}"


def hb(real, null):
    p, lo, hi, _ = RA.hboot(real, null); return [float(p), float(lo), float(hi)]


def cat(ci):
    return C.ci_cat(ci[1], ci[2])


def units_of(cell, pool):
    return [u for u in pool if T5.cell_of(u) == cell]


# ------------------------------------------------------------------------------------------------ N0
def N0():
    chk = json.load(open(f"{T5.T5}/N0/check.json")); sc = json.load(open(f"{T5.T5}/scale.json")); rep = {"check": chk, "scale": sc, "integration": {}}
    from .tier5_N0 import TIME_UNITS
    P("N0 -- INTEGRATION CHECK (no GEA or Fid number)")
    P(f"\n  [1] adapter checks (pass needed): {'all pass' if chk['all_pass'] else 'FAILED'}")
    for u, r in chk["units"].items():
        P(f"    {u:<24} full {r['adapter_full_maxdiff']:.1e}  removed {r['adapter_removed_maxdiff']:.1e}  batched {r['batched_vs_single_maxdiff']:.1e}  "
          f"EiG two-graph {r['eig_two_graph_vs_direct']:.1e}  zero-bond/single-atom finite {all(r['zero_bond_single_atom_finite'].values())}  -> {'pass' if r['pass'] else 'FAIL'}")
    P("\n  [2] per explainer and model (6 timing molecules): runs / finite / non-constant / perturbation reaches the model / mean seconds")
    for name in T5.NEW:
        for u in TIME_UNITS:
            f = f"{T5.T5}/N0/time/{u}__{name}.json"
            if not os.path.exists(f):
                P(f"    {SH[name]:<8} {u:<26} NOT RUN"); continue
            z = json.load(open(f)); rows = z["rows"]
            if "error" in z:
                P(f"    {SH[name]:<8} {u:<26} FAILED: {z['error']}"); rep["integration"][f"{name}|{u}"] = {"status": "failed", "error": z["error"]}; continue
            ok = [r for r in rows if "seconds" in r]; err = [r for r in rows if "error" in r]
            fin = sum(r["finite"] for r in ok); nc = sum(r["nonconstant"] for r in ok); pr = sum(r["perturbation"]["differs"] for r in ok)
            per = {}
            for r in ok:
                per[r["mol_idx"]] = per.get(r["mol_idx"], 0.0) + r["seconds"]
            st = "ok" if ok and not err and fin == len(ok) else "problem"
            P(f"    {SH[name]:<8} {u:<26} {len(ok)}/{len(rows)} runs, finite {fin}, non-constant {nc}, perturbation {pr}/{len(ok)}, "
              f"{np.mean(list(per.values())) if per else float('nan'):8.1f} s/molecule" + (f"  ERRORS: {err[0]['error']}" if err else ""))
            rep["integration"][f"{name}|{u}"] = {"status": st, "runs": len(ok), "of": len(rows), "finite": fin, "nonconstant": nc, "perturbation": pr,
                                                "seconds": float(np.mean(list(per.values()))) if per else None,
                                                "no_change": [r["mol_idx"] for r in ok if not r["perturbation"]["differs"]]}
    P("\n  [3] scale rule (projected N1 hours at 12 workers; first level <= 24 h)")
    for name, s in sc.items():
        P(f"    {SH[name]:<8} -> {s['level']}{' (over 24 h even at L4)' if s.get('over_budget_even_at_L4') else ''}   "
          + "  ".join(f"{lv} {p['hours']:.1f} h" for lv, p in s["projection"].items()))
    save("N0", rep)


# ------------------------------------------------------------------------------------------------ N4
def N4():
    from .tier5_N4 import COMBOS, OLD
    rep = {"cells": {}}; Z = {u: dict(np.load(f"{T5.T5}/N4/{u}.npz", allow_pickle=True)) for u in T5.n4_units()}
    P("N4 -- COMBINATIONS FROM CACHED SCORES (no new explanations); GT molecules with SubgraphX; seeds: all that exist")
    P("  shares are per-molecule GEA comparisons (mean over seeds of the per-seed share, bootstrap CI); Holm per unit = 3 combinations x 3 comparisons")
    held = True
    for cell in CELL_ORDER[:-1]:
        us = units_of(cell, T5.n4_units()); d = cell.split("|")[0]; rule = T5.RULE[d]
        g = lambda w: [Z[u][w] for u in us]
        zeros = [np.zeros(len(Z[u]["mol_idx"])) for u in us]
        sh = {"GNN > SX": hb([(Z[u]["gnnexplainer"] > Z[u]["sx_fixed"]).astype(float) for u in us], zeros),
              "SX > GNN": hb([(Z[u]["sx_fixed"] > Z[u]["gnnexplainer"]).astype(float) for u in us], zeros),
              "tie": hb([(Z[u]["sx_fixed"] == Z[u]["gnnexplainer"]).astype(float) for u in us], zeros),
              "IG > both": hb([(Z[u]["ig"] > np.maximum(Z[u]["gnnexplainer"], Z[u]["sx_fixed"])).astype(float) for u in us], zeros)}
        means = {w: float(np.mean([Z[u][w].mean() for u in us])) for w in OLD}; best = max(means, key=means.get)
        sig = holm_counts([{(c, k): C.wil(Z[u][c], Z[u][b]) for c in COMBOS for k, b in (("rn", "rn"), ("rule", rule), ("best", best))} for u in us])
        R = {"seeds": len(us), "molecules": [len(Z[u]["mol_idx"]) for u in us], "shares": sh, "best_single": best, "best_means": means, "rule": rule, "combos": {}}
        P(f"\n  {cname(cell)} ({len(us)} seeds, molecules per seed {sorted(set(R['molecules']))})")
        P("    shares: " + "; ".join(f"{k} {v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" for k, v in sh.items()))
        P(f"    best single method (highest mean GEA among the six old methods): {SH[best]} {means[best]:.3f}; best rule {SH[rule]}")
        for c in COMBOS:
            x = {k: hb(g(c), g(b)) for k, b in (("rn", "rn"), ("rule", rule), ("best", best))}
            R["combos"][c] = {**x, "mean": float(np.mean([Z[u][c].mean() for u in us])), "signs": {k: sig[(c, k)] for k in x}}
            P(f"    {SH[c]:<22} mean {R['combos'][c]['mean']:.3f}   - random-node {fmt_ci(*x['rn'])} {C.counts(sig[(c, 'rn')])}   "
              f"- {SH[rule]} {fmt_ci(*x['rule'])} {C.counts(sig[(c, 'rule')])}   - {SH[best]} {fmt_ci(*x['best'])} {C.counts(sig[(c, 'best')])}")
        ok = R["combos"]["rank3"]["best"][2] >= 0; held &= ok; R["P-N5_cell"] = ok; rep["cells"][cell] = R
    P(f"\n  P-N5: rank-average of GNN, SX, IG not clearly below the best single method in all 12 cells -> {'yes' if held else 'no'}")
    for cell in CELL_ORDER[:-1]:
        x = rep["cells"][cell]["combos"]["rank3"]["best"]
        if x[2] < 0:
            P(f"    below in {cname(cell)}: {fmt_ci(*x)} vs {SH[rep['cells'][cell]['best_single']]}")
    P(f"  P-N5 {'HELD' if held else 'FAILED'}"); rep["P-N5"] = held
    save("N4", rep)


# ------------------------------------------------------------------------------------------------ N1
def _aligned(Z, w):
    """(GEA of method w, its nulls / rules on the same molecules) for one unit file."""
    pos = {int(m): i for i, m in enumerate(Z["union"])}; ix = [pos[int(m)] for m in Z[f"{w}__mols"]]
    return Z[f"{w}__gea"], {k: Z[k][ix] for k in ("rn", "rc", *RULES3)}


def N1():
    sc = json.load(open(f"{T5.T5}/scale.json")); rep = {"cells": {}, "levels": {k: v["level"] for k, v in sc.items()}}
    pool = [u for u in T5.n1_units() if os.path.exists(f"{T5.T5}/N1/gea/{u}.npz")]
    Z = {u: dict(np.load(f"{T5.T5}/N1/gea/{u}.npz", allow_pickle=True)) for u in pool}
    P("N1 -- MAIN COMPARISON: new explainers next to the old methods (GEA, final protocol)")
    P("  levels (scale rule): " + ", ".join(f"{SH[k]} {v}" for k, v in rep["levels"].items()))
    P("  each method on its own molecules (old methods on the union of the new methods' molecules); Holm per unit: methods x 2 nulls; "
      "separately methods x 3 rules")
    for cell in CELL_ORDER:
        us = units_of(cell, pool)
        if not us:
            continue
        meths = [w for w in NEWV + OLDN if any(len(Z[u].get(f"{w}__mols", [])) for u in us)]
        sig_n = holm_counts([{(w, k): C.wil(*(lambda a, b: (a, b[k]))(*_aligned(Z[u], w))) for w in meths if len(Z[u].get(f"{w}__mols", [])) for k in ("rn", "rc")} for u in us])
        sig_r = holm_counts([{(w, k): C.wil(*(lambda a, b: (a, b[k]))(*_aligned(Z[u], w))) for w in meths if len(Z[u].get(f"{w}__mols", [])) for k in RULES3} for u in us])
        R = {}
        P(f"\n  {cname(cell)}")
        P(f"    {'method':<16}{'seeds':>6}{'mols':>6}{'mean':>7}   {'- random-node':<34}{'- random-conn.':<34}" + "".join(f"{'- ' + SH[h]:<34}" for h in RULES3))
        for w in meths:
            uw = [u for u in us if len(Z[u].get(f"{w}__mols", []))]
            al = [_aligned(Z[u], w) for u in uw]
            x = {k: hb([a for a, _ in al], [b[k] for _, b in al]) for k in ("rn", "rc", *RULES3)}
            R[w] = {**x, "mean": float(np.mean([a.mean() for a, _ in al])), "seeds": len(uw), "mols": [len(a) for a, _ in al],
                    "signs": {k: (sig_n if k in ("rn", "rc") else sig_r)[(w, k)] for k in x}}
            P(f"    {SH[w]:<16}{len(uw):>6}{int(np.median(R[w]['mols'])):>6}{R[w]['mean']:>7.3f}   "
              + "".join(f"{fmt_ci(*x[k]) + ' ' + C.counts(R[w]['signs'][k]):<34}" for k in ("rn", "rc", *RULES3)))
        order = sorted(R, key=lambda w: -R[w]["mean"])
        P("    ordering by mean GEA: " + " > ".join(f"{SH[w]} {R[w]['mean']:.3f}" for w in order))
        rep["cells"][cell] = {"methods": R, "ordering": order}
    _fid(pool, rep)
    _predictions_n1(rep)
    save("N1", rep)


def _fid(pool, rep):
    P("\n  Fid on the 100-molecule subset (or the method's slice): method minus 20 random rankings, probability and logit, R2 and R3; "
      "counts better/worse/n.s. (Holm per unit and space = methods x 2 metrics x 2 rules)")
    rep["fid"] = {}
    for cell in CELL_ORDER:
        us = [u for u in units_of(cell, pool) if os.path.exists(f"{T5.T5}/N1/fid/{u}.npz")]
        if not us:
            continue
        F = {}
        for u in us:
            z = dict(np.load(f"{T5.T5}/N1/fid/{u}.npz")); F[u] = {"base": z}
            for f in os.listdir(f"{T5.T5}/N1/fid"):
                if f.startswith(f"{u}__"):
                    F[u][f[len(u) + 2:-4]] = dict(np.load(f"{T5.T5}/N1/fid/{f}"))
        meths = [w for w in NEWB + ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig"]
                 if any((f"{w}_R2_fidp_prob" in F[u]["base"]) or (w in F[u]) for u in us)]

        def arr(u, w, key):
            z = F[u]["base"]
            if f"{w}_{key}" in z:
                return z[f"{w}_{key}"], z[f"randnode_{key}"]
            if w in F[u]:
                pos = {int(m): i for i, m in enumerate(z["mol_idx"])}; ix = [pos[int(m)] for m in F[u][w]["mol_idx"]]
                return F[u][w][f"{w}_{key}"], z[f"randnode_{key}"][ix]
            return None
        P(f"\n  {cname(cell)}")
        out = {}
        for space in ("prob", "logit"):
            keys = [f"{r}_{m}_{space}" for r in ("R2", "R3") for m in ("fidp", "fidm")]
            sig = holm_counts([{(w, k): C.wil(*arr(u, w, k)) for w in meths if arr(u, w, keys[0]) is not None for k in keys} for u in us])
            P(f"    [{space}] {'method':<12}" + "".join(f"{k.replace('_' + space, ''):<34}" for k in keys))
            for w in meths:
                uw = [u for u in us if arr(u, w, keys[0]) is not None]; cells = ""
                for k in keys:
                    x = hb([arr(u, w, k)[0] for u in uw], [arr(u, w, k)[1] for u in uw])
                    sg = [flip(s, k.split("_")[1]) for s in sig[(w, k)]]
                    out[f"{space}|{w}|{k}"] = {"ci": x, "signs": sg}; cells += f"{fmt_ci(*x) + ' ' + C.counts(sg):<34}"
                P(f"    [{space}] {SH[w]:<12}{cells}")
        rep["fid"][cell] = out


def _predictions_n1(rep):
    cells = rep["cells"]; P("\n  PREDICTIONS")
    # P-N1
    hits = []
    for w in NEWB:
        r = [cells.get(c, {}).get("methods", {}).get(w) for c in ("bxaic_indole|dmpnn", "bxaic_PAINS|dmpnn")]
        if all(x is not None for x in r) and all(x["rn"][1] > 0 and x["deg_high"][1] > 0 for x in r):
            hits.append(w)
    for w in NEWB:
        for c in ("bxaic_indole|dmpnn", "bxaic_PAINS|dmpnn"):
            x = cells.get(c, {}).get("methods", {}).get(w)
            P(f"    P-N1 {SH[w]:<8} {cname(c):<14} " + ("not run" if x is None else f"- random-node {fmt_ci(*x['rn'])}  - high deg {fmt_ci(*x['deg_high'])}"))
    missing = [w for w in NEWB if any(cells.get(c, {}).get("methods", {}).get(w) is None for c in ("bxaic_indole|dmpnn", "bxaic_PAINS|dmpnn"))]
    rep["P-N1"] = "HELD" if hits else ("NOT DECIDED" if missing else "FAILED")
    P(f"  P-N1 {rep['P-N1']}" + (f" ({', '.join(SH[w] for w in hits)})" if hits else (f" (missing: {', '.join(SH[w] for w in missing)})" if missing else "")))
    # P-N2
    c = "mutagenicity_graphxai|dmpnn"; bad, ran = [], []
    for w in NEWB:
        x = cells.get(c, {}).get("methods", {}).get(w)
        if x is None:
            P(f"    P-N2 {SH[w]:<8} not run"); continue
        ran.append(w); P(f"    P-N2 {SH[w]:<8} - hetero {fmt_ci(*x['hetero'])} -> {cat(x['hetero'])}")
        if x["hetero"][1] > 0:
            bad.append(w)
    rep["P-N2"] = (not bad) if len(ran) == len(NEWB) else ("FAILED" if bad else "not decided")
    P(f"  P-N2 {'HELD' if rep['P-N2'] is True else 'FAILED' if bad else 'NOT DECIDED (missing methods)'}")
    # P-N3: old methods from N4 (registered molecules), new from N1
    Z4 = {u: dict(np.load(f"{T5.T5}/N4/{u}.npz", allow_pickle=True)) for u in T5.n4_units()}
    res = {}
    for w in ["gnnexplainer", "pgexplainer", "sx_fixed", "ixg"] + NEWB:
        row = {}
        for cl in CELL_ORDER[:-1]:
            if w in NEWB:
                x = cells.get(cl, {}).get("methods", {}).get(w); row[cl] = None if x is None else x["rn"]
            else:
                us = units_of(cl, T5.n4_units()); al = []
                for u in us:
                    pos = {int(m): i for i, m in enumerate(Z4[u]["all__mols"])}; ix = [pos[int(m)] for m in Z4[u][f"all__{w}__mols"]]
                    al.append((Z4[u][f"all__{w}__gea"], Z4[u]["all__rn"][ix]))
                row[cl] = hb([a for a, _ in al], [b for _, b in al])
        res[w] = row
    P("    P-N3: minus random-node per cell (above = CI entirely above 0)")
    P(f"    {'method':<12}" + "".join(f"{cname(cl):<21}" for cl in CELL_ORDER[:-1]))
    breakers, undecided = [], []
    for w, row in res.items():
        P(f"    {SH[w]:<12}" + "".join(f"{('-' if v is None else ('above' if v[1] > 0 else 'below' if v[2] < 0 else 'incl. 0')):<21}" for v in row.values()))
        vals = list(row.values())
        if all(v is not None and v[1] > 0 for v in vals):
            breakers.append(w)
        elif all(v is None or v[1] > 0 for v in vals) and any(v is None for v in vals):
            undecided.append(w)
    rep["P-N3_table"] = res
    rep["P-N3"] = "FAILED" if breakers else ("NOT DECIDED" if undecided else "HELD")
    P(f"  P-N3 {rep['P-N3']}" + (f": above random-node in all 12 cells: {', '.join(SH[w] for w in breakers)}" if breakers else "")
      + (f"; not decided because of missing cells: {', '.join(SH[w] for w in undecided)}" if undecided and not breakers else ""))
    nb = [w for w in breakers if w in NEWB]
    P(f"    secondary (new methods only): {'some new method is above in all 12 cells: ' + ', '.join(SH[w] for w in nb) if nb else 'no new method is above in all 12 cells'}")


# ------------------------------------------------------------------------------------------------ N2
def N2():
    from .tier5_N23 import KINDS, N2_UNITS
    rep = {}; Z = {u: dict(np.load(f"{T5.T5}/N2/score/{u}.npz", allow_pickle=True)) for u in N2_UNITS}
    P("N2 -- SANITY CHECK: explanation of the trained model vs of the reset model (Tier-2 R checkpoints), same molecules, seeds 0-4")
    P("  GEA(trained) - GEA(reset), Spearman of atom scores (trained vs reset), Jaccard of top-k sets; Holm per (dataset, kind, seed) over the method versions")
    for d in ("mutagenicity_dmpnn", "bxaic_indole"):
        us = [u for u in N2_UNITS if u.startswith(d)]
        for kind in KINDS:
            ws = [w for w in NEWV if f"{w}__{kind}__trained" in Z[us[0]]]
            sig = holm_counts([{w: C.wil(Z[u][f"{w}__{kind}__trained"], Z[u][f"{w}__{kind}__reset"]) for w in ws} for u in us])
            P(f"\n  {d} / {kind} reset")
            for w in ws:
                x = hb([Z[u][f"{w}__{kind}__trained"] for u in us], [Z[u][f"{w}__{kind}__reset"] for u in us])
                sp = [np.nanmean(Z[u][f"{w}__{kind}__spearman"]) for u in us]; jc = [np.mean(Z[u][f"{w}__{kind}__jaccard"]) for u in us]
                rep[f"{d}|{kind}|{w}"] = {"diff": x, "signs": sig[w], "spearman_per_seed": sp, "jaccard_per_seed": jc,
                                          "undefined_spearman": int(sum(np.isnan(Z[u][f"{w}__{kind}__spearman"]).sum() for u in us))}
                P(f"    {SH[w]:<16} trained - reset {fmt_ci(*x)} {C.counts(sig[w])}   Spearman {np.mean(sp):+.3f} (seeds {min(sp):+.2f}..{max(sp):+.2f})   "
                  f"Jaccard {np.mean(jc):.3f}   undefined Spearman {rep[f'{d}|{kind}|{w}']['undefined_spearman']}")
    ok = [w for w in NEWB if all(f"{d}|full|{w}" in rep and rep[f"{d}|full|{w}"]["diff"][1] > 0 for d in ("mutagenicity_dmpnn", "bxaic_indole"))]
    ran = [w for w in NEWB if all(f"{d}|full|{w}" in rep for d in ("mutagenicity_dmpnn", "bxaic_indole"))]
    P("\n  P-N4: every new explainer: CI(trained - full reset) entirely above 0 on both Mutagenicity D-MPNN and indole")
    for w in NEWB:
        P(f"    {SH[w]:<8} " + "  ".join(f"{d}: " + (fmt_ci(*rep[f'{d}|full|{w}']['diff']) if f"{d}|full|{w}" in rep else "not run") for d in ("mutagenicity_dmpnn", "bxaic_indole")))
    held = len(ok) == len(NEWB); rep["P-N4"] = "HELD" if held else ("FAILED" if [w for w in ran if w not in ok] else "NOT DECIDED")
    P(f"  P-N4 {rep['P-N4']}")
    save("N2", rep)


# ------------------------------------------------------------------------------------------------ N3
def N3():
    from .tier5_N23 import N3_TAGS, N3_UNITS
    rep = {}
    P("N3 -- STABILITY: repeated runs with different explainer seeds on the fixed 100-molecule subsets (or the level's slice), seed-0 models")
    for u in N3_UNITS:
        Z = dict(np.load(f"{T5.T5}/N3/score/{u}.npz", allow_pickle=True)); P(f"\n  {u}")
        for name in N3_TAGS:
            if f"{name}__jaccard" not in Z:
                continue
            J = Z[f"{name}__jaccard"]; md = Z[f"{name}__maxdiff"]
            r = {"molecules": len(J), "jaccard_mean": float(J.mean()), "jaccard_median": float(np.median(J)), "identical_share": float(np.mean(J == 1.0)),
                 "share_ge2": float(np.nanmean(Z[f"{name}__share_ge2"])), "max_score_diff": float(md.max()),
                 "consensus_minus_single": hb([Z[f"{name}__consensus_set"]], [Z[f"{name}__single_set"]]),
                 "protocol_minus_rn": hb([Z[f"{name}__protocol"]], [Z[f"{name}__rn"]])}
            rep[f"{u}|{name}"] = r
            if name == "eig":
                P(f"    EiG       {len(J)} molecules: two runs identical on {int((md == 0).sum())} of {len(md)} molecules (max |score difference| {md.max():.2e})")
            else:
                P(f"    {SH[name]:<9} {len(J)} molecules: Jaccard mean {r['jaccard_mean']:.3f}, median {r['jaccard_median']:.3f}; identical in all pairs {r['identical_share']:.2f}; "
                  f"atoms in >= 2 of 3 runs {r['share_ge2']:.3f}; consensus - single (sets) {fmt_ci(*r['consensus_minus_single'])}; "
                  f"final protocol - random-node {fmt_ci(*r['protocol_minus_rn'])}")
    save("N3", rep)


# ------------------------------------------------------------------------------------------------ follow-ups F1, F4-F6
TCELLS = CELL_ORDER[:-1]
OLDR = {"gnnexplainer": "gnnexplainer", "pgexplainer": "pgexplainer", "sx_fixed": "subgraphx", "ig": "ig", "sal": "sal"}   # Tier 2 R_rescore keys
SANITY_CELL = {"mutagenicity_graphxai|dmpnn": ("mutagenicity_dmpnn", "Mutagenicity D-MPNN"), "bxaic_indole|dmpnn": ("bxaic_indole", "B-XAIC indole")}
ROWS = OLDN + NEWB + ["eig_own", "same_own"]
SYM = {"above": "+", "below": "-", "includes 0": "0"}


def _ld(f):
    return dict(np.load(f, allow_pickle=True))


def _F1(rep):
    from .tier5_N23 import N3_TAGS, N3_UNITS
    P("F1 -- STABILITY (exploratory; PLAN_followups.md)")
    P("  Tier 4 L6 covered SubgraphX only and only run-to-run (3 explainer seeds, seed-0 model); there is no Tier-4 GNNExplainer or cross-training-seed report.")
    P("\n  (a) run to run (registered N3), Tier 4 L6 format; seed-0 models; T5a/T5b/T5c (EiG T5a/T5b)")
    a = {}
    for u in N3_UNITS:
        Z = _ld(f"{T5.T5}/N3/score/{u}.npz"); X = _ld(f"{T5.T5}/F/n3/{u}.npz"); P(f"\n  {u}")
        for name in N3_TAGS:
            J = Z[f"{name}__jaccard"]; md = Z[f"{name}__maxdiff"]; n = len(J)
            r = {"molecules": n, "jaccard_mean": float(J.mean()), "jaccard_median": float(np.median(J)), "identical_share": float(np.mean(J == 1.0)),
                 "vs_n1": float(X[f"{name}__vs_n1"].mean()), "share_ge2": float(np.nanmean(Z[f"{name}__share_ge2"])),
                 "cons_size": float(X[f"{name}__cons_size"].mean()), "k": float(X[f"{name}__k"].mean()),
                 "single": float(Z[f"{name}__single_set"].mean()), "consensus": float(Z[f"{name}__consensus_set"].mean()),
                 "cons_minus_single": hb([Z[f"{name}__consensus_set"]], [Z[f"{name}__single_set"]]),
                 "single_minus_rn": hb([Z[f"{name}__single_set"]], [Z[f"{name}__rn"]]), "cons_minus_rn": hb([Z[f"{name}__consensus_set"]], [Z[f"{name}__rn"]]),
                 "protocol_minus_rn": hb([Z[f"{name}__protocol"]], [Z[f"{name}__rn"]]), "identical_scores": int((md == 0).sum()), "max_score_diff": float(md.max())}
            a[f"{u}|{name}"] = r
            P(f"    {SH[name]} ({n} molecules, {len(N3_TAGS[name])} runs)")
            if name == "eig":
                P(f"      registered check (PLAN: EiG output identical): {r['identical_scores']} of {n} molecules identical, max |score difference| {r['max_score_diff']:.2e} "
                  f"-> {'PASSED' if r['identical_scores'] == n else 'FAILED'}")
            P(f"      Jaccard between runs: mean {r['jaccard_mean']:.3f}, median {r['jaccard_median']:.3f}; molecules with identical sets in all pairs {r['identical_share']:.2f}; "
              f"vs the N1 run {r['vs_n1']:.3f}")
            P(f"      atoms picked in >= 2 of {len(N3_TAGS[name])} runs, as a share of atoms picked at least once: {r['share_ge2']:.3f}; consensus set size {r['cons_size']:.1f} (k = {r['k']:.1f})")
            P(f"      GEA (sets scored as sets): single run mean {r['single']:.3f}, consensus {r['consensus']:.3f}; consensus - single {fmt_ci(*r['cons_minus_single'])}")
            P(f"      minus random-node: single {fmt_ci(*r['single_minus_rn'])}, consensus {fmt_ci(*r['cons_minus_rn'])}, single under the final protocol {fmt_ci(*r['protocol_minus_rn'])}")
    rep["F1a"] = a
    P("\n  (b) determinism: same model, same seed (tag T5), run again on the first 20 molecules of the method's list; compared with the N1 rows")
    b = {}
    for u in N3_UNITS:
        Z = _ld(f"{T5.T5}/F/det/{u}.npz")
        for name in T5.NEW:
            idn = Z[f"{name}__identical"]; md = Z[f"{name}__maxdiff"]; jc = Z[f"{name}__jaccard"]
            b[f"{u}|{name}"] = {"molecules": len(idn), "identical": int(idn.sum()), "max_diff": float(md.max()), "jaccard_mean": float(jc.mean()),
                                "deterministic": bool(idn.all())}
            P(f"    {u:<24} {SH[name]:<8} identical {int(idn.sum())}/{len(idn)}   max |score difference| {md.max():.2e}   top-k Jaccard vs N1 {jc.mean():.3f}"
              f"   -> {'deterministic' if idn.all() else 'NOT deterministic'}")
    rep["F1b"] = b
    P("\n  (c) across training seeds: top-k Jaccard between the explanations of two models (different seeds) of the same molecule;")
    P("      only molecules explained under both seeds (test splits differ by seed); mean over (seed pair, molecule); random = exact expected Jaccard of two random k-sets")
    c = {}
    for cell in CELL_ORDER:
        f = f"{T5.T5}/F/cross/{cell.replace('|', '__')}.npz"
        Z = _ld(f); P(f"    {cname(cell)}")
        line = []
        for w in OLDN + NEWB:
            J = Z[f"{w}__jac"]; B = Z[f"{w}__base"]; npair = len(set(Z[f"{w}__pair"].tolist()))
            ok = len(J) >= 20
            c[f"{cell}|{w}"] = {"items": len(J), "pairs": npair, "jaccard": float(J.mean()) if ok else None, "random": float(B.mean()) if ok else None}
            line.append(f"{SH[w]} " + (f"{J.mean():.2f} (rand {B.mean():.2f}, n {len(J)})" if ok else f"n.c. (n {len(J)})"))
        P("      " + "; ".join(line))
    rep["F1c"] = c


def _old_rn_rule(rep):
    """Old methods on their registered molecules (P-N3 sources): minus random-node, minus the dataset's rule, per-seed Holm signs vs random-node."""
    Z4 = {u: _ld(f"{T5.T5}/N4/{u}.npz") for u in T5.n4_units()}; E = {u: _ld(f"{T5.T5}/F/extract/{u}.npz") for u in T5.n4_units()}
    out = {}
    for cell in TCELLS:
        us = units_of(cell, T5.n4_units()); al = {}
        for u in us:
            pos = {int(m): i for i, m in enumerate(Z4[u]["all__mols"])}
            assert [int(m) for m in E[u]["all__mols"]] == [int(m) for m in Z4[u]["all__mols"]], u
            for w in OLDN:
                ix = [pos[int(m)] for m in Z4[u][f"all__{w}__mols"]]
                al.setdefault(w, []).append((Z4[u][f"all__{w}__gea"], Z4[u]["all__rn"][ix], Z4[u]["all__rc"][ix], E[u]["all__rule"][ix]))
        sig = holm_counts([{(w, k): C.wil(al[w][i][0], al[w][i][j]) for w in OLDN for k, j in (("rn", 1), ("rc", 2))} for i in range(len(us))])
        for w in OLDN:
            out[f"{cell}|{w}"] = {"rn": hb([x[0] for x in al[w]], [x[1] for x in al[w]]), "rule": hb([x[0] for x in al[w]], [x[3] for x in al[w]]),
                                  "signs_rn": sig[(w, "rn")], "seeds": len(us)}
    # must match the P-N3 table of N1
    for w, row in rep["N1"]["P-N3_table"].items():
        if w in OLDN:
            for cell, v in row.items():
                assert np.allclose(v, out[f"{cell}|{w}"]["rn"]), (w, cell)
    return out


def _F4(rep):
    P("\nF4 -- SUMMARY TABLE (rows = methods; per benchmark x model: above random / above best rule / sanity / stable across seeds / seconds per molecule)")
    n1, n2, rr = rep["N1"], rep["N2"], json.load(open(f"{C.T2}/R_rescore.json"))["trained_minus_full"]
    old = _old_rn_rule(rep); E = {u: _ld(f"{T5.T5}/F/extract/{u}.npz") for u in T5.n1_units()}
    rows = {}
    for w in ROWS:
        for cell in TCELLS:
            d = cell.split("|")[0]; rule = T5.RULE[d]; r = {}
            if w in OLDN:
                o = old[f"{cell}|{w}"]; r["random"] = cat(o["rn"]); r["rule"] = cat(o["rule"]); signs = o["signs_rn"]; r["random_ci"] = o["rn"]; r["rule_ci"] = o["rule"]
            else:
                m = n1["cells"].get(cell, {}).get("methods", {}).get(w)
                if m is None:
                    rows[(w, cell)] = {"random": "not run"}; continue
                r["random"] = cat(m["rn"]); r["rule"] = cat(m[rule]); signs = m["signs"]["rn"]; r["random_ci"] = m["rn"]; r["rule_ci"] = m[rule]
            if cell in SANITY_CELL:
                key, lab = SANITY_CELL[cell]
                if w in OLDN:
                    x = rr.get(f"{lab}|{OLDR.get(w, '-')}"); r["sanity"] = "not tested" if x is None else x["cat"]; r["sanity_ci"] = None if x is None else x["diff"]
                else:
                    x = n2.get(f"{key}|full|{w}"); r["sanity"] = "not tested" if x is None else cat(x["diff"]); r["sanity_ci"] = None if x is None else x["diff"]
            else:
                r["sanity"] = "not tested"
            r["same_verdict"] = "yes" if len(set(signs)) == 1 else "no"; r["verdict_counts"] = C.counts(signs)
            cx = rep["F1c"].get(f"{cell}|{w.replace('_own', '')}") if not w.endswith("_own") else None
            r["xseed_jaccard"] = None if not cx else cx["jaccard"]; r["xseed_random"] = None if not cx else cx["random"]
            base = w.replace("_own", "")
            if base in T5.NEW:
                sec = [E[u][f"{base}__sec"] for u in units_of(cell, T5.n1_units()) if f"{base}__sec" in E[u]]
                r["sec"] = float(np.mean(np.concatenate(sec))) if sec else None
            else:
                r["sec"] = None
            rows[(w, cell)] = r
    rep["F4"] = {f"{w}|{c}": v for (w, c), v in rows.items()}
    # CSV (wide)
    hdr = ["method"] + [f"{CELL_NAME[c.split('|')[0]]} {c.split('|')[1]}|{f}" for c in TCELLS for f in ("above_random", "above_best_rule", "sanity_trained_vs_reset",
                                                                                                         "same_verdict_all_seeds", "xseed_topk_jaccard", "xseed_random_baseline", "seconds_per_molecule")]
    lines = [",".join(hdr)]
    for w in ROWS:
        vals = [SH[w]]
        for c in TCELLS:
            r = rows[(w, c)]
            if r.get("random") == "not run":
                vals += ["not run"] + [""] * 6; continue
            vals += [r["random"], r["rule"], r["sanity"], f"{r['same_verdict']} ({r['verdict_counts']})",
                     "" if r["xseed_jaccard"] is None else f"{r['xseed_jaccard']:.3f}", "" if r["xseed_random"] is None else f"{r['xseed_random']:.3f}",
                     "not recorded" if r["sec"] is None else f"{r['sec']:.2f}"]
        lines.append(",".join(f'"{v}"' for v in vals))
    open(f"{T5.T5}/summary_table.csv", "w").write("\n".join(lines) + "\n")
    # markdown: one table per benchmark
    md = ["# RQ1 Tier 5 summary table", "",
          "Each cell is: **above random / above best rule / sanity / stable / seconds per molecule**.", "",
          "- **Above random / above best rule.** `+` means the CI is entirely above 0, `-` entirely below 0, `0` includes 0. Old methods are on their registered molecules; new methods are N1, each at its scale level. The best rule is high degree for indole and PAINS and hetero for Mutagenicity.",
          "- **Sanity.** GEA trained minus full reset: `+` / `-` / `0`. Old methods use Tier 2 R_rescore, where PG is the final-rule version; new methods use N2. n.t. = not tested; only the D-MPNN on Mutagenicity and indole was tested.",
          "- **Stable.** `yes` means every seed has the same Holm verdict against random-node; the numbers in brackets are better/worse/n.s. counts. J is the top-k Jaccard across training seeds on shared molecules, with the random baseline in brackets; n.c. = fewer than 20 shared items.",
          "- **Seconds per molecule.** New methods only, measured with 8-12 concurrent jobs on one shared V100. SAME's time is the sum of its two budget runs. The own-size versions come from the same run. Old methods: not recorded.", ""]
    for d in ("bxaic_indole", "bxaic_PAINS", "mutagenicity_graphxai"):
        cs = [c for c in TCELLS if c.startswith(d + "|")]
        md += [f"## {CELL_NAME[d]}", "", "| method | " + " | ".join(cname(c).split(' ', 1)[1] for c in cs) + " |", "|---|" + "---|" * len(cs)]
        for w in ROWS:
            cellv = []
            for c in cs:
                r = rows[(w, c)]
                if r.get("random") == "not run":
                    cellv.append("not run"); continue
                st = f"{r['same_verdict']} ({r['verdict_counts']})" + ("" if r["xseed_jaccard"] is None else f", J {r['xseed_jaccard']:.2f} ({r['xseed_random']:.2f})")
                cellv.append(f"{SYM[r['random']]} / {SYM[r['rule']]} / {SYM.get(r['sanity'], 'n.t.')} / {st} / " + ("n.r." if r["sec"] is None else f"{r['sec']:.1f} s"))
            md.append(f"| {SH[w]} | " + " | ".join(cellv) + " |")
        md.append("")
    open(f"{T5.T5}/summary_table.md", "w").write("\n".join(md))
    P("  written: summary_table.csv, summary_table.md")


def _F5(rep):
    P("\nF5 -- EiG: GEA at its own chosen size vs at the fixed top-25% budget (N1 molecules, all seeds)")
    P("  own size is scored as a set; set GEA moves with set size, so it is also compared with a size-matched random-node null (20 random sets of the same size)")
    P(f"    {'cell':<20}{'seeds':>6}  {'top-25%: GEA':>12}  {'- random-node':<26}{'own size: GEA':>14}  {'- random-node':<26}{'- size-matched null':<26}{'own size / atoms':>17}{'at cap':>8}")
    n1 = rep["N1"]["cells"]; out = {}
    for cell in CELL_ORDER:
        us = units_of(cell, T5.n1_units()); E = {u: _ld(f"{T5.T5}/F/extract/{u}.npz") for u in us}; G = {u: _ld(f"{T5.T5}/N1/gea/{u}.npz") for u in us}
        for u in us:   # the extract's own-set GEA must equal N1's (same molecules; compared by molecule)
            e = dict(zip(map(int, E[u]["eig__mols"]), E[u]["eig_own__gea"])); g = dict(zip(map(int, G[u]["eig_own__mols"]), G[u]["eig_own__gea"]))
            assert set(e) == set(g) and all(abs(e[m] - g[m]) < 1e-12 for m in e), u
        f, o = n1[cell]["methods"]["eig"], n1[cell]["methods"]["eig_own"]
        sz = hb([E[u]["eig_own__gea"] for u in us], [E[u]["eig_own__sizenull"] for u in us])
        frac = [float(E[u]["eig__own_frac"].mean()) for u in us]; cap = float(np.mean(np.concatenate([E[u]["eig__chosen_bonds"] == E[u]["eig__max_bonds"] for u in us])))
        out[cell] = {"fixed_mean": f["mean"], "fixed_rn": f["rn"], "own_mean": o["mean"], "own_rn": o["rn"], "own_sizenull": sz, "own_frac_mean": float(np.mean(frac)),
                     "own_frac_seed_range": [min(frac), max(frac)], "share_at_cap": cap}
        P(f"    {cname(cell):<20}{len(us):>6}  {f['mean']:>12.3f}  {fmt_ci(*f['rn']):<26}{o['mean']:>14.3f}  {fmt_ci(*o['rn']):<26}{fmt_ci(*sz):<26}"
          f"{np.mean(frac):>11.3f} ({min(frac):.2f}-{max(frac):.2f})  {cap:>6.2f}")
    P("  own size / atoms: mean over seeds of the mean fraction (seed range); at cap: share of molecules where EiG chose the maximum number of bonds floor(0.3 x bonds)")
    rep["F5"] = out


def _F6(rep):
    P("\nF6 -- EiG FIDELITY (NOT an independent check: EiG's search maximises Fid+ - Fid-) and GEA-only rankings")
    fid = rep["N1"]["fid"]; out = {}
    P("  EiG Fid on the 100-molecule subsets, minus 20 random rankings, probability space (counts better/worse/n.s.); same numbers as N1.txt")
    for cell in CELL_ORDER:
        if cell not in fid:
            continue
        ks = [f"{r}_{m}_prob" for r in ("R2", "R3") for m in ("fidp", "fidm")]
        P(f"    {cname(cell):<20}" + "  ".join(f"{k.replace('_prob', '')} " + (fmt_ci(*fid[cell][f'prob|eig|{k}']['ci']) + ' ' + C.counts(fid[cell][f'prob|eig|{k}']['signs'])
                                                                        if f"prob|eig|{k}" in fid[cell] else "n/a") for k in ks))
        # rank by Fid+ R2 prob (point estimate) next to GEA rankings
        fr = {w: fid[cell][f"prob|{w}|R2_fidp_prob"]["ci"][0] for w in NEWB + ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig"] if f"prob|{w}|R2_fidp_prob" in fid[cell]}
        out.setdefault(cell, {})["fid_rank"] = sorted(fr, key=lambda w: -fr[w])
    P("\n  Rankings per cell. Fid: by Fid+ R2 (probability) minus random. GEA (1): by GEA minus random-node, each method on its own molecules (N1).")
    P("  GEA (2): by mean GEA on a common set = the SAME L4 slice (25 molecules per seed, seeds 0-4), which every method covers")
    for cell in CELL_ORDER:
        m = rep["N1"]["cells"][cell]["methods"]; ws = [w for w in NEWB + OLDN if w in m]
        g1 = sorted(ws, key=lambda w: -m[w]["rn"][0])
        us = [u for u in units_of(cell, T5.n1_units()) if T5.seed_of(u) < 5]; Z = {u: _ld(f"{T5.T5}/N1/gea/{u}.npz") for u in us}
        com = {}
        for w in NEWB + OLDN:
            v = []
            for u in us:
                if f"{w}__mols" not in Z[u] or "same__mols" not in Z[u] or not len(Z[u]["same__mols"]):
                    v = None; break
                pos = {int(x): i for i, x in enumerate(Z[u][f"{w}__mols"])}
                if not all(int(x) in pos for x in Z[u]["same__mols"]):
                    v = None; break
                v.append(Z[u][f"{w}__gea"][[pos[int(x)] for x in Z[u]["same__mols"]]].mean())
            if v is not None:
                com[w] = float(np.mean(v))
        g2 = sorted(com, key=lambda w: -com[w])
        out.setdefault(cell, {}).update({"gea_rank_own": g1, "gea_rank_common": g2, "gea_common": com})
        P(f"    {cname(cell)}")
        if "fid_rank" in out[cell]:
            P("      Fid+ (R2) : " + " > ".join(SH[w] for w in out[cell]["fid_rank"]))
        P("      GEA (1)   : " + " > ".join(f"{SH[w]} {m[w]['rn'][0]:+.3f}" for w in g1))
        P("      GEA (2)   : " + " > ".join(f"{SH[w]} {com[w]:.3f}" for w in g2))
    rep["F6"] = out


def F():
    rep = {"N1": json.load(open(f"{T5.T5}/N1.json")), "N2": json.load(open(f"{T5.T5}/N2.json"))}
    _F1(rep); _F4(rep); _F5(rep); _F6(rep)
    del rep["N1"], rep["N2"]
    save("F", rep)


if __name__ == "__main__":
    {"N0": N0, "N4": N4, "N1": N1, "N2": N2, "N3": N3, "F": F}[sys.argv[1]]()
