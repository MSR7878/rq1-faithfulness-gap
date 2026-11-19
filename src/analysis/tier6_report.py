"""
RQ1 Tier 6 reports (runs/rq1_tier6/PLAN.md). Every number is method minus null (or trained minus reset) with the hierarchical
bootstrap 95% CI (seeds, then molecules); counts are per-seed better / worse / n.s. after Holm (families as in the PLAN).
    python -m src.analysis.tier6_report T1|T2|T3|T4      -> runs/rq1_tier6/<ITEM>.txt / .json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from . import revision_analysis as RA
from . import tier2_common as C
from . import tier5_common as T5
from .tier2_report import fmt_ci
from .tier4_report import holm_counts
from .tier6_explain import T2_METHODS, T2_UNITS, T3_UNITS, T4_UNITS, T6, subset100

LOG = []
SH = {"eig": "EiG", "eig_own": "EiG (own size)", "same": "SAME", "gstarx": "GStarX", "gnnshap": "GNNShap", "gnnexplainer": "GNN",
      "sx_fixed": "SX fixed", "sal": "Saliency", "ig": "IG", "ixg": "IxG", "pgexplainer": "PG"}


def P(s=""):
    print(s, flush=True); LOG.append(s)


def save(name, rep):
    open(f"{T6}/{name}.txt", "w").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{T6}/{name}.json", "w"), indent=1, default=float)


def load(f):
    return dict(np.load(f, allow_pickle=True))


def hb(real, null):
    p, lo, hi, _ = RA.hboot(real, null); return [float(p), float(lo), float(hi)]


def cat(ci):
    return C.ci_cat(ci[1], ci[2])


def cnt(signs):
    return f"{signs.count('+')}/{signs.count('-')}/{signs.count('.')}"


def pairs(a, b, keep=None):
    """drop molecules where either value is NaN (and those not in keep)"""
    a = np.asarray(a, float); b = np.asarray(b, float); ok = ~(np.isnan(a) | np.isnan(b))
    if keep is not None:
        ok &= np.asarray(keep, bool)
    return a[ok], b[ok]


def table(rows, fam):
    """rows: {key: (real_by_seed, null_by_seed)}; Holm within each seed over the keys in fam. -> {key: dict}"""
    S = len(next(iter(rows.values()))[0]); raw = [{} for _ in range(S)]
    for k, (r, n) in rows.items():
        for s in range(S):
            if k in fam and len(r[s]):
                raw[s][k] = C.wil(r[s], n[s])
    sig = holm_counts(raw); out = {}
    for k, (r, n) in rows.items():
        keep = [s for s in range(S) if len(r[s])]
        ci = hb([r[s] for s in keep], [n[s] for s in keep])
        out[k] = {"mean": float(np.mean([np.mean(r[s]) for s in keep])), "null": float(np.mean([np.mean(n[s]) for s in keep])), "diff": ci, "cat": cat(ci),
                  "counts": cnt(sig.get(k, [])) if k in fam else "-", "seeds": len(keep), "mols_per_seed": [int(len(r[s])) for s in range(S)]}
    return out


# ------------------------------------------------------------------------------------------------ T1
T1M = ["eig", "gnnshap", "same", "gstarx", "sal", "ig"]


def T1():
    Z = [load(f"{T6}/T1/{u}.npz") for u in T2_UNITS]; rep = {}
    P("T1 -- MUTAG-GT D-MPNN, 10 seeds: share of the chosen atoms that are aromatic (top-k, random ties, 20 orders), minus 20 random-connected sets of the same size")
    P("  molecules: each method's molecules with 0 < |aromatic scaffold| < N (the 14 without aromatic atoms are left out)")
    P("  GStarX / SAME: seeds 0-4 = Tier 5 N1 runs, seeds 5-9 = new runs (runs/rq1_tier6/T1/expl). counts = higher/lower/n.s., Holm per seed = 6 (EiG own size not in the family)")
    P("\n  [overlap] share of chosen atoms that are aromatic, minus random-connected sets of the same size")
    P(f"  {'method':<16}{'set':<6}{'mean share':>11}{'null share':>11}{'difference [95% CI]':>30}{'counts':>9}   molecules per seed")
    for st in ("all", "test"):
        rows = {}
        for w in T1M + ["eig_own"]:
            r, n = [], []
            for z in Z:
                keep = z[f"{w}__is_test"].astype(bool) if st == "test" else None
                a, b = pairs(z[f"{w}__share"], z[f"{w}__conn"], keep); r.append(a); n.append(b)
            rows[w] = (r, n)
        res = table(rows, T1M); rep[st] = res
        for w in T1M + ["eig_own"]:
            o = res[w]; small = "  (small)" if st == "test" and w in ("same", "gstarx") else ""
            P(f"  {SH[w]:<16}{st:<6}{o['mean']:>11.3f}{o['null']:>11.3f}{fmt_ci(*o['diff']):>30}{o['counts']:>9}   {o['mols_per_seed']}{small}")
    sz = {w: float(np.mean(np.concatenate([z[f'{w}__size'] for z in Z]))) for w in ("eig", "eig_own")}
    P(f"\n  mean set size: top-k {sz['eig']:.2f} atoms; EiG own size {sz['eig_own']:.2f} atoms")
    P("\n  for reference, Tier 4 L4 (runs/rq1_tier4/L4.txt, not recomputed):")
    on = False
    for line in open("runs/rq1_tier4/L4.txt").read().splitlines():
        if line.startswith("  [overlap]"):
            on = True
        if on and line.strip().startswith(("SX fixed", "GNN ", "IG ")):
            P("  " + line)
    save("T1", rep)


# ------------------------------------------------------------------------------------------------ T2
def T2():
    Z = [load(f"{T6}/T2/{u}.npz") for u in T2_UNITS]; rep = {}
    P("T2 -- MUTAG-GT D-MPNN, 10 seeds: GEA minus random-node when the aromatic rings are not available")
    P("  counts = better/worse/n.s. vs random-node, Holm per seed and condition = 5 methods")
    nm = len(Z[0]["a__mols"]); nb = len(Z[0]["b__mols"])

    def block(title, key, rows_of):
        rows = {w: rows_of(w) for w in T2_METHODS}; res = table(rows, T2_METHODS); rep[key] = res
        P(f"\n  {title}")
        P(f"  {'method':<10}{'GEA':>8}{'null':>8}{'GEA - random-node [95% CI]':>32}{'':>3}{'category':<12}{'counts':>8}")
        for w in T2_METHODS:
            o = res[w]; P(f"  {SH[w]:<10}{o['mean']:>8.3f}{o['null']:>8.3f}{fmt_ci(*o['diff']):>32}   {o['cat']:<12}{o['counts']:>8}")
        return res
    # before
    block(f"[before] full graph, all {nm + nb} molecules (= Tier 5 N1 protocol)", "before_all",
          lambda w: ([np.concatenate([z[f"a__{w}"], z[f"full__{w}"]]) for z in Z], [np.concatenate([z["a__rn"], z["full__rn"]]) for z in Z]))
    block(f"[before] full graph, the {nb} molecules with an aromatic scaffold", "before_174",
          lambda w: ([z[f"full__{w}"] for z in Z], [z["full__rn"] for z in Z]))
    A = block(f"(a) the {nm} molecules with no aromatic atom (the same molecules in every seed; full graph)", "a",
              lambda w: ([z[f"a__{w}"] for z in Z], [z["a__rn"] for z in Z]))
    kp = Z[0]["b__k_p"]; nr = Z[0]["b__nrem"]
    P(f"\n  (b) scaffold hidden, {nb} molecules; remaining atoms: mean {nr.mean():.1f} (min {nr.min()}, max {nr.max()})")
    P(f"      primary k = max(1, floor(0.25 n_remaining)): k = 1 for {np.mean(kp == 1):.0%} of molecules; random-node GEA on the remaining atoms: mean {np.mean([z['b__rn_p'].mean() for z in Z]):.3f}")
    P(f"      secondary k = min(n_remaining, floor(0.25 N)): mean k {np.mean(Z[0]['b__k_s']):.2f}; random-node GEA mean {np.mean([z['b__rn_s'].mean() for z in Z]):.3f}")
    B = {}
    for rule in ("R1", "R2"):
        for t, lab in (("p", "primary k"), ("s", "secondary k")):
            def rows_of(w, rule=rule, t=t):
                r, n = [], []
                for z in Z:
                    a, b = pairs(z[f"b__{rule}__{w}__gea_{t}"], z[f"b__rn_{t}"]); r.append(a); n.append(b)
                return r, n
            B[(rule, t)] = block(f"(b)-{rule}, {lab}", f"b_{rule}_{t}", rows_of)
        P(f"      {rule}: predicted class changed by hiding the scaffold in {np.mean([z[f'b__{rule}__eig__flip'].mean() for z in Z]):.1%} of molecules (mean over seeds)")
        for w in T2_METHODS:
            e = int(sum(z[f"b__{rule}__{w}__error"].sum() for z in Z)); ti = np.mean([z[f"b__{rule}__{w}__tied"].mean() for z in Z])
            sec = np.mean([z[f"b__{rule}__{w}__sec"].mean() for z in Z])
            P(f"      {rule} {SH[w]:<9} errors {e:>3}; scores on the remaining atoms all tied in {ti:.1%} of molecules; {sec:.2f} s per molecule")
        rep[f"flip_{rule}"] = float(np.mean([z[f'b__{rule}__eig__flip'].mean() for z in Z]))
    # P-T2
    cells = {("eig", "a"): A["eig"], ("gnnshap", "a"): A["gnnshap"], ("eig", "R1"): B[("R1", "p")]["eig"], ("gnnshap", "R1"): B[("R1", "p")]["gnnshap"],
             ("eig", "R2"): B[("R2", "p")]["eig"], ("gnnshap", "R2"): B[("R2", "p")]["gnnshap"]}
    ok = {f"{w}|{c}": o["diff"][1] > 0 for (w, c), o in cells.items()}
    P("\n  P-T2 (EiG and GNNShap above random-node in (a), (b)-R1 and (b)-R2, primary k; all 6 needed):")
    for (w, c), o in cells.items():
        P(f"    {SH[w]:<8} {c:<3} {fmt_ci(*o['diff'])}  -> {'above' if o['diff'][1] > 0 else 'not above'}")
    held = all(ok.values()); P(f"  P-T2 {'HELD' if held else 'FAILED'} ({sum(ok.values())} of 6 cells above)")
    rep["P-T2"] = {"held": held, "cells": ok}
    save("T2", rep)


# ------------------------------------------------------------------------------------------------ T3
CELLS3 = [("bxaic_PAINS", "PAINS D-MPNN"), ("bxaic_indole_gcn", "indole GCN"), ("bxaic_PAINS_gcn", "PAINS GCN"), ("mutagenicity_gcn", "Mutagenicity GCN")]
M3 = ["eig", "gnnshap", "sal", "ig"]


def T3():
    rep = {}
    P("T3 -- FULL-RESET SANITY CHECK (as N2): GEA(trained) - GEA(full reset), the unit's 100-molecule subset, seeds 0-4")
    P("  counts = trained better/worse/n.s., Holm per seed = 4 methods. Passes = CI entirely above 0.")
    P("  Spearman = trained vs reset atom scores (mean over molecules and seeds); Jaccard = trained vs reset top-k sets")
    for base, name in CELLS3:
        us = [f"{base}_s{s}" for s in range(5)]; Z = [load(f"{T6}/T3/{u}.npz") for u in us]
        rows = {w: ([z[f"{w}__trained"] for z in Z], [z[f"{w}__reset"] for z in Z]) for w in M3}; res = table(rows, M3)
        P(f"\n  {name}: trained acc {[round(float(z['trained_acc']), 3) for z in Z]}, AUROC {[round(float(z['trained_auroc']), 3) for z in Z]}")
        P(f"  {'':<14}reset acc {[round(float(z['reset_acc']), 3) for z in Z]}, AUROC {[round(float(z['reset_auroc']), 3) for z in Z]}, "
          f"predicted positive {[round(float(z['reset_frac_pos']), 2) for z in Z]}")
        P(f"  {'method':<10}{'trained':>9}{'reset':>8}{'trained - reset [95% CI]':>30}{'counts':>8}{'Spearman':>10}{'Jaccard':>9}   passes")
        for w in M3:
            o = res[w]; sp = float(np.nanmean(np.concatenate([z[f"{w}__spearman"] for z in Z]))); jc = float(np.mean(np.concatenate([z[f"{w}__jaccard"] for z in Z])))
            o.update({"spearman": sp, "jaccard": jc, "passes": o["diff"][1] > 0})
            P(f"  {SH[w]:<10}{o['mean']:>9.3f}{o['null']:>8.3f}{fmt_ci(*o['diff']):>30}{o['counts']:>8}{sp:>10.2f}{jc:>9.2f}   {'yes' if o['passes'] else 'no'}")
        rep[name] = res
    save("T3", rep)


# ------------------------------------------------------------------------------------------------ T4
def T4():
    from .tier5_N1 import new_rows
    rep = {}
    P("T4 -- SECONDS PER MOLECULE, old methods (GPU 1, 8 concurrent jobs, one job per unit and method; 100 molecules per unit; one untimed warm-up)")
    P("  PG train = one training run on the unit's stored PGExplainer training molecules (total seconds, and per training molecule)")
    P("  load check: EiG and GNNShap re-timed in the same run vs their Tier 5 N1 times on the same molecules")
    P(f"\n  {'unit':<26}{'GNN':>7}{'PG train':>10}{'/mol':>7}{'PG inf':>8}{'Sal':>7}{'IG':>7}{'IxG':>7}   {'EiG now/N1':>12}{'GNNShap now/N1':>16}")
    for u in T4_UNITS:
        J = {w: json.load(open(f"{T6}/T4/time/{u}__{w}.json")) for w in ("gnnexplainer", "pgexplainer", "grad", "eig", "gnnshap")}
        mols = set(subset100(u)); n1 = {}
        for w in ("eig", "gnnshap"):
            r = new_rows(u, w); n1[w] = float(np.mean([v["seconds"] for m, v in r.items() if m in mols]))
        o = {"gnnexplainer": float(np.mean(J["gnnexplainer"]["seconds"])), "pg_train_total": J["pgexplainer"]["train_seconds"],
             "pg_train_molecules": J["pgexplainer"]["train_molecules"], "pg_train_per_mol": J["pgexplainer"]["train_seconds"] / J["pgexplainer"]["train_molecules"],
             "pg_inference": float(np.mean(J["pgexplainer"]["seconds"])), **{w: float(np.mean(J["grad"][f"seconds_{w}"])) for w in ("sal", "ig", "ixg")},
             "eig_now": float(np.mean(J["eig"]["seconds"])), "eig_n1": n1["eig"], "gnnshap_now": float(np.mean(J["gnnshap"]["seconds"])), "gnnshap_n1": n1["gnnshap"]}
        rep[u] = o
        P(f"  {u:<26}{o['gnnexplainer']:>7.2f}{o['pg_train_total']:>10.1f}{o['pg_train_per_mol']:>7.2f}{o['pg_inference']:>8.4f}{o['sal']:>7.4f}{o['ig']:>7.4f}{o['ixg']:>7.4f}"
          f"   {o['eig_now']:>5.2f}/{o['eig_n1']:<6.2f}{o['gnnshap_now']:>8.2f}/{o['gnnshap_n1']:<6.2f}")
    r = [rep[u]["eig_now"] / rep[u]["eig_n1"] for u in T4_UNITS] + [rep[u]["gnnshap_now"] / rep[u]["gnnshap_n1"] for u in T4_UNITS]
    P(f"\n  load check: now / N1 time ratio for EiG and GNNShap ranges {min(r):.2f}-{max(r):.2f} (median {np.median(r):.2f})")
    save("T4", rep)


if __name__ == "__main__":
    {"T1": T1, "T2": T2, "T3": T3, "T4": T4}[sys.argv[1]]()
