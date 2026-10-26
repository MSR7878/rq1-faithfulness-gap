"""
RQ1 revision analyses T1, T5, T2, T3 (runs/rq1_revision/PLAN.md). Reads the
re-scoring pass (runs/rq1_revision/pass/{G,E}) and stored arrays; writes
runs/rq1_revision/<item>.txt and .json.

    python -m src.analysis.revision_analysis t1 | t5 | t2 | t3
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, kendalltau, ttest_1samp, wilcoxon

from ..data.standard import TOX21_TASKS
from .mutagenicity_cv_report import rank_direction
from .mutagenicity_report import holm

R = "runs/rq1_revision"
PG, PE = f"{R}/pass/G", f"{R}/pass/E"
EXPL = ["gnnexplainer", "pgexplainer", "subgraphx"]
SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
RULES = ["R1", "R2", "R3"]
BX = ["indole", "PAINS", "X", "P"]
ARCHS = ["dmpnn", "gin", "gine", "gcn"]
LOG = []


def P(s=""):
    print(s); LOG.append(s)


def save(name, rep):
    open(f"{R}/{name}.txt", "w", encoding="utf-8").write("\n".join(LOG) + "\n")
    json.dump(rep, open(f"{R}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def wp(a, b=None):
    a = np.asarray(a, float); b = np.zeros_like(a) if b is None else np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 6 or np.allclose(a, b):
        return 1.0, 0
    return float(wilcoxon(a, b, zero_method="wilcox").pvalue), rank_direction(a, b)


def fam_signs(raw):
    """raw: key -> (p, direction). Holm over the keys -> key -> (+ / - / .)."""
    adj = holm({k: v[0] for k, v in raw.items()})
    return {k: ("+" if raw[k][1] > 0 and adj[k] < .05 else "-" if raw[k][1] < 0 and adj[k] < .05 else ".") for k in raw}


def cnt(sg):
    return f"{sg.count('+')}/{sg.count('-')}/{sg.count('.')}"


def gload(u):
    return dict(np.load(f"{PG}/{u}.npz"))


def eload(u):
    return dict(np.load(f"{PE}/{u}.npz"))


T1_SETS = {"Mutagenicity D-MPNN": [f"mutagenicity_dmpnn_s{s}" for s in range(5)],
           **{f"B-XAIC {t}": [f"bxaic_{t}_s{s}" for s in range(5)] for t in BX},
           "MUTAG-GT (10 seeds)": [f"mutag_graphxai_s{s}" for s in range(10)]}
MODEL_SETS = {**T1_SETS, **{f"Mutagenicity {a.upper()}": [f"mutagenicity_{a}_s{s}" for s in range(5)] for a in ("gin", "gine", "gcn")}}
# "better" direction for the GT mask vs the null: Fid+ higher, Fid- lower, GEF lower
BETTER = {"fidp": +1, "fidm": -1, "gef": -1}


def beat_sign(sign, metric):
    """translate the rank-test sign of (GT - null) into beats (+) / loses (-) / n.s. (.)"""
    if sign == ".":
        return "."
    return "+" if (sign == "+") == (BETTER[metric] > 0) else "-"


# ============================================================================ T1
def t1():
    rep = {}
    P("T1 -- ORACLE TEST: the ground-truth mask vs size-matched random masks under R1, R2, R3")
    P("  '+' = GT mask better than the null (Fid+ higher, Fid- lower, GEF lower), '-' = worse, '.' = n.s.")
    P("  Holm family per (dataset, seed): probability space 3 metrics x 3 rules x 2 nulls = 18 tests; logit space Fid+/Fid- = 12 tests.")
    for space, metrics in (("prob", ["fidp", "fidm", "gef"]), ("logit", ["fidp", "fidm"])):
        P(f"\n  [{space} space] counts better/worse/n.s. over seeds, and mean gap GT - null (mean over seeds)")
        P(f"  {'dataset':<22}{'metric':<6}{'null':<6}" + "".join(f"{r:>22}" for r in RULES))
        for ds, units in T1_SETS.items():
            signs = {}; gaps = {}
            for u in units:
                g = gload(u); raw = {}
                for m in metrics:
                    key = f"{m}_{space}" if m != "gef" else "gef"
                    for r in RULES:
                        for nl in ("node", "conn"):
                            raw[(m, r, nl)] = wp(g[f"gt_{r}_{key}"], g[f"{nl}_{r}_{key}"])
                            gaps.setdefault((m, r, nl), []).append(float(np.nanmean(g[f"gt_{r}_{key}"] - g[f"{nl}_{r}_{key}"])))
                for k, sgn in fam_signs(raw).items():
                    signs.setdefault(k, []).append(beat_sign(sgn, k[0]))
            for m in metrics:
                for nl in ("node", "conn"):
                    P(f"  {ds:<22}{m:<6}{nl:<6}" + "".join(f"{cnt(signs[(m, r, nl)]):>12}{np.mean(gaps[(m, r, nl)]):>+10.3f}" for r in RULES))
            rep[f"{ds}|{space}"] = {f"{m}|{r}|{nl}": {"signs": signs[(m, r, nl)], "mean_gap": float(np.mean(gaps[(m, r, nl)]))}
                                    for m in metrics for r in RULES for nl in ("node", "conn")}
    # predictions
    P("\n  P-T1a: GT Fid+ (probability) significantly above random-connected in >= 4/5 seeds under R1 AND R2, every dataset")
    ok_all = True; tot = {"R2": 0, "R3": 0}
    for ds in [d for d in T1_SETS if not d.startswith("MUTAG-GT")]:
        s = rep[f"{ds}|prob"]
        c = {r: s[f"fidp|{r}|conn"]["signs"].count("+") for r in RULES}
        held = c["R1"] >= 4 and c["R2"] >= 4; ok_all &= held; tot["R2"] += c["R2"]; tot["R3"] += c["R3"]
        P(f"    {ds:<22} R1 {c['R1']}/5  R2 {c['R2']}/5  R3 {c['R3']}/5 -> {'meets' if held else 'does not meet'}")
    P(f"  P-T1a {'HELD' if ok_all else 'DID NOT HOLD'}")
    P(f"  P-T1b: seeds with GT Fid+ significantly above random-connected, summed over the 5 datasets: R2 {tot['R2']}/25, R3 {tot['R3']}/25"
      f" -> {'HELD' if tot['R3'] < tot['R2'] else 'DID NOT HOLD'}")
    rep["P-T1a"] = ok_all; rep["P-T1b"] = {"R2": tot["R2"], "R3": tot["R3"], "held": tot["R3"] < tot["R2"]}

    # R3 reliance for every model
    P("\n  R3 RELIANCE (group removal vs random sets, logit margin drop = Fid+ logit). Holm family per (model, seed): 3 rules x 2 nulls = 6.")
    P(f"  {'model':<26}" + "".join(f"{r + ' ' + nl:>14}" for r in RULES for nl in ("conn", "node")))
    rep["reliance"] = {}
    for ds, units in MODEL_SETS.items():
        signs = {}
        for u in units:
            g = gload(u)
            raw = {(r, nl): wp(g[f"gt_{r}_fidp_logit"], g[f"{nl}_{r}_fidp_logit"]) for r in RULES for nl in ("conn", "node")}
            for k, sgn in fam_signs(raw).items():
                signs.setdefault(k, []).append(sgn)
        P(f"  {ds:<26}" + "".join(f"{cnt(signs[(r, nl)]):>14}" for r in RULES for nl in ("conn", "node")))
        rep["reliance"][ds] = {f"{r}|{nl}": signs[(r, nl)] for r in RULES for nl in ("conn", "node")}

    # inflation check (P-T1c)
    P("\n  INFLATION CHECK (P-T1c): d = (explainer R2 - R3) - (random-node R2 - R3), probability space; random-node at the explainer's top-k size.")
    P("  Holm family per (dataset, seed) = 3 explainers x 2 metrics (Fid+, GEF). '+' = explainer inflates more than random.")
    larger = 0; cells = 0; rep["inflation"] = {}
    P(f"  {'dataset':<22}{'explainer':<6}{'metric':<6}{'mean expl R2-R3':>16}{'mean rand R2-R3':>16}{'counts':>10}   cell")
    for ds, units in T1_SETS.items():
        signs, ex, ra = {}, {}, {}
        for u in units:
            e = eload(u); raw = {}
            for x in EXPL:
                for m, key in (("Fid+", "fidp_prob"), ("GEF", "gef")):
                    de = e[f"{x}_R2_{key}"] - e[f"{x}_R3_{key}"]; dr = e[f"randnode_R2_{key}"] - e[f"randnode_R3_{key}"]
                    raw[(x, m)] = wp(de - dr); ex.setdefault((x, m), []).append(float(de.mean())); ra.setdefault((x, m), []).append(float(dr.mean()))
            for k, sgn in fam_signs(raw).items():
                signs.setdefault(k, []).append(sgn)
        need = 6 if len(units) == 10 else 3
        for x in EXPL:
            for m in ("Fid+", "GEF"):
                lg = signs[(x, m)].count("+") >= need; larger += lg; cells += 1
                P(f"  {ds:<22}{SH[x]:<6}{m:<6}{np.mean(ex[(x, m)]):>16.3f}{np.mean(ra[(x, m)]):>16.3f}{cnt(signs[(x, m)]):>10}   {'LARGER' if lg else ''}")
                rep["inflation"][f"{ds}|{x}|{m}"] = {"signs": signs[(x, m)], "expl": float(np.mean(ex[(x, m)])), "rand": float(np.mean(ra[(x, m)])), "larger": bool(lg)}
    P(f"  P-T1c: cells where the explainer's inflation is larger than random: {larger}/{cells} -> {'HELD' if larger <= cells // 2 - (1 if cells % 2 == 0 else 0) else 'DID NOT HOLD'}"
      f" (held if <= {cells // 2 - 1 if cells % 2 == 0 else cells // 2})")
    rep["P-T1c"] = {"larger": larger, "cells": cells, "held": larger <= 17}
    save("t1", rep)


# ============================================================================ T5
def t5():
    rep = {}
    P("T5 -- SUFFICIENCY: keep only S, hide the rest; m(kept S) vs mean m(kept random connected set of |S|) (logit margin)")
    P("  Equivalent to comparing Fid- (m(G) - m(kept)), since m(G) cancels. '+' = keeping S preserves the prediction better.")
    P("  Holm family per (model, seed) = R1, R2, R3 (3 tests). Primary rule R1.")
    P(f"  {'model':<26}" + "".join(f"{r:>12}" for r in RULES) + "   mean m(kept S) - m(kept conn) per rule")
    for ds, units in MODEL_SETS.items():
        signs, gaps = {}, {}
        for u in units:
            g = gload(u)
            raw = {r: wp(g[f"conn_{r}_fidm_logit"], g[f"gt_{r}_fidm_logit"]) for r in RULES}   # m(S) - m(conn) = Fid-(conn) - Fid-(S)
            for r, sgn in fam_signs(raw).items():
                signs.setdefault(r, []).append(sgn)
                gaps.setdefault(r, []).append(float(np.nanmean(g[f"conn_{r}_fidm_logit"] - g[f"gt_{r}_fidm_logit"])))
        P(f"  {ds:<26}" + "".join(f"{cnt(signs[r]):>12}" for r in RULES) + "   " + "  ".join(f"{np.mean(gaps[r]):+.3f}" for r in RULES))
        rep[ds] = {r: {"signs": signs[r], "gap": float(np.mean(gaps[r]))} for r in RULES}
    ok = all(rep[d]["R1"]["signs"].count("+") >= 4 for d in ["Mutagenicity D-MPNN"] + [f"B-XAIC {t}" for t in BX]) and \
        rep["MUTAG-GT (10 seeds)"]["R1"]["signs"].count("+") >= 8
    P(f"\n  P-T5 (R1; >= 4/5 Mutagenicity D-MPNN and each B-XAIC task, >= 8/10 MUTAG-GT): {'HELD' if ok else 'DID NOT HOLD'}")
    rep["P-T5"] = ok
    # comparison with the earlier sufficiency number (Part A, R1, uncorrected test, one test per seed)
    P("\n  Earlier 'sufficiency' (Part A, f53b612): R1 only, m(kept S) vs mean m(kept random connected), one uncorrected test per seed;")
    P("  it was 5/0/0 for every model except MUTAG-GT seeds 0-4 (5/0/0 too). Same quantity as the R1 column here; differences come")
    P("  only from the random draws (new seeded stream) and Holm over 3 rules.")
    save("t5", rep)


# ============================================================================ T2
def unit_groups():
    g = {"MUTAG": ["mutag"], "BBBP": ["bbbp"], "MUTAG-GT": [f"mutag_graphxai_s{s}" for s in range(10)],
         "Tox21": [f"tox21_{ep}_s{s}" for ep in TOX21_TASKS for s in range(5)]}
    g.update({f"B-XAIC {t}": [f"bxaic_{t}_s{s}" for s in range(5)] for t in BX})
    g.update({f"Mutagenicity {a.upper() if a != 'dmpnn' else 'D-MPNN'}": [f"mutagenicity_{a}_s{s}" for s in range(5)] for a in ARCHS})
    return g


def gea_arrays(u):
    """top-k (random ties) and mean-threshold GEA per explainer for GT units; None otherwise."""
    if u.startswith("mutag_graphxai"):
        s = int(u.rsplit("_s", 1)[1])
        tk = np.load(f"runs/rq1_followups/phase2b/gea/audit_mutag_graphxai_s{s}.npz" if s < 5 else f"runs/rq1_final_export/gea/audit_mutag_graphxai_s{s}.npz")
        mt = np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/mutag_graphxai_s{s}.npz" if s < 5 else f"runs/rq1_followups/phase1/null_audit_arrays/mutag_graphxai_s{s}.npz")
        return {e: tk[f"{e}_b"] for e in EXPL}, {e: mt[f"{e}_mean"] for e in EXPL}
    if u.startswith("bxaic_"):
        t, s = u[6:].rsplit("_s", 1)
        tk = np.load(f"runs/rq1_followups/phase2b/gea/audit_bxaic_{t}_s{s}.npz"); mt = np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/bxaic_{t}_s{s}.npz")
        return {e: tk[f"{e}_b"] for e in EXPL}, {e: mt[f"{e}_mean"] for e in EXPL}
    if u.startswith("mutagenicity_"):
        a, s = u[13:].rsplit("_s", 1); A = {"dmpnn": "D-MPNN", "gin": "GIN", "gine": "GINE", "gcn": "GCN"}[a]
        d = {"D-MPNN": "runs/mutagenicity", "GIN": "runs/arch_gin/mutagenicity_graphxai", "GINE": "runs/arch_gine/mutagenicity_graphxai",
             "GCN": "runs/arch_gcn/mutagenicity_graphxai"}[A]
        tk = np.load(f"runs/rq1_followups/phase2b/gea/arch_{A}_s{s}.npz"); mt = np.load(f"{d}/arrays_s{s}.npz")
        return {e: tk[f"{e}_b"] for e in EXPL}, {e: mt[f"{e}_mean_set"] for e in EXPL}
    return None


def rank_of(vals, higher_better):
    """explainer order (list of names, best first) by mean value."""
    return sorted(EXPL, key=lambda e: -vals[e] if higher_better else vals[e])


def tau(r1, r2):
    pos = {e: i for i, e in enumerate(r1)}
    return float(kendalltau([pos[e] for e in EXPL], [r2.index(e) for e in EXPL]).statistic)


def t2():
    rep = {}
    P("T2 -- RANKINGS OF THE 3 EXPLAINERS UNDER EACH RULE AND CUT (probability space; random ties)")
    P("  Fid+ higher = better; Fid- and GEF lower = better; GEA higher = better. Kendall tau with 3 items is -1, -1/3, 1/3 or 1.")
    MET = [("Fid+", "fidp_prob", True), ("Fid-", "fidm_prob", False), ("GEF", "gef", False)]
    all_taus = []; per_ds = {}
    for ds, units in unit_groups().items():
        taus_rule = []; pair_taus = {}; flips = 0; flip_cells = 0
        for u in units:
            e = eload(u); ranks, mol = {}, {}
            for lab, key, hb in MET:
                for r in RULES:
                    v = {x: e[f"{x}_{r}_{key}"] for x in EXPL}; mol[(lab, r)] = (v, hb)
                    ranks[(lab, r)] = rank_of({x: float(v[x].mean()) for x in EXPL}, hb)
            ga = gea_arrays(u)
            if ga is not None:
                ranks[("GEA", "top-k")] = rank_of({x: float(ga[0][x].mean()) for x in EXPL}, True)
                ranks[("GEA", "mean-thr")] = rank_of({x: float(ga[1][x].mean()) for x in EXPL}, True)
            for lab, _, _ in MET:
                for r1, r2 in itertools.combinations(RULES, 2):
                    t = tau(ranks[(lab, r1)], ranks[(lab, r2)]); taus_rule.append(t); all_taus.append(t)
            for k1, k2 in itertools.combinations(sorted(ranks), 2):
                pair_taus.setdefault((k1, k2), []).append(tau(ranks[k1], ranks[k2]))
            # significant flips: pairwise Wilcoxon, Holm over the 3 pairs within (metric, rule)
            sig = {}
            for lab, key, hb in MET:
                for r in RULES:
                    v, _ = mol[(lab, r)]
                    raw = {(a, b): wp(v[a], v[b]) for a, b in itertools.combinations(EXPL, 2)}
                    sg = fam_signs(raw)
                    for pr, s_ in sg.items():
                        sig[(lab, r, pr)] = s_
            for lab, _, _ in MET:
                for pr in itertools.combinations(EXPL, 2):
                    for r1, r2 in itertools.combinations(RULES, 2):
                        flip_cells += 1
                        a, b = sig[(lab, r1, pr)], sig[(lab, r2, pr)]
                        flips += (a != "." and b != "." and a != b)
        med = float(np.median(taus_rule)); per_ds[ds] = med
        P(f"\n  -- {ds} ({len(units)} units): same metric across rules, median tau {med:+.2f} (range {min(taus_rule):+.2f} to {max(taus_rule):+.2f});"
          f" significant order flips between rules: {flips} of {flip_cells} (pair x metric x rule-pair x unit)")
        for (k1, k2), ts in sorted(pair_taus.items()):
            P(f"     tau {k1[0]}-{k1[1]:<9} vs {k2[0]}-{k2[1]:<9} median {np.median(ts):+.2f}  range {min(ts):+.2f}..{max(ts):+.2f}")
        rep[ds] = {"median_tau_same_metric": med, "flips": flips, "flip_cells": flip_cells,
                   "pair_taus": {f"{k1}|{k2}": [float(np.median(ts)), float(min(ts)), float(max(ts))] for (k1, k2), ts in pair_taus.items()}}
    m = float(np.median(all_taus))
    P(f"\n  P-T2: median Kendall tau over all (unit, metric, rule pair) = {m:+.2f} -> {'HELD' if m <= 1 / 3 + 1e-9 else 'DID NOT HOLD'} (held if <= +1/3)")
    rep["P-T2"] = {"median_tau": m, "held": m <= 1 / 3 + 1e-9, "per_dataset": per_ds}
    # mean-threshold vs top-k flip on Mutagenicity (D-MPNN)
    P("\n  Mutagenicity D-MPNN: SubgraphX vs GNNExplainer GEA per seed under each cut; Holm family per seed = the 2 cuts")
    rep["mt_vs_topk"] = {}
    for s in range(5):
        tk, mt = gea_arrays(f"mutagenicity_dmpnn_s{s}")
        raw = {"top-k": wp(tk["subgraphx"], tk["gnnexplainer"]), "mean-thr": wp(mt["subgraphx"], mt["gnnexplainer"])}
        sg = fam_signs(raw); flip = sg["top-k"] != "." and sg["mean-thr"] != "." and sg["top-k"] != sg["mean-thr"]
        P(f"    seed {s}: top-k SX {tk['subgraphx'].mean():.3f} vs GNN {tk['gnnexplainer'].mean():.3f} [{sg['top-k']}]   "
          f"mean-thr SX {mt['subgraphx'].mean():.3f} vs GNN {mt['gnnexplainer'].mean():.3f} [{sg['mean-thr']}]   significant flip: {flip}")
        rep["mt_vs_topk"][s] = {"topk": sg["top-k"], "meanthr": sg["mean-thr"], "flip": flip}
    save("t2", rep)


# ============================================================================ T3
B_REPS, RNG_SEED = 2000, 2026
CEIL = {"MUTAG-GT": 0.751, "indole": 0.724, "Mutagenicity": 0.551}


def hboot(real, null, ceiling=None, reps=B_REPS, seed=RNG_SEED):
    """real, null: list over seeds of per-molecule arrays. Returns (point, lo, hi) of the mean over seeds of the per-seed mean
    difference; with ceiling: normalised (mean real - mean null)/(ceiling - mean null), ratio of means within each replicate."""
    rng = np.random.default_rng(seed); S = len(real)

    def stat(idx_s, idx_m):
        if ceiling is None:
            return float(np.mean([np.mean(real[s][m] - null[s][m]) for s, m in zip(idx_s, idx_m)]))
        a = np.mean([np.mean(real[s][m]) for s, m in zip(idx_s, idx_m)]); b = np.mean([np.mean(null[s][m]) for s, m in zip(idx_s, idx_m)])
        return float((a - b) / (ceiling - b))
    point = stat(range(S), [np.arange(len(real[s])) for s in range(S)])
    vals = []
    for _ in range(reps):
        ss = rng.integers(0, S, S)
        vals.append(stat(ss, [rng.integers(0, len(real[s]), len(real[s])) for s in ss]))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), np.array(vals)


def t3():
    rep = {}
    P("T3 -- EFFECT SIZES, hierarchical bootstrap (seeds, then molecules within seed; 2,000 reps; seed 2026; 95% percentile CI)")
    P("  Estimand: mean over seeds of the per-seed mean difference (equal seed weight). Normalised GEA: ratio of means per replicate.")
    cells = {}
    def mutag(seeds):
        out = {}
        for s in seeds:
            o, n, = None, None
            tk = np.load(f"runs/rq1_followups/phase2b/gea/audit_mutag_graphxai_s{s}.npz" if s < 5 else f"runs/rq1_final_export/gea/audit_mutag_graphxai_s{s}.npz")
            st = np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/mutag_graphxai_s{s}.npz" if s < 5 else f"runs/rq1_followups/phase1/null_audit_arrays/mutag_graphxai_s{s}.npz")
            out[s] = ({e: tk[f"{e}_b"] for e in EXPL}, st["rn_k"])
        return out
    cells[("MUTAG-GT seeds 0-4", "MUTAG-GT")] = mutag(range(5))
    cells[("MUTAG-GT seeds 0-9", "MUTAG-GT")] = mutag(range(10))
    cells[("B-XAIC indole", "indole")] = {s: ({e: np.load(f"runs/rq1_followups/phase2b/gea/audit_bxaic_indole_s{s}.npz")[f"{e}_b"] for e in EXPL},
                                              np.load(f"runs/mutagenicity/exploratory/null_audit_arrays/bxaic_indole_s{s}.npz")["rn_k"]) for s in range(5)}
    for A, d in (("D-MPNN", "runs/mutagenicity"), ("GIN", "runs/arch_gin/mutagenicity_graphxai"), ("GINE", "runs/arch_gine/mutagenicity_graphxai"),
                 ("GCN", "runs/arch_gcn/mutagenicity_graphxai")):
        cells[(f"Mutagenicity {A}", "Mutagenicity")] = {s: ({e: np.load(f"runs/rq1_followups/phase2b/gea/arch_{A}_s{s}.npz")[f"{e}_b"] for e in EXPL},
                                                           np.load(f"{d}/arrays_s{s}.npz")["rand_node_topk_set"]) for s in range(5)}
    P(f"\n  top-k GEA minus random-node GEA, and normalised GEA (ceilings MUTAG-GT 0.751, indole 0.724, Mutagenicity 0.551)")
    P(f"  {'cell':<26}{'explainer':<6}{'GEA - RandNode [95% CI]':>32}{'normalised GEA [95% CI]':>32}")
    rep["gea"] = {}
    for (name, ck), per in cells.items():
        seeds = sorted(per)
        for e in EXPL:
            real = [per[s][0][e] for s in seeds]; null = [per[s][1] for s in seeds]
            d = hboot(real, null); nm = hboot(real, null, CEIL[ck])
            P(f"  {name:<26}{SH[e]:<6}{d[0]:>+10.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{'':>3}{nm[0]:>+10.3f} [{nm[1]:+.3f}, {nm[2]:+.3f}]")
            rep["gea"][f"{name}|{e}"] = {"diff": d[:3], "norm": nm[:3]}
    # SX vs RandConn (mean-threshold)
    P("\n  SubgraphX mean-threshold GEA minus size-matched random-connected GEA")
    conn = {"MUTAG-GT seeds 0-9": [(np.load("runs/rq1_followups/phase1/connectivity_arrays_mutag_graphxai.npz")[f"sx_s{s}"],
                                    np.load("runs/rq1_followups/phase1/connectivity_arrays_mutag_graphxai.npz")[f"rand_conn_s{s}"]) for s in range(10)],
            "B-XAIC indole": [(np.load("runs/prio/connectivity_arrays_indole.npz")[f"sx_s{s}"], np.load("runs/prio/connectivity_arrays_indole.npz")[f"rand_conn_s{s}"]) for s in range(5)]}
    for A, d in (("D-MPNN", "runs/mutagenicity"), ("GIN", "runs/arch_gin/mutagenicity_graphxai"), ("GINE", "runs/arch_gine/mutagenicity_graphxai"),
                 ("GCN", "runs/arch_gcn/mutagenicity_graphxai")):
        conn[f"Mutagenicity {A}"] = [(np.load(f"{d}/arrays_s{s}.npz")["subgraphx_mean_set"], np.load(f"{d}/arrays_s{s}.npz")["rand_conn_mean_subgraphx_set"]) for s in range(5)]
    conn["MUTAG-GT seeds 0-4"] = conn["MUTAG-GT seeds 0-9"][:5]
    rep["sx_conn"] = {}
    for name, pairs in conn.items():
        d = hboot([a for a, _ in pairs], [b for _, b in pairs])
        P(f"  {name:<26}{d[0]:>+10.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        rep["sx_conn"][name] = d[:3]
    # GCN SubgraphX check
    P("\n  GCN SubgraphX, mean-threshold vs size-matched random-connected, per seed")
    P(f"  {'seed':>4}{'n':>5}{'SX':>7}{'conn':>7}{'zero diff':>10}{'tied |d|':>9}{'Wilcoxon p':>12}{'Holm p (4)':>11}{'sign test p':>12}  bootstrap 95% CI of mean diff")
    from .mutagenicity_cv_report import FAMILY
    rep["gcn_sx"] = {}
    for s in range(5):
        z = dict(np.load(f"runs/arch_gcn/mutagenicity_graphxai/arrays_s{s}.npz")); a, b = z["subgraphx_mean_set"], z["rand_conn_mean_subgraphx_set"]
        dd = a - b; nz = dd[dd != 0]
        raw = {lab: float(wilcoxon(z[r], z[nl], zero_method="wilcox").pvalue) for lab, r, nl in FAMILY}
        adj = holm(raw); pw = raw["SX vs RandConn"]
        sp = float(binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0
        _, ties = np.unique(np.round(np.abs(nz), 12), return_counts=True)
        rng = np.random.default_rng(RNG_SEED); bs = [np.mean(dd[rng.integers(0, len(dd), len(dd))]) for _ in range(B_REPS)]
        P(f"  {s:>4}{len(dd):>5}{a.mean():>7.3f}{b.mean():>7.3f}{np.mean(dd == 0):>10.2f}{int((ties > 1).sum()):>9}{pw:>12.2e}{adj['SX vs RandConn']:>11.2e}{sp:>12.2e}"
          f"  [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
        rep["gcn_sx"][s] = {"n": len(dd), "zero_frac": float(np.mean(dd == 0)), "wilcoxon": pw, "holm": adj["SX vs RandConn"], "sign": sp,
                            "boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "median_diff": float(np.median(dd))}
    # Tox21 TOST
    P("\n  Tox21 equivalence (TOST, margin 0.05): probability-space random-node-corrected Fid+ / Fid- under R3 (random ties).")
    P("  'No-separation' cell = n.s. in >= 3/5 seeds in the stored Part B test (logit space, random-node, Holm over 18 within unit).")
    from .followups_phase2b import merged_B
    from .reliance_fidelity_report import b_signs
    eq = und = diff = 0; rep["tost"] = {}
    P(f"  {'endpoint':<16}{'expl':<5}{'metric':<6}{'mean diff':>10}{'90% CI':>20}{'95% CI':>20}  verdict   per-unit TOST")
    for ep in TOX21_TASKS:
        units = [f"tox21_{ep}_s{s}" for s in range(5)]
        Bs = {u: merged_B(u) for u in units}
        for x in EXPL:
            for met, lab in (("fid_plus", "Fid+"), ("fid_minus", "Fid-")):
                st = [b_signs(Bs[u][0])["node"][(x, "R3", met)] for u in units]
                if st.count(".") < 3:
                    continue
                real = [Bs[u][1][f"{x}_R3_{met}_prob"] for u in units]; null = [Bs[u][1][f"rand_node_R3_{met}_prob"] for u in units]
                pt, _, _, vals = hboot(real, null)
                c90 = (np.percentile(vals, 5), np.percentile(vals, 95)); c95 = (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
                v = "equivalent" if (c90[0] > -0.05 and c90[1] < 0.05) else ("different" if (c95[0] > 0 or c95[1] < 0) else "undecided")
                eq += v == "equivalent"; und += v == "undecided"; diff += v == "different"
                tost = 0
                for r_, n_ in zip(real, null):
                    dd = r_ - n_
                    p1 = ttest_1samp(dd, -0.05, alternative="greater").pvalue; p2 = ttest_1samp(dd, 0.05, alternative="less").pvalue
                    tost += max(p1, p2) < 0.05
                P(f"  {ep:<16}{SH[x]:<5}{lab:<6}{pt:>+10.3f}{f'[{c90[0]:+.3f}, {c90[1]:+.3f}]':>20}{f'[{c95[0]:+.3f}, {c95[1]:+.3f}]':>20}  {v:<10}{tost}/5")
                rep["tost"][f"{ep}|{x}|{lab}"] = {"diff": pt, "ci90": list(c90), "ci95": list(c95), "verdict": v, "unit_tost": tost}
    P(f"  no-separation cells: {eq + und + diff}; equivalent {eq}, undecided {und}, different {diff}")
    save("t3", rep)


if __name__ == "__main__":
    {"t1": t1, "t5": t5, "t2": t2, "t3": t3}[sys.argv[1]]()
