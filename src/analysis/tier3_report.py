"""
RQ1 Tier 3 reports: writes runs/rq1_tier3/<name>.txt/.json.

    python -m src.analysis.tier3_report E|F|X
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy.stats import kendalltau

from . import revision_analysis as RA
from . import tier2_common as C
from . import tier2_report as T2R
from .tier2_report import fmt_ci, load
from .tier3_F import METHODS

P = T2R.P
T = "runs/rq1_tier3"
T2 = C.T2
SH = {**T2R.SH, "sx_fixed": "SX-fixed"}
VAR = ("all", "first", "last")
MUT = {g: us for g, us in C.GROUPS.items() if g.startswith("Mutagenicity")}


def save(name, rep):
    open(f"{T}/{name}.txt", "w", encoding="utf-8").write("\n".join(T2R.LOG) + "\n")
    json.dump(rep, open(f"{T}/{name}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ------------------------------------------------------------------ E
def E():
    rep = {}
    P("E -- EDGE-MASK INJECTION ABLATION (EXPLORATORY, no prediction; P-E1 and P-E2 withdrawn, wrong premise; see PLAN.md)")
    P("  GNNExplainer's atom scores come from its node mask, trained jointly with the edge mask, so it may barely respond to the variants.")
    P("  PGExplainer's atom scores come from the edge mask. Stored PG = final rule; variant PG = float64 logit-based scores.")
    mp = json.load(open(f"{T}/E/mask_path.json"))
    P(f"\n  [0] mask path on one graph per model; all checks pass: {mp['all_checks_pass']}")
    P(f"  {'model':<8}{'MP modules':>11}{'mask calls':>11}{'all vs stock PyG':>18}" + "".join(f"{'ones vs zeros ' + v:>21}" for v in VAR) + f"{'all vs first (frac)':>21}{'all vs last (frac)':>20}")
    for a in ("dmpnn", "gin", "gine", "gcn"):
        x = mp[a]
        P(f"  {a:<8}{x['mp_modules']:>11}{x['all']['mask_calls_per_forward']:>11}{x['frac_all_vs_stock_pyg']:>18.1e}" + "".join(f"{x[v]['max_abs_diff_ones_vs_zeros']:>21.3f}" for v in VAR) + f"{x['frac_all_vs_first']:>21.3f}{x['frac_all_vs_last']:>20.3f}")
    Z = {u: load(f"{T}/E/gea/{u}.npz") for us in MUT.values() for u in us}
    P("\n  [1] GEA minus null. Holm per (model, seed): variants 2 x 3 x 2 = 12; stored 2 x 2 = 4")
    P(f"  {'model':<22}{'variant':<9}{'method':<10}{'minus random-node [95% CI]':>32}{'counts':>8}{'minus random-conn. [95% CI]':>32}{'counts':>8}{'GEA':>7}")
    rep["gea"] = {}
    rows = [("stored", m) for m in ("gnnexplainer", "pgexplainer")] + [(v, m) for v in VAR for m in ("gnnexplainer", "pgexplainer")]
    for g, us in MUT.items():
        sig = {}
        for u in us:
            z = Z[u]
            for fam in (("stored",), VAR):
                raw = {(v, m, r): C.wil(z[f"{v}_{m}"], z[r]) for v in fam for m in ("gnnexplainer", "pgexplainer") for r in ("rn", "rc")}
                for k, s in C.holm_signs(raw).items():
                    sig.setdefault(k, []).append(s)
        for v, m in rows:
            a = [Z[u][f"{v}_{m}"] for u in us]
            pn, ln, hn = C.boot(a, [Z[u]["rn"] for u in us]); pc, lc, hc = C.boot(a, [Z[u]["rc"] for u in us]); mean = float(np.mean([x.mean() for x in a]))
            rep["gea"][f"{g}|{v}|{m}"] = {"rn": [pn, ln, hn], "rc": [pc, lc, hc], "cat": C.ci_cat(ln, hn), "signs_rn": sig[(v, m, "rn")], "signs_rc": sig[(v, m, "rc")], "gea": mean}
            P(f"  {g:<22}{v:<9}{SH[m]:<10}{fmt_ci(pn, ln, hn):>32}{C.counts(sig[(v, m, 'rn')]):>8}{fmt_ci(pc, lc, hc):>32}{C.counts(sig[(v, m, 'rc')]):>8}{mean:>7.3f}")
    P("\n  [2] how much each explainer responds: variant minus the 'all' re-run (95% CI); mean Spearman of atom scores with 'all'; 'all' re-run vs stored")
    P(f"  {'model':<22}{'method':<10}{'first - all':>28}{'rho':>7}{'last - all':>28}{'rho':>7}{'all - stored':>28}{'rho':>7}")
    rep["response"] = {}
    # rho is undefined when a score vector is constant; average over the seeds where it is defined anywhere
    def mrho(key, us):
        return float(np.mean([np.nanmean(Z[u][key]) for u in us if (~np.isnan(Z[u][key])).any()]))
    for g, us in MUT.items():
        for m in ("gnnexplainer", "pgexplainer"):
            al = [Z[u][f"all_{m}"] for u in us]; f = C.boot([Z[u][f"first_{m}"] for u in us], al); l = C.boot([Z[u][f"last_{m}"] for u in us], al)
            s0 = "stored_pg64" if m == "pgexplainer" else "stored_gnnexplainer"; st = C.boot(al, [Z[u][s0] for u in us])
            rep["response"][f"{g}|{m}"] = {"first": list(f), "last": list(l), "all_vs_stored": list(st), "rho_first": mrho(f"rho_first_{m}", us), "rho_last": mrho(f"rho_last_{m}", us), "rho_stored": mrho(f"rho_stored_{m}", us)}
            P(f"  {g:<22}{SH[m]:<10}{fmt_ci(*f):>28}{mrho(f'rho_first_{m}', us):>7.2f}{fmt_ci(*l):>28}{mrho(f'rho_last_{m}', us):>7.2f}{fmt_ci(*st):>28}{mrho(f'rho_stored_{m}', us):>7.2f}")
    P("      'all - stored' for PG compares with the float64 scores of the T4 re-run, so both sides use the same ranking rule.")
    P("      molecules with undefined rho (one side has constant atom scores; its top-k is then all random ties), per seed:")
    rep["undefined_rho"] = {}
    for g, us in MUT.items():
        for k in ("rho_first_pgexplainer", "rho_last_pgexplainer", "rho_stored_pgexplainer", "rho_first_gnnexplainer", "rho_last_gnnexplainer"):
            n = [int(np.isnan(Z[u][k]).sum()) for u in us]
            if any(n):
                rep["undefined_rho"][f"{g}|{k[4:]}"] = n
                P(f"        {g:<22}{k[4:]:<24}{' '.join(f'{x}/{len(Z[u][k])}' for x, u in zip(n, us))}")
    P("\n  [3] ordering by mean GEA (best first); SX fixed is not re-run")
    rep["order"] = {}
    for g, us in MUT.items():
        sx = float(np.mean([Z[u]["sx_fixed"].mean() for u in us])); cells = []
        for v in ("stored",) + VAR:
            vals = {"GNN": float(np.mean([Z[u][f"{v}_gnnexplainer"].mean() for u in us])), "PG": float(np.mean([Z[u][f"{v}_pgexplainer"].mean() for u in us])), "SX": sx}
            o = sorted(vals, key=lambda k: -vals[k]); rep["order"][f"{g}|{v}"] = {"order": o, "gea": vals}
            cells.append(f"{v} " + " > ".join(f"{k} {vals[k]:.3f}" for k in o))
        P(f"  {g}"); [P(f"    {c}") for c in cells]
    save("E", rep)


# ------------------------------------------------------------------ F
def _r2_means(u, rule, key):
    E_, S = load(f"runs/rq1_revision/pass/E/{u}.npz"), load(f"{T2}/S/fid/{u}.npz")
    return {"gnnexplainer": float(E_[f"gnnexplainer_{rule}_{key}"].mean()), "pgexplainer": float(E_[f"pgexplainer_{rule}_{key}"].mean()), "subgraphx": float(S[f"sx_fixed_{rule}_{key}"].mean())}


def _f_means(u, key):
    z = load(f"{T}/F/score/{u}.npz")
    return {"gnnexplainer": float(z[f"gnnexplainer_{key}"].mean()), "pgexplainer": float(z[f"pgexplainer_{key}"].mean()), "subgraphx": float(z[f"sx_fixed_{key}"].mean())}


def _gea_rank(u):
    H, S = load(f"{T2}/H/gea/{u}.npz"), load(f"{T2}/S/gea/{u}.npz")
    return RA.rank_of({"gnnexplainer": float(H["gnnexplainer"].mean()), "pgexplainer": float(H["pgexplainer"].mean()), "subgraphx": float(S["sx_fixed"].mean())}, True)


def F():
    rep = {}; G = C.GROUPS; Z = {u: load(f"{T}/F/score/{u}.npz") for us in G.values() for u in us}
    P("F -- FINE-TUNED COPY AND BOUNDED HIDING RULE F (see PLAN.md). Fid- under F is NOT comparable with R2 (F hides at most 50% of N).")
    P("\n  [0] clean test metrics, copy minus original; limit 0.02")
    P(f"  {'group':<22}{'acc before':>11}{'acc after':>11}{'max |d acc|':>12}{'AUROC before':>13}{'AUROC after':>12}{'max |d AUROC|':>14}{'train s':>9}")
    rep["copies"] = {}; bad = []
    for g, us in G.items():
        ft = [torch.load(f"{T}/F/ckpt/{u}.pt", map_location="cpu", weights_only=False)["finetune"] for u in us]
        da = [abs(x["test_acc_after"] - x["test_acc_before"]) for x in ft]; dr = [abs(x["test_auroc_after"] - x["test_auroc_before"]) for x in ft]
        for u, x, y in zip(us, da, dr):
            rep["copies"][u] = {"d_acc": x, "d_auroc": y}
            if x > 0.02 or y > 0.02:
                bad.append(f"{u}: |d acc| {x:.3f}, |d AUROC| {y:.3f}")
        P(f"  {g:<22}{np.mean([x['test_acc_before'] for x in ft]):>11.3f}{np.mean([x['test_acc_after'] for x in ft]):>11.3f}{max(da):>12.3f}{np.mean([x['test_auroc_before'] for x in ft]):>13.3f}{np.mean([x['test_auroc_after'] for x in ft]):>12.3f}{max(dr):>14.3f}{np.mean([x['seconds'] for x in ft]):>9.1f}")
    P(f"  copies outside the 0.02 limit: {len(bad)} of 50 (kept, per PLAN)"); [P(f"    {b}") for b in bad]; rep["outside_limit"] = bad
    P("\n  [1] molecules cut down by the 0.5 bound")
    P(f"  {'group':<22}{'explained':>10}{'top-k Fid+ cut':>15}{'top-k Fid- cut':>15}{'GT mols':>9}{'GT Fid+ cut':>13}{'GT Fid- cut':>13}")
    rep["cuts"] = {}
    for g, us in G.items():
        r = {"explained": sum(len(Z[u]["mol_idx"]) for u in us), "fidp": sum(int(Z[u]["cut_expl_fidp"]) for u in us), "fidm": sum(int(Z[u]["cut_expl_fidm"]) for u in us),
             "gt": sum(len(Z[u]["G_mol_idx"]) for u in us), "gt_fidp": sum(int(Z[u]["G_cut_fidp"].sum()) for u in us), "gt_fidm": sum(int(Z[u]["G_cut_fidm"].sum()) for u in us)}
        rep["cuts"][g] = r
        P(f"  {g:<22}{r['explained']:>10}{r['fidp']:>15}{r['fidm']:>15}{r['gt']:>9}{r['gt_fidp']:>13}{r['gt_fidm']:>13}")
    rep["methods"] = {}
    for space in ("prob", "logit"):
        P(f"\n  [2 {space}] methods under F minus null, 95% CI, counts better/worse/n.s.; Holm per (unit, space) = 7 x 2 x 2 = 28")
        P(f"  {'group':<22}{'method':<10}{'metric':<6}{'minus random-node':>30}{'counts':>9}{'minus random-conn.':>30}{'counts':>9}{'mean':>9}")
        for g, us in G.items():
            sig = {}
            for u in us:
                z = Z[u]; raw = {(m, mt, r): C.wil(z[f"{m}_{mt}_{space}"], z[f"{r}_{mt}_{space}"]) for m in METHODS for mt in ("fidp", "fidm") for r in ("rn", "rc")}
                for k, s in C.holm_signs(raw).items():
                    b = "." if s == "." else ("+" if (s == "+") == (k[1] == "fidp") else "-"); sig.setdefault(k, []).append(b)
            for m in METHODS:
                for mt in ("fidp", "fidm"):
                    a = [Z[u][f"{m}_{mt}_{space}"] for u in us]; n = C.boot(a, [Z[u][f"rn_{mt}_{space}"] for u in us]); c = C.boot(a, [Z[u][f"rc_{mt}_{space}"] for u in us])
                    rep["methods"][f"{g}|{m}|{mt}|{space}"] = {"rn": list(n), "rc": list(c), "signs_rn": sig[(m, mt, "rn")], "signs_rc": sig[(m, mt, "rc")]}
                    P(f"  {g:<22}{SH[m]:<10}{'Fid+' if mt == 'fidp' else 'Fid-':<6}{fmt_ci(*n):>30}{C.counts(sig[(m, mt, 'rn')]):>9}{fmt_ci(*c):>30}{C.counts(sig[(m, mt, 'rc')]):>9}{np.mean([x.mean() for x in a]):>9.3f}")
    # oracle
    rep["oracle"] = {}
    RUL = ["R1", "R2", "R3"]
    for space, mets in (("prob", ["fidp", "fidm", "gef"]), ("logit", ["fidp", "fidm"])):
        P(f"\n  [3 {space}] T1 repeat: ground truth minus size-matched null. counts better/worse/n.s. and mean gap. R1-R3 = original models (T1 family); F and R1c (R1 removal) = fine-tuned copy (family {12 if space == 'prob' else 8})")
        P(f"  {'group':<22}{'metric':<6}{'null':<6}" + "".join(f"{r:>18}" for r in RUL + ["F", "R1c"]))
        for g, us in G.items():
            sig = {}; gap = {}
            for u in us:
                o = load(f"runs/rq1_revision/pass/G/{u}.npz"); z = Z[u]; ko = lambda mt: f"{mt}_{space}" if mt != "gef" else "gef"
                raw = {(mt, r, nl): RA.wp(o[f"gt_{r}_{ko(mt)}"], o[f"{nl}_{r}_{ko(mt)}"]) for mt in mets for r in RUL for nl in ("node", "conn")}
                raw2 = {(mt, r, nl): RA.wp(z[f"G_gt_{r}_{ko(mt)}"], z[f"G_{nl}_{r}_{ko(mt)}"]) for mt in mets for r in ("F", "R1c") for nl in ("node", "conn")}
                for rw in (raw, raw2):
                    for k, s in RA.fam_signs(rw).items():
                        sig.setdefault(k, []).append(RA.beat_sign(s, k[0]))
                for mt in mets:
                    for nl in ("node", "conn"):
                        for r in RUL:
                            gap.setdefault((mt, r, nl), []).append(float(np.nanmean(o[f"gt_{r}_{ko(mt)}"] - o[f"{nl}_{r}_{ko(mt)}"])))
                        for r in ("F", "R1c"):
                            gap.setdefault((mt, r, nl), []).append(float(np.nanmean(z[f"G_gt_{r}_{ko(mt)}"] - z[f"G_{nl}_{r}_{ko(mt)}"])))
            for mt in mets:
                for nl in ("node", "conn"):
                    cells = "".join(f"{C.counts(sig[(mt, r, nl)]):>10}{np.mean(gap[(mt, r, nl)]):>+8.3f}" for r in RUL + ["F", "R1c"])
                    P(f"  {g:<22}{mt:<6}{nl:<6}{cells}")
                    for r in RUL + ["F", "R1c"]:
                        rep["oracle"][f"{g}|{space}|{mt}|{r}|{nl}"] = {"signs": sig[(mt, r, nl)], "gap": float(np.mean(gap[(mt, r, nl)]))}
    P("\n  P-F1: ground truth beats random-connected on Fid+ (prob) under F in >= 4/5 seeds on all 8 valid groups, and on MUTAG-GT does not beat it under R1c")
    ok1 = True
    for g in C.VALID:
        n = rep["oracle"][f"{g}|prob|fidp|F|conn"]["signs"].count("+"); ok1 &= n >= 4; P(f"    {g:<22} {n}/5  {'meets' if n >= 4 else 'does not meet'}")
    mg = "MUTAG-GT (10 seeds)"; us = G[mg]; d = C.boot([Z[u]["G_gt_R1c_fidp_prob"] for u in us], [Z[u]["G_conn_R1c_fidp_prob"] for u in us]); nb = rep["oracle"][f"{mg}|prob|fidp|R1c|conn"]["signs"].count("+")
    ok2 = not d[1] > 0 and nb < 8
    P(f"    MUTAG-GT R1c Fid+ ground truth - random-connected {fmt_ci(*d)}; {nb}/10 seeds significantly better -> {'does not beat' if ok2 else 'beats'}")
    dF = C.boot([Z[u]["G_gt_F_fidp_prob"] for u in us], [Z[u]["G_conn_F_fidp_prob"] for u in us]); P(f"    (for reference, MUTAG-GT under F {fmt_ci(*dF)}; logit R1c counts {C.counts(rep['oracle'][f'{mg}|logit|fidp|R1c|conn']['signs'])})")
    rep["P-F1"] = {"part1": ok1, "part2": ok2, "held": ok1 and ok2, "mutag_r1c": list(d), "mutag_r1c_better": nb}
    P(f"  P-F1 {'HELD' if ok1 and ok2 else 'FAILED'} (part 1 {'held' if ok1 else 'failed'}, part 2 {'held' if ok2 else 'failed'})")
    # rankings
    P("\n  [4] T2 repeat: ranking of GNN, PG, SX fixed. Kendall tau with the top-k GEA ranking, median over units (range) per group; 3 items, so tau is -1, -1/3, 1/3 or 1")
    MET = [("Fid+", "fidp_prob", True), ("Fid-", "fidm_prob", False), ("GEF", "gef", False)]; vals = {}
    for g, us in G.items():
        for u in us:
            ga = _gea_rank(u)
            for lab, key, hb in MET:
                vals.setdefault((g, lab, "F"), []).append(RA.tau(RA.rank_of(_f_means(u, key), hb), ga))
                for r in RUL:
                    vals.setdefault((g, lab, r), []).append(RA.tau(RA.rank_of(_r2_means(u, r, key), hb), ga))
    P(f"  {'group':<22}{'metric':<6}" + "".join(f"{r:>12}" for r in RUL + ["F"]))
    rep["tau_gea"] = {}
    for g in G:
        for lab, _, _ in MET:
            P(f"  {g:<22}{lab:<6}" + "".join(f"{np.median(vals[(g, lab, r)]):>+12.2f}" for r in RUL + ["F"]))
            for r in RUL + ["F"]:
                rep["tau_gea"][f"{g}|{lab}|{r}"] = vals[(g, lab, r)]
    P("\n  all rules in one table: valid groups (40 units), median tau with GEA; share of units with tau = 1")
    allv = {(lab, r): [t for g in C.VALID for t in vals[(g, lab, r)]] for lab, _, _ in MET for r in RUL + ["F"]}
    for lab, _, _ in MET:
        P(f"    {lab:<6}" + "".join(f"{r} {np.median(allv[(lab, r)]):+.2f} ({np.mean(np.isclose(allv[(lab, r)], 1)):.2f})   " for r in RUL + ["F"]))
    same = []
    for g in C.VALID:
        for u in G[g]:
            for lab, key, hb in MET:
                same += [RA.tau(RA.rank_of(_f_means(u, key), hb), RA.rank_of(_r2_means(u, r, key), hb)) for r in RUL]
    P(f"    same metric, F vs each of R1, R2, R3: median tau {np.median(same):+.2f}")
    a, b = float(np.median(allv[("Fid+", "F")])), float(np.median(allv[("Fid+", "R2")])); heldf2 = a > b
    tv = lambda x: {f"{v:+.2f}": int(np.sum(np.isclose(x, v))) for v in (-1, -1 / 3, 1 / 3, 1)}
    P(f"\n  P-F2: median tau(Fid+, GEA) over 40 units: F {a:+.2f}, R2 {b:+.2f} -> {'HELD' if heldf2 else 'FAILED'} (a tie counts as failed)")
    P(f"    tau counts F {tv(allv[('Fid+', 'F')])}; R2 {tv(allv[('Fid+', 'R2')])}; mean tau F {np.mean(allv[('Fid+', 'F')]):+.3f}, R2 {np.mean(allv[('Fid+', 'R2')]):+.3f}")
    rep["P-F2"] = {"F": a, "R2": b, "held": heldf2, "counts_F": tv(allv[("Fid+", "F")]), "counts_R2": tv(allv[("Fid+", "R2")])}
    # secondary: 7 methods
    t7f, t7r = [], []
    for g in C.VALID:
        for u in G[g]:
            H, Gg, S = load(f"{T2}/H/gea/{u}.npz"), load(f"{T2}/G/gea/{u}.npz"), load(f"{T2}/S/gea/{u}.npz"); gea = [H["gnnexplainer"].mean(), H["pgexplainer"].mean(), S["sx_fixed"].mean(), Gg["sal"].mean(), Gg["ig"].mean(), Gg["ixg"].mean(), H["hetero"].mean()]
            E_, Sf, Gf, Hf = (load(f) for f in (f"runs/rq1_revision/pass/E/{u}.npz", f"{T2}/S/fid/{u}.npz", f"{T2}/G/fid/{u}.npz", f"{T2}/H/fid/{u}.npz"))
            r2 = [E_["gnnexplainer_R2_fidp_prob"].mean(), E_["pgexplainer_R2_fidp_prob"].mean(), Sf["sx_fixed_R2_fidp_prob"].mean(), Gf["sal_R2_fidp_prob"].mean(), Gf["ig_R2_fidp_prob"].mean(), Gf["ixg_R2_fidp_prob"].mean(), Hf["hetero_R2_fidp_prob"].mean()]
            t7f.append(kendalltau([Z[u][f"{m}_fidp_prob"].mean() for m in METHODS], gea).statistic); t7r.append(kendalltau(r2, gea).statistic)
    P(f"  secondary, not scored: 7 methods, median tau(Fid+, GEA): F {np.nanmedian(t7f):+.2f}, R2 {np.nanmedian(t7r):+.2f}")
    rep["secondary_7"] = {"F": float(np.nanmedian(t7f)), "R2": float(np.nanmedian(t7r))}
    save("F", rep)


# ------------------------------------------------------------------ X
def X():
    rep = {}
    M = {"Mutagenicity D-MPNN s5-9": [f"mutagenicity_dmpnn_s{s}" for s in range(5, 10)], "B-XAIC indole s5-9": [f"bxaic_indole_s{s}" for s in range(5, 10)]}
    fm = lambda u: load(f"{T}/X/fid/M/{u}.npz")
    P("X -- SMALL GAPS AND THE NEW FIGURE")
    rep["M"] = T2R.fid_table(M, fm, fm, ["gnnexplainer", "pgexplainer", "sx_fixed", "ig", "hetero"], "\n  [1] Fid+ / Fid- vs random-node, item-M seeds 5-9; Holm family per (unit, method, space) = 6")
    R = {f"{d} {k}": [f"{b}_s{s}_{k}" for s in range(5)] for d, b in (("Mutagenicity D-MPNN", "mutagenicity_dmpnn"), ("B-XAIC indole", "bxaic_indole")) for k in ("full", "head")}
    fr = lambda u: load(f"{T}/X/fid/R/{u}.npz"); fs = lambda u: load(f"{T}/X/fid/R/{u}_sx.npz")
    rep["R"] = T2R.fid_table(R, fr, fr, ["gnnexplainer", "pgexplainer", "ig", "sal"], "\n  [2a] the same, item-R randomised models, all explained molecules")
    rep["R_sx"] = T2R.fid_table(R, fs, fs, ["sx_fixed"], "\n  [2b] the same, item-R randomised models, SubgraphX fixed budget on its 100-molecule subset")
    n = json.load(open("figures/final/fig1_v2_numbers.json"))
    P("\n  [3] numbers shown in figures/final/fig1_v2 (mean over seeds of per-seed mean GEA; minus random-node with 95% CI; display only: pooled box medians)")
    for panel, x in n.items():
        P(f"\n  {panel}: {x['n_seeds']} seeds, molecules per method {x['n_molecules']}")
        P(f"    random-node band {x['rn_band'][0]:.3f}-{x['rn_band'][1]:.3f}; best trivial rule {x['best_rule']} band {x['rule_band'][0]:.3f}-{x['rule_band'][1]:.3f} (chosen after seeing the panel; means: {', '.join(f'{k} {v:.3f}' for k, v in x['rule_means'].items())})")
        for m, v in x["methods"].items():
            P(f"    {m:<14} mean {v['mean']:.3f}  minus random-node {fmt_ci(*v['minus_rn'])}  box median {v['median']:.3f}  per-seed " + " ".join(f"{a:.3f}" for a in v["per_seed"]))
        P(f"    PG final rule, not shown: mean {x['pg_final_rule']['mean']:.3f}, minus random-node {fmt_ci(*x['pg_final_rule']['minus_rn'])}")
    rep["fig"] = n
    save("X", rep)


if __name__ == "__main__":
    {"E": E, "F": F, "X": X}[sys.argv[1]]()
