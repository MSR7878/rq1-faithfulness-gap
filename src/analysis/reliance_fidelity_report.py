"""
Report for RQ1 Part A (ground-truth reliance) and Part B (logit / random-
corrected fidelity). Reads runs/rq1_reliance_fidelity/part{A,B}/<unit>.npz
(written by src/analysis/reliance_fidelity.py) and, for the GEA comparison,
the existing GEA arrays (read-only). Scoring rules:
configs/predictions_reliance_fidelity.md.

    python -m src.analysis.reliance_fidelity_report | tee runs/rq1_reliance_fidelity/report.txt
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..data.standard import TOX21_TASKS
from .mutagenicity_cv_report import rank_direction
from .mutagenicity_report import holm, star, wpair
from .reliance_fidelity import ARCH_DIRS, BXAIC, EXPL, MASKINGS, OUT, SEEDS

SH = {"gnnexplainer": "GNN", "pgexplainer": "PG", "subgraphx": "SX"}
P = print


def load(part, name):
    f = f"{OUT}/part{part}/{name}.npz"
    return dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None


def test(a, b=None):
    """paired Wilcoxon of a vs b (or a vs 0); returns (p_raw, rank direction)."""
    a = np.asarray(a, float); b = np.zeros_like(a) if b is None else np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 6:
        return 1.0, 0
    return wpair(a, b), rank_direction(a, b)


def family(raw: dict) -> dict:
    """raw: key -> (p, dir). Holm over the keys; returns key -> '+', '-' or '.' plus p_holm."""
    adj = holm({k: v[0] for k, v in raw.items()})
    return {k: ("+" if (raw[k][1] > 0 and adj[k] < 0.05) else "-" if (raw[k][1] < 0 and adj[k] < 0.05) else ".", adj[k]) for k in raw}


def counts(signs):
    return f"{signs.count('+')}/{signs.count('-')}/{signs.count('.')}"


# ============================================================================ Part A
A_GROUPS = [("MUTAG-GT, D-MPNN", [f"mutag_graphxai_s{s}" for s in SEEDS])] + \
           [(f"Mutagenicity, {a.upper() if a != 'dmpnn' else 'D-MPNN'}", [f"mutagenicity_{a}_s{s}" for s in SEEDS]) for a in ARCH_DIRS] + \
           [(f"B-XAIC {t}, D-MPNN", [f"bxaic_{t}_s{s}" for s in SEEDS]) for t in BXAIC]
A_TESTS = [("R1", "conn"), ("R2", "conn"), ("R1", "node"), ("R2", "node")]


def part_a(rep):
    P("=" * 130); P(" PART A -- ground-truth reliance: delta_GT = m(G) - m(G without S), null = same for 20 random sets of size |S|"); P("=" * 130)
    P(" margin m = logit_c - logit_other (c = predicted class on G); per-mask values averaged over a molecule's valid GT masks;")
    P(" tests per seed: gap vs 0 for {R1,R2} x {random-connected, random-node}, Holm over the 4; direction = rank test.")
    rep["A"] = {}
    for title, units in A_GROUPS:
        D = {u: load("A", u) for u in units}
        if any(v is None for v in D.values()):
            P(f"\n-- {title}: MISSING {[u for u, v in D.items() if v is None]}"); rep["A"][title] = "missing"; continue
        P(f"\n-- {title}")
        P(f"  {'seed':>4}{'n':>5}{'skip':>5} |{'dGT R1':>8}{'conn R1':>9}{'gap':>8}{'':>3} |{'dGT R2':>8}{'conn R2':>9}{'gap':>8}{'':>3}"
          f" |{'node R1 gap':>12}{'':>3}{'node R2 gap':>12}{'':>3} | suff GT vs conn (expl.)")
        sig = {t: [] for t in A_TESTS}; suff = []; rows = {}
        for s, u in zip(SEEDS, units):
            a = D[u]; ok = ~np.isnan(a["delta_gt_R1"])
            g = {(mk, ref): a[f"delta_gt_{mk}"][ok] - a[f"null_{ref}_{mk}"][ok] for mk, ref in A_TESTS}
            F = family({t: test(g[t]) for t in A_TESTS})
            for t in A_TESTS:
                sig[t].append(F[t][0])
            sp, sd = test(a["suff_gt"][ok], a["suff_null_conn"][ok])
            ss = "+" if (sd > 0 and sp < 0.05) else "-" if (sd < 0 and sp < 0.05) else "."
            suff.append(ss)
            mean = lambda k: float(np.mean(a[k][ok]))
            P(f"  {s:>4}{int(ok.sum()):>5}{int(np.sum(a['n_skipped'])):>5} |{mean('delta_gt_R1'):>8.3f}{mean('null_conn_R1'):>9.3f}"
              f"{float(np.mean(g[('R1','conn')])):>8.3f}{F[('R1','conn')][0]:>3} |{mean('delta_gt_R2'):>8.3f}{mean('null_conn_R2'):>9.3f}"
              f"{float(np.mean(g[('R2','conn')])):>8.3f}{F[('R2','conn')][0]:>3} |{float(np.mean(g[('R1','node')])):>12.3f}{F[('R1','node')][0]:>3}"
              f"{float(np.mean(g[('R2','node')])):>12.3f}{F[('R2','node')][0]:>3} | {mean('suff_gt'):.3f} vs {mean('suff_null_conn'):.3f} {ss}")
            rows[s] = {"n": int(ok.sum()), "skipped_masks": int(np.sum(a["n_skipped"])), "mean_masks": float(np.mean(a["n_masks"][ok])),
                       **{f"{k}": mean(k) for k in ("delta_gt_R1", "delta_gt_R2", "null_conn_R1", "null_conn_R2", "null_node_R1", "null_node_R2",
                                                     "suff_gt", "suff_null_conn", "m_full")},
                       "signs": {f"{mk}_{ref}": F[(mk, ref)][0] for mk, ref in A_TESTS}, "p_holm": {f"{mk}_{ref}": F[(mk, ref)][1] for mk, ref in A_TESTS},
                       "suff_sign": ss}
        P("  direction counts above/below/n.s. over 5 seeds: " + "   ".join(f"{mk}-{ref} {counts(sig[(mk, ref)])}" for mk, ref in A_TESTS)
          + f"   | sufficiency (exploratory, uncorrected) {counts(suff)}")
        rep["A"][title] = {"per_seed": rows, "counts": {f"{mk}_{ref}": counts(sig[(mk, ref)]) for mk, ref in A_TESTS}, "suff_counts": counts(suff),
                           "signs": {f"{mk}_{ref}": sig[(mk, ref)] for mk, ref in A_TESTS}}
    # predictions
    P("\n  REGISTERED PREDICTIONS")
    mu = rep["A"].get("Mutagenicity, D-MPNN"); mg = rep["A"].get("MUTAG-GT, D-MPNN")
    if isinstance(mu, dict):
        r1, r2 = mu["signs"]["R1_conn"].count("+"), mu["signs"]["R2_conn"].count("+")
        held = r1 >= 4 and r2 >= 4
        P(f"  (A1) Mutagenicity D-MPNN, delta_GT significantly above the random-connected null in >= 4/5 seeds under R1 and R2:"
          f" R1 {r1}/5, R2 {r2}/5 -> {'HELD' if held else 'DID NOT HOLD'}")
        rep["A1"] = {"R1_above": r1, "R2_above": r2, "held": held}
    if isinstance(mg, dict):
        r1, r2 = 5 - mg["signs"]["R1_conn"].count("+"), 5 - mg["signs"]["R2_conn"].count("+")
        held = r1 >= 3 and r2 >= 3
        P(f"  (A2) MUTAG-GT D-MPNN, delta_GT NOT significantly above the random-connected null in >= 3/5 seeds (R1 and R2):"
          f" R1 {r1}/5, R2 {r2}/5 not above -> {'HELD' if held else 'DID NOT HOLD'}")
        rep["A2"] = {"R1_not_above": r1, "R2_not_above": r2, "held": held}


# ============================================================================ Part B
B_GROUPS = [("MUTAG (single split)", ["mutag"]), ("BBBP (single split)", ["bbbp"]),
            ("MUTAG-GT, D-MPNN", [f"mutag_graphxai_s{s}" for s in SEEDS])] + \
           [(f"Tox21 {ep}, D-MPNN", [f"tox21_{ep}_s{s}" for s in SEEDS]) for ep in TOX21_TASKS] + \
           [(f"B-XAIC {t}, D-MPNN", [f"bxaic_{t}_s{s}" for s in SEEDS]) for t in BXAIC] + \
           [(f"Mutagenicity, {a.upper() if a != 'dmpnn' else 'D-MPNN'}", [f"mutagenicity_{a}_s{s}" for s in SEEDS]) for a in ARCH_DIRS]
B1_DATASETS = {"MUTAG-GT": [f"mutag_graphxai_s{s}" for s in SEEDS], "BBBP": ["bbbp"],
               "Tox21": [f"tox21_{ep}_s{s}" for ep in TOX21_TASKS for s in SEEDS],
               "Mutagenicity": [f"mutagenicity_dmpnn_s{s}" for s in SEEDS]}
B_KEYS = [(e, mk, met) for e in EXPL for mk in MASKINGS for met in ("fid_plus", "fid_minus")]


def cv(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1) / abs(np.mean(x))) if len(x) > 1 and np.mean(x) != 0 else float("inf")


def b_signs(a):
    """per unit: random-corrected logit value vs 0 for the 18 tests, one Holm family per reference."""
    out = {}
    for ref in ("node", "conn"):
        raw = {(e, mk, met): test(a[f"{e}_{mk}_{met}_logit"] - a[f"rand_{ref}_{mk}_{met}_logit"]) for e, mk, met in B_KEYS}
        F = family(raw)
        out[ref] = {k: F[k][0] for k in raw}
    return out


def part_b(rep):
    P("\n" + "=" * 130); P(" PART B -- logit-margin fidelity and random-corrected fidelity"); P("=" * 130)
    P(" Fid+ = m(G) - m(G\\E), Fid- = m(G) - m(E); E = top-k 0.25 of the cached explanation; random-corrected = value - mean of 20 draws.")
    P(" tests per unit: corrected value vs 0, 3 explainers x 3 maskings x 2 metrics = 18, Holm over the 18 (one family per reference).")
    rep["B"] = {}
    U = {}
    for _, units in B_GROUPS:
        for u in units:
            U[u] = load("B", u)
    missing = [u for u, v in U.items() if v is None]
    P(f"\n  units loaded {len(U) - len(missing)}/{len(U)}" + (f"   MISSING: {missing}" if missing else ""))
    devs = [float(v["validation_max_abs_dev"]) for v in U.values() if v is not None]   # exempt cells (amendment) reported in the tie section
    mism = sum(int((~v["pred_matches_target"].astype(bool)).sum()) for v in U.values() if v is not None)
    P(f"  validation: recomputed probability Fid+/Fid- vs stored per-molecule values, max |dev| over all units {max(devs):.1e};"
      f" molecules whose prediction differs from the cached target: {mism}")
    rep["B_validation"] = {"max_abs_dev": max(devs), "pred_target_mismatch": mism, "missing": missing}

    # ---- B1
    P("\n  [B1] CV = sd/|mean| of per-molecule Fid+ under R3 (mean fill); dataset value = median over units (range in brackets)")
    P(f"  {'dataset':<14}{'units':>6} | " + " | ".join(f"{SH[e]:>3} logit CV {'':>12} prob CV {'':>12}" for e in EXPL))
    rep["B1"] = {}; ok_all = True
    for ds, units in B1_DATASETS.items():
        cells = []
        rep["B1"][ds] = {}
        for e in EXPL:
            lc = [cv(U[u][f"{e}_R3_fid_plus_logit"]) for u in units if U[u] is not None]
            pc = [cv(U[u][f"{e}_R3_fid_plus_prob"]) for u in units if U[u] is not None]
            ml, mp = float(np.median(lc)), float(np.median(pc))
            ok_all &= ml < 1
            cells.append(f"{ml:>7.2f} [{min(lc):.2f}-{max(lc):.2f}]{'':>2}{mp:>7.2f} [{min(pc):.2f}-{max(pc):.2f}]")
            rep["B1"][ds][e] = {"logit_cv_median": ml, "logit_cv_range": [min(lc), max(lc)], "prob_cv_median": mp, "prob_cv_range": [min(pc), max(pc)]}
        P(f"  {ds:<14}{len(units):>6} | " + " | ".join(cells))
    P(f"  (B1) logit CV < 1 for all 3 explainers on all 4 diffuse datasets -> {'HELD' if ok_all else 'DID NOT HOLD'}")
    rep["B1"]["held"] = bool(ok_all)

    # ---- random-corrected fidelity
    P("\n  [B2] random-corrected logit fidelity: mean corrected value over seeds, and direction counts above/below/n.s. over the units")
    rep["B2"] = {}
    for title, units in B_GROUPS:
        D = {u: U[u] for u in units if U[u] is not None}
        if not D:
            continue
        single = len(units) == 1
        P(f"\n  -- {title}" + ("   (single split: descriptive; signs shown for that one unit)" if single else ""))
        S = {u: b_signs(a) for u, a in D.items()}
        P(f"  {'':<4}{'':<4}" + "".join(f"{mk + ' ' + met.replace('fid_', 'Fid').replace('plus', '+').replace('minus', '-'):>30}" for mk in MASKINGS for met in ("fid_plus", "fid_minus")))
        rep["B2"][title] = {}
        for e in EXPL:
            for ref in ("node", "conn"):
                line = f"  {SH[e]:<4}{ref:<5}"
                for mk in MASKINGS:
                    for met in ("fid_plus", "fid_minus"):
                        corr = [float(np.mean(a[f"{e}_{mk}_{met}_logit"] - a[f"rand_{ref}_{mk}_{met}_logit"])) for a in D.values()]
                        sg = [S[u][ref][(e, mk, met)] for u in D]
                        line += f"{np.mean(corr):>12.3f} {counts(sg):>13}    "[:30]
                        rep["B2"][title][f"{e}|{ref}|{mk}|{met}"] = {"mean_corrected": float(np.mean(corr)), "signs": sg, "counts": counts(sg)}
                P(line)

    # Tox21 totals over the 60 units
    P("\n  Tox21, totals over all 60 units (12 endpoints x 5 seeds), above/below/n.s.:")
    tox = [u for ep in TOX21_TASKS for u in (f"tox21_{ep}_s{s}" for s in SEEDS) if U[u] is not None]
    S = {u: b_signs(U[u]) for u in tox}
    rep["B2"]["Tox21 totals"] = {}
    for e in EXPL:
        for ref in ("node", "conn"):
            parts = []
            for mk in MASKINGS:
                for met in ("fid_plus", "fid_minus"):
                    sg = [S[u][ref][(e, mk, met)] for u in tox]
                    parts.append(f"{mk} {met.replace('fid_', 'Fid').replace('plus', '+').replace('minus', '-')} {counts(sg)}")
                    rep["B2"]["Tox21 totals"][f"{e}|{ref}|{mk}|{met}"] = counts(sg)
            P(f"   {SH[e]:<3} vs {ref:<5} " + "   ".join(parts))
    return U


# ============================================================================ GEA comparison
def gea_compare(rep, U):
    P("\n" + "=" * 130); P(" GEA vs random-node-corrected logit Fid+ under R3: per-unit direction (+ above / - below / . n.s.)"); P("=" * 130)
    AUD = "runs/mutagenicity/exploratory/null_audit_arrays"
    src = {f"mutag_graphxai_s{s}": (f"{AUD}/mutag_graphxai_s{s}.npz", "{e}_topk", "rn_k") for s in SEEDS}
    for t in BXAIC:
        src.update({f"bxaic_{t}_s{s}": (f"{AUD}/bxaic_{t}_s{s}.npz", "{e}_topk", "rn_k") for s in SEEDS})
    for a, d in ARCH_DIRS.items():
        src.update({f"mutagenicity_{a}_s{s}": (f"{d}/arrays_s{s}.npz", "{e}_topk_set", "rand_node_topk_set") for s in SEEDS})
    diffs, rep["GEA_vs_Fid"] = [], {}
    for u, (f, rk, nk) in src.items():
        if U.get(u) is None or not os.path.exists(f):
            continue
        z = np.load(f)
        F = family({e: test(z[rk.format(e=e)], z[nk]) for e in EXPL})
        sb = b_signs(U[u])["node"]
        for e in EXPL:
            g, fp = F[e][0], sb[(e, "R3", "fid_plus")]
            rep["GEA_vs_Fid"][f"{u}|{e}"] = {"gea": g, "fidplus_R3": fp}
            if g != fp:
                diffs.append((u, SH[e], g, fp))
    P(f"  cells compared: {len(rep['GEA_vs_Fid'])};  cells whose direction differs: {len(diffs)}")
    for u, e, g, fp in diffs:
        P(f"    {u:<28}{e:<4} GEA {g}   Fid+ R3 (random-node corrected) {fp}")
    rep["GEA_vs_Fid_differ"] = diffs


# ============================================================================ tie-break findings
def tie_findings(rep, U):
    P("\n" + "=" * 130); P(" TIE-BREAK FINDINGS -- SubgraphX on the two single-split runs (amendment of 2026-10-19; exploratory)"); P("=" * 130)
    rep["ties"] = {}
    for name in ("mutag", "bbbp"):
        a, f = U.get(name), f"{OUT}/ties/{name}_subgraphx_ties.npz"
        if a is None or not os.path.exists(f):
            P(f"  {name}: MISSING"); continue
        z = dict(np.load(f, allow_pickle=True)); n = len(z["mol_idx"])
        P(f"\n  -- {name.upper()}  (n = {n}; SubgraphX coalition != top-k budget in {int((z['coalition_size'] != z['budget']).sum())} molecules;"
          f" recomputation max |dev| from stored {float(a['validation_exempt_max_abs_dev']):.3f})")
        P(f"  {'':<14}{'stored (Sept)':>14}{'recomputed now':>16}{'mols differing':>16} | 5 random tie orders: mean of molecule averages [min, max];  mean per-molecule range")
        r = {}
        for mk in MASKINGS:
            for met in ("fid_plus", "fid_minus"):
                st, cur = a[f"subgraphx_{mk}_{met}_prob_stored"], a[f"subgraphx_{mk}_{met}_prob"]
                lab = f"{mk} {met.replace('fid_', 'Fid').replace('plus', '+').replace('minus', '-')}"
                for sp in ("prob", "logit"):
                    o = z[f"{mk}_{met}_{sp}"]; avg = o.mean(0); rng_ = (o.max(1) - o.min(1)).mean()
                    left = (f"{float(st.mean()):>14.3f}{float(cur.mean()):>16.3f}{int((np.abs(st - cur) > 1e-4).sum()):>16}" if sp == "prob" else f"{'':>46}")
                    P(f"  {lab + ' ' + sp:<14}{left} | {avg.mean():>8.3f} [{avg.min():.3f}, {avg.max():.3f}];  {rng_:.3f}")
                    r[f"{mk}|{met}|{sp}"] = {"orders_mean": float(avg.mean()), "orders_min": float(avg.min()), "orders_max": float(avg.max()),
                                             "mean_per_molecule_range": float(rng_)}
                r[f"{mk}|{met}|stored_vs_current"] = {"stored_mean": float(st.mean()), "current_mean": float(cur.mean()),
                                                      "n_differing": int((np.abs(st - cur) > 1e-4).sum())}
        if bool(z["gt_from_mutag_graphxai"]):
            g = z["gea_topk"]; avg = g.mean(0)
            P(f"  GEA top-k (MUTAG-GT mask; graphs verified identical): current order {np.nanmean(z['gea_topk_current']):.3f};"
              f" 5 orders {avg.mean():.3f} [{avg.min():.3f}, {avg.max():.3f}]; mean per-molecule range {(g.max(1) - g.min(1)).mean():.3f}")
            r["gea"] = {"current": float(np.nanmean(z["gea_topk_current"])), "orders_mean": float(avg.mean()), "orders_min": float(avg.min()), "orders_max": float(avg.max())}
        else:
            P("  GEA: not applicable (no ground truth" + (" for BBBP)" if name == "bbbp" else "; MUTAG graphs not identical to MUTAG-GT's)"))
        rep["ties"][name] = r


def main():
    rep = {}
    part_a(rep)
    U = part_b(rep)
    gea_compare(rep, U)
    tie_findings(rep, U)
    json.dump(rep, open(f"{OUT}/report.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    P(f"\nsaved -> {OUT}/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
