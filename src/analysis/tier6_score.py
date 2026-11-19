"""
RQ1 Tier 6 (runs/rq1_tier6/PLAN.md): per-unit numbers.

  t1 --unit U   aromatic share of the chosen atoms (top-k, random ties, 20 orders) minus 20 random-connected sets of the same size,
                EiG / GNNShap / SAME / GStarX / Saliency / IG (+ EiG own-size set), molecules with 0 < |scaffold| < N
                -> runs/rq1_tier6/T1/<unit>.npz
  t2 --unit U   (a) the molecules with no aromatic atom: full-graph GEA and the random-node null;
                (b) scaffold hidden (R1, R2): GEA on the remaining atoms at the primary and secondary k, random-node null on the
                remaining atoms, prediction changes, all-tied scores, errors; the full-graph GEA on the same molecules ("before")
                -> runs/rq1_tier6/T2/<unit>.npz
  t3 --unit U   as N2: GEA trained vs full reset, Spearman, top-k Jaccard (EiG, GNNShap, Saliency, IG) -> runs/rq1_tier6/T3/<unit>.npz

    python -m src.analysis.tier6_score t1|t2|t3 --unit U
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch
from scipy.stats import spearmanr

from . import tier2_common as C
from . import tier5_common as T5
from .connectivity_baseline import build_adj, random_connected_subset
from .tier5_N1 import new_rows
from .tier5_N23 import jac, topk_set
from .tier6_explain import T2_METHODS, T6, aromatic, reset_ckpt, subset100, t1_mols, t2_mols

T1_METHODS = ["eig", "gnnshap", "same", "gstarx", "sal", "ig"]


def rows_in(d):
    return {int(r["mol_idx"]): r for f in sorted(glob.glob(f"{d}/shard_*_of_*.pt")) for r in C.torch_load(f)}


def test_set(unit):
    return {int(i) for i in torch.load(T5.spec(unit)["ckpt"], map_location="cpu", weights_only=False)["split"]["test"]}


def t1_scores(unit):
    """method -> {mi: scores}; GStarX / SAME seeds 0-4 from N1, seeds 5-9 from T1."""
    old = T5.old_scores(unit); out = {"sal": old["sal"], "ig": old["ig"]}
    for w in ("eig", "gnnshap"):
        r = new_rows(unit, w); out[w] = {m: np.asarray(v["node"], float) for m, v in r.items()}
        if w == "eig":
            out["eig_own"] = {m: np.asarray(v["own"], float) > 0.5 for m, v in r.items()}
    for w in ("gstarx", "same"):
        if T5.seed_of(unit) < 5:
            r = new_rows(unit, w); want = T5.level_mols(unit, "L2" if w == "gstarx" else "L4")
        else:
            r = rows_in(f"{T6}/T1/expl/{w}/{unit}"); want = t1_mols(unit, w)
        miss = [m for m in want if m not in r]
        if miss:
            raise RuntimeError(f"{unit} {w}: {len(miss)} molecules missing")
        out[w] = {m: np.asarray(r[m]["node"], float) for m in want}
    return out


def t1(unit):
    dl, _ = C.load_data(C.dataset_of(unit)); S = t1_scores(unit); te = test_set(unit); A = {"unit": unit}
    for w in T1_METHODS + ["eig_own"]:
        sh, nul, mols, size, ist = [], [], [], [], []
        for m in sorted(S[w]):
            d = dl[m]; n = d.num_nodes; sc = aromatic(d)
            if not 0 < len(sc) < n:
                continue
            arom = np.zeros(n, bool); arom[sc] = True; adj = build_adj(d.edge_index.cpu(), n)
            share = lambda mask: float((mask & arom).sum() / max(1, mask.sum()))
            if w == "eig_own":
                msk = S[w][m]; s = int(msk.sum()); v = share(msk) if s else np.nan; r = C.mol_rng(f"{unit}|T6|conn|own", m)
            else:
                s = C.kbudget(n); v = C.gea_random_ties(S[w][m], s, share, f"{unit}|{w}|T6arom", m); r = C.mol_rng(f"{unit}|T6|conn", m)
            nul.append(float(np.mean([arom[list(random_connected_subset(adj, n, s, r))].mean() for _ in range(C.N_DRAWS)])) if s else np.nan)
            sh.append(v); mols.append(m); size.append(s); ist.append(m in te)
        for k, v in (("share", sh), ("conn", nul), ("mols", mols), ("size", size), ("is_test", ist)):
            A[f"{w}__{k}"] = np.array(v)
    C.save_npz(f"{T6}/T1/{unit}.npz", **A); print(f"{unit}: T1 scored", flush=True)


def gea_on(scores, gt, k, stream, mi):
    g = np.asarray(gt, bool)

    def scr(mask):
        u = int((mask | g).sum()); return int((mask & g).sum()) / u if u else 0.0
    return C.gea_random_ties(np.asarray(scores, float), k, scr, stream, mi), scr


def t2(unit):
    dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); old = T5.old_scores(unit); A = {"unit": unit}
    full = {w: old[w] for w in ("gnnexplainer", "sx_fixed", "ig")}
    for w in ("eig", "gnnshap"):
        full[w] = {m: np.asarray(v["node"], float) for m, v in new_rows(unit, w).items()}
    # (a) no aromatic atom
    noar = [m for m in T5.gt_mols(unit) if not aromatic(dl[m])]
    A["a__mols"] = np.array(noar); A["a__rn"] = T5.null_arrays(unit, noar)["rn"]
    for w in T2_METHODS:
        A[f"a__{w}"] = C.gea_array(unit, w, lambda x, s=full[w]: s[x], noar, dkey)
    # before: full graph on the molecules with a scaffold
    mix = t2_mols(unit); A["b__mols"] = np.array(mix); A["full__rn"] = T5.null_arrays(unit, mix)["rn"]
    for w in T2_METHODS:
        A[f"full__{w}"] = C.gea_array(unit, w, lambda x, s=full[w]: s[x], mix, dkey)
    # (b) hidden; nulls on the remaining atoms
    info = {}
    for m in mix:
        d = dl[m]; n = d.num_nodes; rem = [v for v in range(n) if v not in set(aromatic(d))]; nr = len(rem)
        gt = d.node_gt_mask.cpu().numpy().astype(bool)[rem]
        kp = max(1, int(0.25 * nr)); ks = min(nr, max(1, int(0.25 * n))); info[m] = (rem, gt, kp, ks)
    for tag, idx in (("p", 2), ("s", 3)):
        rn = []
        for m in mix:
            rem, gt, kp, ks = info[m]; k = (kp, ks)[idx - 2]; r = C.mol_rng(f"{unit}|T6|rn|k{k}", m); _, scr = gea_on(np.zeros(len(rem)), gt, k, "x", m)
            draws = []
            for _ in range(C.N_DRAWS):
                msk = np.zeros(len(rem), bool); msk[r.permutation(len(rem))[:k]] = True; draws.append(scr(msk))
            rn.append(float(np.mean(draws)))
        A[f"b__rn_{tag}"] = np.array(rn); A[f"b__k_{tag}"] = np.array([info[m][idx] for m in mix])
    A["b__nrem"] = np.array([len(info[m][0]) for m in mix])
    for rule in ("R1", "R2"):
        for w in T2_METHODS:
            R = rows_in(f"{T6}/T2/expl/{rule}/{w}/{unit}"); miss = [m for m in mix if m not in R]
            if miss:
                raise RuntimeError(f"{unit} {rule} {w}: {len(miss)} molecules not run")
            gp, gs, tied, err, flip, sec = [], [], [], [], [], []
            for m in mix:
                r = R[m]; rem, gt, kp, ks = info[m]
                if r["rem"] != rem:
                    raise RuntimeError(f"{unit} {rule} {w} {m}: remaining atoms differ")
                flip.append(r["pred_full"] != r["pred_hidden"]); sec.append(r["seconds"])
                if "error" in r:
                    err.append(True); tied.append(False); gp.append(np.nan); gs.append(np.nan); continue
                s = np.asarray(r["node"], float); err.append(False); tied.append(bool(s.max() == s.min()))
                gp.append(gea_on(s, gt, kp, f"{unit}|{w}|{rule}|T6gea", m)[0]); gs.append(gea_on(s, gt, ks, f"{unit}|{w}|{rule}|T6gea", m)[0])
            for k, v in (("gea_p", gp), ("gea_s", gs), ("tied", tied), ("error", err), ("flip", flip), ("sec", sec)):
                A[f"b__{rule}__{w}__{k}"] = np.array(v)
    C.save_npz(f"{T6}/T2/{unit}.npz", **A); print(f"{unit}: T2 scored ({len(noar)} without aromatic atoms, {len(mix)} with)", flush=True)


def t3(unit):
    dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); mols = subset100(unit); old = T5.old_scores(unit); A = {"unit": unit}
    ck = torch.load(reset_ckpt(unit), map_location="cpu", weights_only=False)
    A["reset_acc"] = ck["metrics"]["test_acc"]; A["reset_auroc"] = ck["metrics"]["test_auroc"]; A["reset_frac_pos"] = ck["metrics"]["frac_pred_pos"]
    tr = T5.spec(unit)["ckpt"]; tck = torch.load(tr, map_location="cpu", weights_only=False)
    A["trained_acc"] = tck["metrics"]["test_acc"]; A["trained_auroc"] = tck["metrics"]["test_auroc"]
    for w in ("eig", "gnnshap", "sal", "ig"):
        if w in ("eig", "gnnshap"):
            a_ = new_rows(unit, w); b_ = rows_in(f"{T6}/T3/expl/{w}/{unit}")
            ta = {m: np.asarray(a_[m]["node"], float) for m in mols if m in a_}; rb = {m: np.asarray(b_[m]["node"], float) for m in mols if m in b_}
        else:
            ta = old[w]; rb = C.npz_scores(f"{T6}/T3/grad/{unit}.npz", w)
        miss = [m for m in mols if m not in ta or m not in rb]
        if miss:
            raise RuntimeError(f"{unit} {w}: {len(miss)} molecules missing")
        g_t, g_r, sp, jc = [], [], [], []
        for m in mols:
            d = dl[m]; k = C.kbudget(d.num_nodes); scr = C.scorer_for(dkey, d, m); a, b = np.asarray(ta[m], float), np.asarray(rb[m], float)
            g_t.append(C.gea_random_ties(a, k, scr, f"{unit}|{w}|gea", m)); g_r.append(C.gea_random_ties(b, k, scr, f"{unit}|{w}|full|gea", m))
            sp.append(spearmanr(a, b).correlation if (a.max() > a.min() and b.max() > b.min()) else np.nan)
            jc.append(jac(topk_set(a, k, f"{unit}|{w}|jac", m), topk_set(b, k, f"{unit}|{w}|full|jac", m)))
        for nm, v in (("trained", g_t), ("reset", g_r), ("spearman", sp), ("jaccard", jc)):
            A[f"{w}__{nm}"] = np.array(v, float)
    A["mols"] = np.array(mols); C.save_npz(f"{T6}/T3/{unit}.npz", **A); print(f"{unit}: T3 scored", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["t1", "t2", "t3"]); a.add_argument("--unit"); a = a.parse_args()
    {"t1": t1, "t2": t2, "t3": t3}[a.cmd](a.unit)
