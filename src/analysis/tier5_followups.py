"""
RQ1 Tier 5 follow-ups F1-F6 (runs/rq1_tier5/PLAN_followups.md). Exploratory; no new predictions. Runs on ada (needs the .pt rows).

  extract --unit U   seconds per molecule (N1 rows); EiG own-size fraction, bonds chosen / cap / bonds, own-set GEA vs a size-matched
                     random-node null (20 sets); old methods: GEA of the dataset's best rule on their registered molecules (N4 all__)
                     -> F/extract/<unit>.npz
  cross --cell C     top-k Jaccard across training seeds on molecules explained in both seeds, all methods, with the exact random
                     baseline (two independent random k-sets) -> F/cross/<cell>.npz
  n3 --unit U        L6-format extras for N3: Jaccard of each run vs the N1 run, consensus size, k -> F/n3/<unit>.npz
  det --unit U       determinism: the F/det reruns (tag T5, same model) vs the N1 rows -> F/det/<unit>.npz

    python -m src.analysis.tier5_followups extract|cross|n3|det --unit U | --cell C
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
from scipy.stats import hypergeom

from . import tier2_common as C
from . import tier5_common as T5
from .tier5_N1 import OLD, new_rows, scale
from .tier5_N23 import N3_TAGS, jac, mols_for, topk_set

F = f"{T5.T5}/F"
N_SIZE_NULL = 20


def _nrows(unit, name, sc):
    lv = sc[name]["level"]; mols = T5.level_mols(unit, lv)
    if mols is None:
        return None, None
    rows = new_rows(unit, name)
    miss = [m for m in mols if m not in rows]
    if miss:
        raise RuntimeError(f"{unit} {name}: {len(miss)} molecules missing")
    return mols, rows


def extract(unit):
    sc = scale(); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); A = {"unit": unit}
    for name in T5.NEW:
        mols, rows = _nrows(unit, name, sc)
        if mols is None:
            continue
        A[f"{name}__mols"] = np.array(mols); A[f"{name}__sec"] = np.array([rows[m]["seconds"] for m in mols], float)
        A[f"{name}__n"] = np.array([rows[m]["n"] for m in mols])
        if name == "eig":
            own = [np.asarray(rows[m]["own"], float) > 0.5 for m in mols]
            A["eig__own_frac"] = np.array([o.sum() / len(o) for o in own])
            for k in ("bonds", "chosen_bonds", "max_bonds"):
                A[f"eig__{k}"] = np.array([rows[m]["meta_main"][k] for m in mols])
            g, nul = [], []
            for m, o in zip(mols, own):
                d = dl[m]; scr = C.scorer_for(dkey, d, m); s = int(o.sum()); r = C.mol_rng(f"{unit}|eig_own|sizenull", m)
                g.append(scr(o))
                draws = []
                for _ in range(N_SIZE_NULL):
                    msk = np.zeros(len(o), bool); msk[r.permutation(len(o))[:s]] = True; draws.append(scr(msk))
                nul.append(float(np.mean(draws)))
            A["eig_own__gea"] = np.array(g, float); A["eig_own__sizenull"] = np.array(nul, float)
    if unit in T5.n4_units():
        Z4 = np.load(f"{T5.T5}/N4/{unit}.npz", allow_pickle=True); mols = [int(m) for m in Z4["all__mols"]]
        rule = T5.RULE[dkey]; hd = C.heuristic_dict(unit, mols, rule)
        A["all__mols"] = np.array(mols); A["all__rule"] = C.gea_array(unit, rule, lambda x, s=hd: s[x], mols, dkey); A["rule"] = rule
    C.save_npz(f"{F}/extract/{unit}.npz", **A); print(f"{unit}: extracted", flush=True)


def _exp_jac(n, k):
    """E[|A&B| / |A|B|] for two independent uniform k-subsets of n atoms (overlap is hypergeometric)."""
    x = np.arange(0, k + 1); p = hypergeom(n, k, k).pmf(x)
    return float(np.sum(p * x / (2 * k - x)))


def cross(cell):
    cell = cell.replace("__", "|")                       # the job list writes cells as dataset__arch (no shell pipe)
    sc = scale(); us = sorted([u for u in T5.n1_units() if T5.cell_of(u) == cell], key=T5.seed_of)
    dkey = C.dataset_of(us[0]); dl, _ = C.load_data(dkey); S = {}
    for u in us:
        gt = set(T5.gt_mols(u)); src = T5.old_scores(u); S[u] = {w: {m: src[w][m] for m in gt if m in src[w]} for w in OLD}
        for name in T5.NEW:
            mols, rows = _nrows(u, name, sc)
            if mols is not None:
                S[u][name] = {m: np.asarray(rows[m]["node"], float) for m in mols}
    A = {"cell": cell, "units": np.array(us)}
    for w in OLD + T5.NEW:
        J, B, P, M = [], [], [], []
        for a, b in itertools.combinations([u for u in us if w in S[u]], 2):
            for m in sorted(set(S[a][w]) & set(S[b][w])):
                n = dl[m].num_nodes; k = C.kbudget(n)
                sa = topk_set(S[a][w][m], k, f"{a}|{w}|xseed", m); sb = topk_set(S[b][w][m], k, f"{b}|{w}|xseed", m)
                J.append(jac(sa, sb)); B.append(_exp_jac(n, k)); P.append(f"{a}|{b}"); M.append(m)
        A[f"{w}__jac"] = np.array(J, float); A[f"{w}__base"] = np.array(B, float); A[f"{w}__pair"] = np.array(P); A[f"{w}__mol"] = np.array(M)
    C.save_npz(f"{F}/cross/{cell.replace('|', '__')}.npz", **A); print(f"{cell}: cross-seed done", flush=True)


def n3(unit):
    sc = scale(); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); A = {"unit": unit}
    for name, tags in N3_TAGS.items():
        mols = mols_for(unit, sc[name]["level"]); R = {t: new_rows(unit, name, f"{T5.T5}/N3/expl/{name}/{t}/{unit}") for t in tags}
        base = new_rows(unit, name); vs, csize, kk = [], [], []
        for m in mols:
            k = C.kbudget(dl[m].num_nodes)
            sets = [topk_set(np.asarray(R[t][m]["node"], float), k, f"{unit}|{name}|{t}|jac", m) for t in tags]
            s0 = topk_set(np.asarray(base[m]["node"], float), k, f"{unit}|{name}|T5|jac", m)
            vs.append(np.mean([jac(s, s0) for s in sets])); csize.append(int((np.sum(sets, 0) >= 2).sum())); kk.append(k)
        A[f"{name}__mols"] = np.array(mols); A[f"{name}__vs_n1"] = np.array(vs); A[f"{name}__cons_size"] = np.array(csize); A[f"{name}__k"] = np.array(kk)
    C.save_npz(f"{F}/n3/{unit}.npz", **A); print(f"{unit}: n3 extras done", flush=True)


def det(unit):
    sc = scale(); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); A = {"unit": unit}
    for name in T5.NEW:
        rr = new_rows(unit, name, f"{F}/det/{name}/{unit}"); base = new_rows(unit, name); mols = sorted(rr)
        same, md, jc = [], [], []
        for m in mols:
            a = np.asarray(base[m]["node"], float); b = np.asarray(rr[m]["node"], float); k = C.kbudget(dl[m].num_nodes)
            same.append(bool(np.array_equal(a, b))); md.append(float(np.abs(a - b).max()))
            jc.append(jac(topk_set(a, k, f"{unit}|{name}|T5|jac", m), topk_set(b, k, f"{unit}|{name}|T5|jac", m)))
        A[f"{name}__mols"] = np.array(mols); A[f"{name}__identical"] = np.array(same); A[f"{name}__maxdiff"] = np.array(md); A[f"{name}__jaccard"] = np.array(jc)
        A[f"{name}__sec"] = np.array([rr[m]["seconds"] for m in mols], float)
    C.save_npz(f"{F}/det/{unit}.npz", **A); print(f"{unit}: determinism compared", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["extract", "cross", "n3", "det"]); a.add_argument("--unit"); a.add_argument("--cell")
    a = a.parse_args()
    if a.cmd == "cross":
        cross(a.cell)
    else:
        {"extract": extract, "n3": n3, "det": det}[a.cmd](a.unit)
