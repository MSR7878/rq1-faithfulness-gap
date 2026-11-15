"""
RQ1 Tier 5, items N2 (sanity check) and N3 (stability) (runs/rq1_tier5/PLAN.md).

  n2   per unit (Mutagenicity D-MPNN / indole s0-4) and reset kind (full, head): GEA of the trained model's explanation (N1 run) and
       of the reset model's explanation (N2 run) on the same molecules; Spearman of the two atom-score vectors; Jaccard of the
       two top-k sets (one random order within boundary ties). Own-size sets (eig_own, same_own): GEA as sets and set Jaccard.
       -> runs/rq1_tier5/N2/score/<unit>.npz
  n3   per N3 unit and method with randomness: the runs T5a, T5b, T5c (EiG: T5a, T5b). Per molecule: mean pairwise Jaccard of
       the top-k sets, share of atoms picked in >= 2 of 3 runs, consensus set (>= 2 of 3) and single-run sets scored as sets,
       the final-protocol GEA of each run; EiG: max |score difference| between its two runs. -> runs/rq1_tier5/N3/score/<unit>.npz

    python -m src.analysis.tier5_N23 n2|n3 --unit U
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import spearmanr

from . import tier2_common as C
from . import tier5_common as T5
from .ties_random import boundary_tie, tie_order, topk_mask
from .tier5_N1 import new_rows, scale

N2_UNITS = [f"mutagenicity_dmpnn_s{s}" for s in range(5)] + [f"bxaic_indole_s{s}" for s in range(5)]
N3_UNITS = ["mutagenicity_dmpnn_s0", "bxaic_indole_s0", "mutag_graphxai_s0"]
KINDS = ["full", "head"]
N3_TAGS = {"eig": ["T5a", "T5b"], "same": ["T5a", "T5b", "T5c"], "gstarx": ["T5a", "T5b", "T5c"], "gnnshap": ["T5a", "T5b", "T5c"]}


def topk_set(s, k, stream, mi):
    s = np.asarray(s, dtype=np.float64)
    if boundary_tie(s, k):
        s = tie_order(s, C.mol_rng(stream, mi)).numpy()
    return topk_mask(s, k)


def jac(a, b):
    u = int((a | b).sum())
    return int((a & b).sum()) / u if u else 1.0


def mols_for(unit, level):
    return T5.level_mols(unit, "L1" if level == "L0" else level)


def n2(unit):
    sc = scale(); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); A = {"unit": unit}
    for name in T5.NEW:
        if name not in sc or sc[name].get("skipped"):
            continue
        mols = mols_for(unit, sc[name]["level"]); tr = new_rows(unit, name)
        for kind in KINDS:
            rs = new_rows(unit, name, f"{T5.T5}/N2/expl/{name}/{kind}/{unit}")
            miss = [m for m in mols if m not in rs or m not in tr]
            if miss:
                raise RuntimeError(f"{unit} {name} {kind}: {len(miss)} molecules missing")
            for w, key in ((name, "node"), (f"{name}_own", "own")):
                if key == "own" and name not in ("eig", "same"):
                    continue
                g_t, g_r, sp, jc = [], [], [], []
                for m in mols:
                    d = dl[m]; k = C.kbudget(d.num_nodes); scr = C.scorer_for(dkey, d, m)
                    a, b = np.asarray(tr[m][key], float), np.asarray(rs[m][key], float)
                    if key == "own":
                        sa, sb = a > 0.5, b > 0.5; g_t.append(scr(sa)); g_r.append(scr(sb))
                    else:
                        g_t.append(C.gea_random_ties(a, k, scr, f"{unit}|{w}|gea", m)); g_r.append(C.gea_random_ties(b, k, scr, f"{unit}|{w}|{kind}|gea", m))
                        sa, sb = topk_set(a, k, f"{unit}|{w}|jac", m), topk_set(b, k, f"{unit}|{w}|{kind}|jac", m)
                    sp.append(spearmanr(a, b).correlation if (a.max() > a.min() and b.max() > b.min()) else np.nan); jc.append(jac(sa, sb))
                for nm, v in (("trained", g_t), ("reset", g_r), ("spearman", sp), ("jaccard", jc)):
                    A[f"{w}__{kind}__{nm}"] = np.array(v, dtype=float)
                A[f"{w}__{kind}__mols"] = np.array(mols)
    C.save_npz(f"{T5.T5}/N2/score/{unit}.npz", **A); print(f"{unit}: N2 scored", flush=True)


def n3(unit):
    sc = scale(); dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); A = {"unit": unit}
    for name, tags in N3_TAGS.items():
        if name not in sc or sc[name].get("skipped"):
            continue
        mols = mols_for(unit, sc[name]["level"])
        R = {t: new_rows(unit, name, f"{T5.T5}/N3/expl/{name}/{t}/{unit}") for t in tags}
        J, S2, single, cons, prot, maxdiff = [], [], [], [], [], []
        for m in mols:
            d = dl[m]; n = d.num_nodes; k = C.kbudget(n); scr = C.scorer_for(dkey, d, m)
            sv = [np.asarray(R[t][m]["node"], float) for t in tags]
            sets = [topk_set(s, k, f"{unit}|{name}|{t}|jac", m) for s, t in zip(sv, tags)]
            J.append(np.mean([jac(sets[i], sets[j]) for i in range(len(tags)) for j in range(i + 1, len(tags))]))
            cnt = np.sum(sets, 0); picked = cnt > 0
            S2.append(float((cnt >= 2).sum() / picked.sum()) if picked.any() else np.nan)
            single.append(np.mean([scr(s) for s in sets])); cons.append(scr(cnt >= 2))
            prot.append(np.mean([C.gea_random_ties(s, k, scr, f"{unit}|{name}|{t}|gea", m) for s, t in zip(sv, tags)]))
            maxdiff.append(float(max(np.abs(sv[i] - sv[0]).max() for i in range(1, len(tags)))))
        for nm, v in (("jaccard", J), ("share_ge2", S2), ("single_set", single), ("consensus_set", cons), ("protocol", prot), ("maxdiff", maxdiff)):
            A[f"{name}__{nm}"] = np.array(v, dtype=float)
        A[f"{name}__mols"] = np.array(mols); A[f"{name}__rn"] = T5.null_arrays(unit, mols)["rn"]
    C.save_npz(f"{T5.T5}/N3/score/{unit}.npz", **A); print(f"{unit}: N3 scored", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["n2", "n3"]); a.add_argument("--unit"); a = a.parse_args()
    {"n2": n2, "n3": n3}[a.cmd](a.unit)
