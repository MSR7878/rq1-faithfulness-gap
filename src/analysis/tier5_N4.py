"""
RQ1 Tier 5, item N4 (runs/rq1_tier5/PLAN.md): combinations from cached scores, no new explanations.
Per unit, on the GT molecules that have SubgraphX: GEA (final protocol) of the six old methods, the three trivial rules, the
rank-average of (GNNExplainer, SubgraphX, IG) and of (GNNExplainer, SubgraphX), and IG smoothed over bonded neighbours.
    python -m src.analysis.tier5_N4 --unit U      -> runs/rq1_tier5/N4/<unit>.npz
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import rankdata

from . import tier2_common as C
from . import tier5_common as T5

OLD = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig", "ixg"]
COMBOS = ["rank3", "rank2", "ig_smooth"]


def rank_avg(score_lists):
    return np.mean([rankdata(np.asarray(s, dtype=np.float64), method="average") / len(s) for s in score_lists], 0)


def smooth(ig, edge_index):
    n = len(ig); tot = np.asarray(ig, dtype=np.float64).copy(); cnt = np.ones(n)
    for a, b in edge_index.T:                           # directed edges: each bonded neighbour b of a is added once
        tot[a] += ig[b]; cnt[a] += 1
    return tot / cnt


def run(unit):
    dkey = C.dataset_of(unit); dl, _ = C.load_data(dkey); src = T5.old_scores(unit)
    mols = [m for m in T5.gt_mols(unit) if m in src["sx_fixed"]]
    A = {"unit": unit, "mol_idx": np.array(mols), **T5.null_arrays(unit, mols)}
    for w in OLD:
        A[w] = C.gea_array(unit, w, lambda x, s=src[w]: s[x], mols, dkey)
    for h in ("hetero", "deg_high", "deg_low"):
        hd = C.heuristic_dict(unit, mols, h); A[h] = C.gea_array(unit, h, lambda x, s=hd: s[x], mols, dkey)
    comb = {"rank3": {m: rank_avg([src["gnnexplainer"][m], src["sx_fixed"][m], src["ig"][m]]) for m in mols},
            "rank2": {m: rank_avg([src["gnnexplainer"][m], src["sx_fixed"][m]]) for m in mols},
            "ig_smooth": {m: smooth(src["ig"][m], dl[m].edge_index.numpy()) for m in mols}}
    for w in COMBOS:
        A[w] = C.gea_array(unit, f"T5|{w}", lambda x, s=comb[w]: s[x], mols, dkey)
    # for P-N3: the old methods on all of the unit's GT molecules that have their scores (registered molecules; same streams)
    allgt = T5.gt_mols(unit); A["all__mols"] = np.array(allgt)
    for k, v in T5.null_arrays(unit, allgt).items():
        A[f"all__{k}"] = v
    for w in OLD:
        mw = [m for m in allgt if m in src[w]]
        A[f"all__{w}__mols"] = np.array(mw); A[f"all__{w}__gea"] = C.gea_array(unit, w, lambda x, s=src[w]: s[x], mw, dkey)
    C.save_npz(f"{T5.T5}/N4/{unit}.npz", **A)
    print(f"{unit}: {len(mols)} molecules with SubgraphX", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit"); run(a.parse_args().unit)
