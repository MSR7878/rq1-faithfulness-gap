"""
RQ1 Tier 4, items M1 and M2 (runs/rq1_tier4/PLAN.md).

  prep    M1: the B-XAIC D-MPNN splits as json (train.py --split-json) and the fixed 100-GT-molecule SubgraphX subsets
          (runs/rq1_tier4/subsets.json; indole checked against the Tier-2 subsets).
  check   M1: every new unit explained exactly the D-MPNN unit's molecules; exits 1 otherwise.
  score   GEA (final protocol) on the GT-bearing explained molecules: GNNExplainer, PGExplainer (own float64 logits), Saliency, IG,
          Input x Gradient, hetero, high degree, low degree; SubgraphX fixed budget (M1: on the 100-molecule subset; M2: all).
          Nulls: M1 = the D-MPNN unit's Tier-2 null file (read only, same molecules); M2 = runs/rq1_tier4/nulls.
          Also the clean test metrics of the model and whether it meets the bar.   -> runs/rq1_tier4/<M1|M2>/gea/<unit>.npz

    python -m src.analysis.tier4_M prep|check
    python -m src.analysis.tier4_M score --unit U
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import tier2_common as C
from . import tier4_common as T

METHODS = ["gnnexplainer", "pgexplainer", "sal", "ig", "ixg", "hetero", "deg_high", "deg_low"]


def prep():
    for t in T.M1_TASKS:
        for s in range(5):
            T.export_split(f"bxaic_{t}_s{s}")
    S = T.m1_subsets(); print({u: len(v) for u, v in S.items()}, flush=True)


def check():
    bad = []
    for u in T.m1_units():
        if T.test_idx_of(u) != [int(m) for m in C.stored_cache(T.dmpnn_of(u))["key"]["test_idx"]]:
            bad.append(u)
    print(f"M1 molecule check: {len(T.m1_units()) - len(bad)} of {len(T.m1_units())} units explain the D-MPNN's molecules" + (f"; differ: {bad}" if bad else ""), flush=True)
    return not bad


def score(unit):
    m1 = unit in T.m1_units(); item = "M1" if m1 else "M2"; D = f"{T.T4}/{item}"; dkey = C.dataset_of(unit); sp = C.orig_spec(unit)
    ti = T.test_idx_of(unit); gt = C.gt_mol_idx(unit, ti, dkey)
    if m1:
        dm = T.dmpnn_of(unit)
        if ti != [int(m) for m in C.stored_cache(dm)["key"]["test_idx"]]:
            raise RuntimeError(f"{unit}: explained molecules differ from {dm}")
        nl = T.tier2_nulls(dm, gt, dkey); sx_mols = T.m1_subsets()[dm]
    else:
        nl = T.nulls4(unit, gt, dkey); sx_mols = gt
    g = f"{D}/grad/{unit}.npz"
    src = {"gnnexplainer": T.cache_scores(sp["cache"], "gnnexplainer"), "pgexplainer": T.pg64(unit), "sal": C.npz_scores(g, "sal"),
           "ig": C.npz_scores(g, "ig"), "ixg": C.npz_scores(g, "ixg"), **{h: C.heuristic_dict(unit, ti, h) for h in ("hetero", "deg_high", "deg_low")}}
    sx = C.sx_shard_scores(f"{D}/sx/{unit}")
    missing = [m for m in sx_mols if m not in sx]
    if missing:
        raise RuntimeError(f"{unit}: {len(missing)} molecules without SubgraphX")
    ck = torch.load(sp["ckpt"], map_location="cpu", weights_only=False); met = ck["metrics"]
    A = {"mol_idx": np.array(gt), "rn": nl["rn"], "rc": nl["rc"], "n_explained": len(ti), "sx_mol_idx": np.array(sx_mols),
         "test_acc": met["test_acc"], "test_auroc": met["test_auroc"], "meets_bar": met["test_acc"] >= T.BAR_ACC and met["test_auroc"] >= T.BAR_AUROC}
    for w in METHODS:
        A[w] = C.gea_array(unit, w, lambda x, s=src[w]: s[x], gt, dkey)
    A["sx_fixed"] = C.gea_array(unit, "sx_fixed", lambda x: sx[x], sx_mols, dkey)
    C.save_npz(f"{D}/gea/{unit}.npz", unit=unit, **A)
    print(f"{unit}: {len(gt)} GT molecules of {len(ti)} explained; SubgraphX on {len(sx_mols)}; test acc {met['test_acc']:.3f} AUROC {met['test_auroc']:.3f}"
          f"{'' if A['meets_bar'] else ' (MISSES THE BAR)'}", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["prep", "check", "score"]); a.add_argument("--unit"); a = a.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "check":
        sys.exit(0 if check() else 1)
    else:
        score(a.unit)
