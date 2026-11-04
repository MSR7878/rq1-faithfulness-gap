"""
RQ1 Tier 4, item L1: MUTAG-GT re-scored on all 188 explained molecules, with each seed's test molecules flagged, so the
report can compare test-only with all-188 (runs/rq1_tier4/PLAN.md). No new explanations.

  gea   GEA (final protocol) for 9 methods: GNNExplainer (stored), PGExplainer (float64 logits), SubgraphX fixed budget,
        Saliency, IG, Input x Gradient, hetero, high degree, low degree; Tier-2 nulls (read only).  -> L1/gea/<unit>.npz
  fid   Fid+ / Fid- / GEF under R1, R2, R3, both spaces, the same 9 methods + 20 random rankings.  -> L1/fid/<unit>.npz

    python -m src.analysis.tier4_L1 gea|fid --unit U
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from . import tier2_common as C
from . import tier4_common as T
from .revision_pass import all_specs
from .tier2_fid import run_fid

METHODS = ["gnnexplainer", "pgexplainer", "sx_fixed", "sal", "ig", "ixg", "hetero", "deg_high", "deg_low"]


def sources(unit, ti):
    g = f"{C.T2}/G/grad/{unit}.npz"
    return {"gnnexplainer": C.stored_scores(unit, "gnnexplainer"), "pgexplainer": T.pg64(unit), "sx_fixed": C.sx_shard_scores(f"{C.T2}/S/sx/{unit}"),
            "sal": C.npz_scores(g, "sal"), "ig": C.npz_scores(g, "ig"), "ixg": C.npz_scores(g, "ixg"),
            **{h: C.heuristic_dict(unit, ti, h) for h in ("hetero", "deg_high", "deg_low")}}


def test_set(unit):
    ck = torch.load(C.orig_spec(unit)["ckpt"], map_location="cpu", weights_only=False)
    return {int(i) for i in ck["split"]["test"]}


def gea(unit):
    dkey = C.dataset_of(unit); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; gt = C.gt_mol_idx(unit, ti, dkey)
    nl = T.tier2_nulls(unit, gt, dkey); src = sources(unit, ti); te = test_set(unit)
    missing = [m for m in ti for w in METHODS if m not in src[w]]
    if missing:
        raise RuntimeError(f"{unit}: {len(missing)} (molecule, method) scores missing")
    A = {"mol_idx": np.array(gt), "is_test": np.array([m in te for m in gt]), "rn": nl["rn"], "rc": nl["rc"]}
    for w in METHODS:
        A[w] = C.gea_array(unit, w, lambda x, s=src[w]: s[x], gt, dkey)
    C.save_npz(f"{T.T4}/L1/gea/{unit}.npz", unit=unit, **A)
    print(f"{unit}: L1 GEA on {len(gt)} GT molecules ({int(A['is_test'].sum())} test)", flush=True)


def fid(unit):
    dkey = C.dataset_of(unit); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]; src = sources(unit, ti)
    out = f"{T.T4}/L1/fid/{unit}.npz"
    run_fid(unit, C.orig_spec(unit)["ckpt"], all_specs()[unit]["out"], dkey, {w: src[w] for w in METHODS}, ti, out, with_randnode=True)
    z = dict(np.load(out)); te = test_set(unit)
    C.save_npz(out, **z, is_test=np.array([int(m) in te for m in z["mol_idx"]]))
    print(f"{unit}: L1 Fid on {len(ti)} molecules", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["gea", "fid"]); a.add_argument("--unit", required=True); a = a.parse_args()
    {"gea": gea, "fid": fid}[a.cmd](a.unit)
