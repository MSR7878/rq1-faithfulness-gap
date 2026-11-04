"""
RQ1 Tier 4, item L3: PGExplainer saturation on Mutagenicity (runs/rq1_tier4/PLAN.md). Per GT-bearing explained molecule:
whether the T4 re-run's float32 edge mask is constant (main split; the same run as the float64 logit scores) and whether the
stored float32 mask is (secondary); PGExplainer GEA ranked by the float64 logits and under the final rule; random-node null.

    python -m src.analysis.tier4_L3 --unit U        -> runs/rq1_tier4/L3/<unit>.npz
"""
from __future__ import annotations

import argparse

import numpy as np

from . import tier2_common as C
from . import tier4_common as T


def constant(edge_vals) -> bool:
    a = np.asarray(edge_vals, dtype=np.float32)
    return len(a) == 0 or bool(a.max() == a.min())


def run(unit):
    dkey = C.dataset_of(unit); blob = C.stored_cache(unit); ti = [int(m) for m in blob["key"]["test_idx"]]
    rr = C.torch_load(f"runs/rq1_revision/logits/rerun/{unit}_cache.pt")
    if [int(m) for m in rr["key"]["test_idx"]] != ti:
        raise RuntimeError(f"{unit}: re-run molecule list differs from the stored one")
    pos = {m: i for i, m in enumerate(ti)}; gt = C.gt_mol_idx(unit, ti, dkey)
    H = np.load(f"{C.T2}/H/gea/{unit}.npz")
    if [int(m) for m in H["mol_idx"]] != gt:
        raise RuntimeError(f"{unit}: Tier-2 H molecule list differs")
    pg = T.pg64(unit)
    A = {"mol_idx": np.array(gt), "const_rerun": np.array([constant(rr["results"]["pgexplainer"][pos[m]]["edge_importance"]) for m in gt]),
         "const_stored": np.array([constant(blob["results"]["pgexplainer"][pos[m]]["edge_importance"]) for m in gt]),
         "pg64": C.gea_array(unit, "pgexplainer", lambda x: pg[x], gt, dkey), "pg_final": H["pgexplainer"], "rn": H["rn"], "rc": H["rc"]}
    C.save_npz(f"{T.T4}/L3/{unit}.npz", unit=unit, **A)
    print(f"{unit}: {len(gt)} GT molecules, constant (re-run) {int(A['const_rerun'].sum())}, constant (stored) {int(A['const_stored'].sum())}", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--unit", required=True); run(a.parse_args().unit)
