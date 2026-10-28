"""
RQ1 Tier 2, item M: GEA (final protocol) for the new seeds 5-9 of Mutagenicity D-MPNN and B-XAIC indole:
GNNExplainer (run_phase3 cache), PGExplainer (own float64 logit node scores), SubgraphX (scoring budget), IG,
heteroatom heuristic; shared nulls drawn for each new unit. Seeds 0-4 come from items H, G and S.
Writes runs/rq1_tier2/M/gea/<unit>.npz.

    python -m src.analysis.tier2_M UNIT ...
"""
from __future__ import annotations

import sys

import numpy as np

from . import tier2_common as C

T = f"{C.T2}/M"


def score(unit):
    sp = C.m_units()[unit]; dkey = C.dataset_of(unit); cache = C.torch_load(sp["cache"])
    ti = [int(m) for m in cache["key"]["test_idx"]]; gt = C.gt_mol_idx(unit, ti, dkey); nl = C.nulls(unit, gt, dkey)
    sx = C.sx_shard_scores(f"{T}/sx/{unit}")
    missing = [m for m in ti if m not in sx]
    if missing:
        raise RuntimeError(f"{unit}: {len(missing)} molecules without SubgraphX")
    src = {"gnnexplainer": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in zip(ti, cache["results"]["gnnexplainer"])},
           "pgexplainer": C.npz_scores(f"{T}/gp/{unit}_pg_logits.npz", "node_score64"), "sx_fixed": sx,
           "ig": C.npz_scores(f"{T}/grad/{unit}.npz", "ig"), "hetero": C.heuristic_dict(unit, ti, "hetero")}
    A = {"mol_idx": np.array(gt), "rn": nl["rn"], "rc": nl["rc"], "n_explained": len(ti)}
    for m, s in src.items():
        A[m] = C.gea_array(unit, m, lambda x, s=s: s[x], gt, dkey)
    C.save_npz(f"{T}/gea/{unit}.npz", unit=unit, **A)
    print(f"{unit}: {len(gt)} GT molecules of {len(ti)} explained", flush=True)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        score(u)
