"""
RQ1 Tier 2, item R: score the randomised-model explanations (runs/rq1_tier2/R/).

Per (unit, kind in full/head): GEA under the final protocol on the trained unit's GT molecules and shared nulls --
GNNExplainer (run_phase3 cache), PGExplainer (own float64 logit node scores), IG and Saliency (all GT molecules),
SubgraphX (scoring budget; the fixed 100-molecule subset). Trained vs randomised atom scores per GT molecule:
Spearman correlation (molecules with a constant score vector on either side are skipped and counted) and the
Jaccard of the two top-k sets (one fixed random order within ties on each side). Trained scores: GNNExplainer and
SubgraphX-legacy from the stored cache, PGExplainer final rule, IG / Saliency from item G, SubgraphX fixed budget
from item S (filled when it exists).
Writes runs/rq1_tier2/R/score/<unit>_<kind>.npz.

    python -m src.analysis.tier2_R UNIT ...
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import spearmanr

from . import tier2_common as C
from .ties_random import tie_order, topk_mask

T = f"{C.T2}/R"
METHODS = ["gnnexplainer", "pgexplainer", "subgraphx", "ig", "sal"]


def trained_scores(unit):
    pg, _ = C.pg_final(unit); g = f"{C.T2}/G/grad/{unit}.npz"
    out = {"gnnexplainer": C.stored_scores(unit, "gnnexplainer"), "pgexplainer": pg, "ig": C.npz_scores(g, "ig"), "sal": C.npz_scores(g, "sal")}
    d = f"{C.T2}/S/sx/{unit}"
    out["subgraphx"] = C.sx_shard_scores(d) if os.path.isdir(d) else {}
    return out


def rand_scores(unit, kind):
    cache = C.torch_load(f"{T}/gp/{kind}/{unit}_cache.pt")
    ti = [int(m) for m in cache["key"]["test_idx"]]
    assert ti == [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]], f"{unit} {kind}: explained set differs from the trained unit"
    g = f"{T}/grad/{kind}/{unit}.npz"
    return {"gnnexplainer": {m: np.asarray(r["node_importance"], dtype=np.float64) for m, r in zip(ti, cache["results"]["gnnexplainer"])},
            "pgexplainer": C.npz_scores(f"{T}/gp/{kind}/{unit}_pg_logits.npz", "node_score64"),
            "ig": C.npz_scores(g, "ig"), "sal": C.npz_scores(g, "sal"), "subgraphx": C.sx_shard_scores(f"{T}/sx/{kind}/{unit}")}


def score(unit, kind):
    dkey = C.dataset_of(unit); ti = [int(m) for m in C.stored_cache(unit)["key"]["test_idx"]]
    gt = C.gt_mol_idx(unit, ti, dkey); nl = C.nulls(unit, gt, dkey); pos = {m: i for i, m in enumerate(gt)}
    rs, tr = rand_scores(unit, kind), trained_scores(unit)
    A = {"unit": unit, "kind": kind}
    for m in METHODS:
        mols = [x for x in gt if x in rs[m]]
        if m == "subgraphx":
            assert len(mols) == 100 or len(mols) == len(gt), (unit, kind, len(mols))
        A[f"{m}_mol_idx"] = np.array(mols)
        A[f"{m}_gea"] = C.gea_array(f"{unit}|{kind}", m, lambda x, s=rs[m]: s[x], mols, dkey)
        A[f"{m}_rn"] = nl["rn"][[pos[x] for x in mols]]; A[f"{m}_rc"] = nl["rc"][[pos[x] for x in mols]]
        rho, jac = [], []
        for x in mols:
            if x not in tr[m]:
                rho.append(np.nan); jac.append(np.nan); continue
            a, b = np.asarray(tr[m][x], float), np.asarray(rs[m][x], float); k = C.kbudget(len(a))
            rho.append(np.nan if a.std() == 0 or b.std() == 0 else float(spearmanr(a, b).statistic))
            ta = topk_mask(tie_order(a, C.mol_rng(f"{unit}|{m}|trained-jac", x)), k); tb = topk_mask(tie_order(b, C.mol_rng(f"{unit}|{kind}|{m}|jac", x)), k)
            jac.append(float((ta & tb).sum() / max((ta | tb).sum(), 1)))
        A[f"{m}_spearman"] = np.array(rho); A[f"{m}_jaccard"] = np.array(jac)
    C.save_npz(f"{T}/score/{unit}_{kind}.npz", **A)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        for k in ("full", "head"):
            score(u, k); print(f"{u} {k}: scored", flush=True)
